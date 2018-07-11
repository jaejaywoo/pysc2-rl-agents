import numpy as np
import tensorflow as tf

from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.lib.actions import TYPES as ACTION_TYPES

from rl.pre_processing import Preprocessor
from rl.pre_processing import is_spatial_action, stack_ndarray_dicts
from rl.util import send_notification

class A2CRunner():
  def __init__(self,
               agent,
               envs,
               slack,
               summary_writer=None,
               train=True,
               n_steps=8,
               discount=0.99):
    """
    Args:
      agent: A2CAgent instance.
      envs: SubprocVecEnv instance.
      summary_writer: summary writer to log episode scores.
      train: whether to train the agent.
      n_steps: number of agent steps for collecting rollouts.
      discount: future reward discount.
    """
    self.agent = agent
    self.envs = envs
    self.slack = slack
    self.summary_writer = summary_writer
    self.train = train
    self.n_steps = n_steps
    self.discount = discount
    self.preproc = Preprocessor(self.envs.observation_spec()[0])
    self.episode_counter = 0
    self.best_n_mean_counter = 0
    self.mean_score = 0.0
    self.cumulative_score = 0.0

  def reset(self):
    obs_raw = self.envs.reset()
    self.last_obs = self.preproc.preprocess_obs(obs_raw)
    self.episode_last = [None for i in range(self.envs.n_envs)]
    self.worker_scores = [0 for i in range(self.envs.n_envs)]

  def get_mean_score(self):
    return self.cumulative_score / self.episode_counter

  def _print_agent_actions(self, actions):
    fn_ids, arg_ids = actions
    id_str = " ".join(map(str, fn_ids))
    print("episode %d | Sampled action IDs: " % (self.episode_counter) + id_str)

  def _summarize_episode(self, timestep, worker_id=None):
    score = timestep.observation["score_cumulative"][0]
    self.worker_scores[worker_id] = score

    if self.summary_writer is not None:
      summary = tf.Summary()
      summary.value.add(tag='sc2/episode_score', simple_value=score)
      self.summary_writer.add_summary(summary, self.episode_counter)

    print("episode %d: score = %f" % (self.episode_counter, score))
    self.episode_counter += 1
    return score

  def _summarize_best_and_mean(self):
    mean_score = self.mean_score / self.envs.n_envs
    best_score = max(self.worker_scores)
    if self.summary_writer is not None:
      summary = tf.Summary()
      summary.value.add(tag='sc2/mean_score', simple_value=mean_score)
      summary.value.add(tag='sc2/best_score', simple_value=best_score)
      self.summary_writer.add_summary(summary, self.best_n_mean_counter)

    print("step %d: MEAN SCORE = %f" % (self.best_n_mean_counter, mean_score))
    print("step %d: BEST SCORE = %f" % (self.best_n_mean_counter, best_score))
    self.episode_last = [None for i in range(self.envs.n_envs)]
    self.mean_score = 0
    self.best_n_mean_counter += 1

  def run_batch(self, train_summary=False, lstm=False):
    """Collect trajectories for a single batch and train (if self.train).

    Args:
      train_summary: return a Summary of the training step (losses, etc.).

    Returns:
      result: None (if not self.train) or the return value of agent.train.
    """
    shapes = (self.n_steps, self.envs.n_envs)
    values = np.zeros(shapes, dtype=np.float32)
    rewards = np.zeros(shapes, dtype=np.float32)
    dones = np.zeros(shapes, dtype=np.float32)
    all_obs = []
    all_actions = []
    all_scores = []  # TODO: Unused local var?

    # TODO: Why do you save last_obs?
    last_obs = self.last_obs
    lstm_state = self.agent.lstm_state_init if lstm else None

    for n in range(self.n_steps):
      actions, value_estimate, lstm_state = self.agent.step(last_obs, lstm_state)
      actions = mask_unused_argument_samples(actions)
      size = last_obs['screen'].shape[1:3]

      values[n, :] = value_estimate
      all_obs.append(last_obs)
      all_actions.append(actions)

      pysc2_actions = actions_to_pysc2(actions, size)
      obs_raw = self.envs.step(pysc2_actions)
      last_obs = self.preproc.preprocess_obs(obs_raw)
      rewards[n, :] = [t.reward for t in obs_raw]
      dones[n, :] = [t.last() for t in obs_raw]

      for i, t in enumerate(obs_raw):
        if t.last():
          score = self._summarize_episode(t, worker_id=i)
          self.cumulative_score += score
          self.mean_score += score
          self.episode_last[i] = t.last()

    # Get episode mean score of workers
    if all(self.episode_last):
      self._summarize_best_and_mean()

    self.last_obs = last_obs
    next_values = self.agent.get_value(last_obs, lstm_state)

    returns, advs = compute_returns_advantages(
        rewards, dones, values, next_values, self.discount)

    actions = stack_and_flatten_actions(all_actions)
    obs = flatten_first_dims_dict(stack_ndarray_dicts(all_obs))
    returns = flatten_first_dims(returns)
    advs = flatten_first_dims(advs)

    if self.train:
      return self.agent.train(
          obs, actions, returns, advs,
          summary=train_summary, lstm_state=lstm_state
      )

    return None


def compute_returns_advantages(rewards, dones, values, next_values, discount):
  """Compute returns and advantages from received rewards and value estimates.

  Args:
    rewards: array of shape [n_steps, n_env] containing received rewards.
    dones: array of shape [n_steps, n_env] indicating whether an episode is
      finished after a time step.
    values: array of shape [n_steps, n_env] containing estimated values.
    next_values: array of shape [n_env] containing estimated values after the
      last step for each environment.
    discount: scalar discount for future rewards.

  Returns:
    returns: array of shape [n_steps, n_env]
    advs: array of shape [n_steps, n_env]
  """
  returns = np.zeros([rewards.shape[0] + 1, rewards.shape[1]])

  returns[-1, :] = next_values
  for t in reversed(range(rewards.shape[0])):
    future_rewards = discount * returns[t + 1, :] * (1 - dones[t, :])
    returns[t, :] = rewards[t, :] + future_rewards

  returns = returns[:-1, :]
  advs = returns - values

  return returns, advs


def actions_to_pysc2(actions, size):
  """Convert agent action representation to FunctionCall representation."""
  height, width = size
  fn_id, arg_ids = actions
  actions_list = []
  for n in range(fn_id.shape[0]):
    a_0 = fn_id[n]
    a_l = []

    for arg_type in FUNCTIONS._func_list[a_0].args:
      arg_id = arg_ids[arg_type][n]
      if is_spatial_action[arg_type]:
        arg = [arg_id % width, arg_id // height]
      else:
        arg = [arg_id]
      a_l.append(arg)
    action = FunctionCall(a_0, a_l)
    actions_list.append(action)
  return actions_list


def mask_unused_argument_samples(actions):
  """Replace sampled argument id by -1 for all arguments not used
  in a steps action (in-place).
  """
  fn_id, arg_ids = actions
  for n in range(fn_id.shape[0]):
    a_0 = fn_id[n]

    unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[a_0].args)
    for arg_type in unused_types:
      arg_ids[arg_type][n] = -1
  return (fn_id, arg_ids)


def flatten_first_dims(x):
  new_shape = [x.shape[0] * x.shape[1]] + list(x.shape[2:])
  return x.reshape(*new_shape)


def flatten_first_dims_dict(x):
  return {k: flatten_first_dims(v) for k, v in x.items()}


def stack_and_flatten_actions(lst, axis=0):
  fn_id_list, arg_dict_list = zip(*lst)
  fn_id = np.stack(fn_id_list, axis=axis)
  fn_id = flatten_first_dims(fn_id)
  arg_ids = stack_ndarray_dicts(arg_dict_list, axis=axis)
  arg_ids = flatten_first_dims_dict(arg_ids)
  return (fn_id, arg_ids)
