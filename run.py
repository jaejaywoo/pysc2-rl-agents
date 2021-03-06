import sys
import os
import shutil
import sys
from math import isnan
import argparse
from functools import partial
from slacker import Slacker
from numpy import random

import tensorflow as tf
from tensorflow.python import debug as tfdbg

from rl.agents.a2c.runner import A2CRunner
from rl.agents.a2c.agent import A2CAgent
from rl.networks.fully_conv import FullyConv
from rl.environment import SubprocVecEnv, make_sc2env, SingleEnv
from rl.util import set_global_seeds, send_notification

# Workaround for pysc2 flags
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['run.py'])


parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')
parser.add_argument('experiment_id', type=str,
                    help='identifier to store experiment results')
parser.add_argument('--eval', action='store_true',
                    help='if false, episode scores are evaluated')
parser.add_argument('--load', action='store_true',
                    help='if true, loads the model from last ckpt and continue training')
parser.add_argument('--debug', action='store_true',
                    help='if true, check for undefined numerics')
parser.add_argument('--tfdbg', action='store_true',
                    help='if true, deploy Tensorflow debugger wrapper')
parser.add_argument('--ow', action='store_true',
                    help='overwrite existing experiments (if --train=True)')
parser.add_argument('--map', type=str, default='MoveToBeacon',
                    help='name of SC2 map')
parser.add_argument('--lstm', action='store_true',
                    help='if true, implement FullyConvLSTM policy')
parser.add_argument('--vis', action='store_true',
                    help='render with pygame')
parser.add_argument('--max_windows', type=int, default=1,
                    help='maximum number of visualization windows to open')
parser.add_argument('--res', type=int, default=32,
                    help='screen and minimap resolution')
parser.add_argument('--envs', type=int, default=32,
                    help='number of environments simulated in parallel')
parser.add_argument('--step_mul', type=int, default=8,
                    help='number of game steps per agent step')
parser.add_argument('--steps_per_batch', type=int, default=16,
                    help='number of agent steps when collecting trajectories for a single batch')
parser.add_argument('--discount', type=float, default=0.99,
                    help='discount for future rewards')
parser.add_argument('--num_timesteps', type=int, default=int(50e6),
                    help='total number of game steps to run')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu device id')
parser.add_argument('--nchw', action='store_true',
                    help='train fullyConv in NCHW mode')
parser.add_argument('--summary_iters', type=int, default=10,
                    help='record training summary after this many iterations')
parser.add_argument('--save_iters', type=int, default=200,
                    help='store checkpoint after this many iterations')
parser.add_argument('--max_to_keep', type=int, default=4,
                    help='maximum number of checkpoints to keep before discarding older ones')
parser.add_argument('--entropy_weight', type=float, default=1e-3,
                    help='weight of entropy loss')
parser.add_argument('--value_loss_weight', type=float, default=0.5,
                    help='weight of value function loss')
parser.add_argument('--lr', type=float, default=0.00095,
                    help='initial learning rate')
parser.add_argument('--save_dir', type=str, default=os.path.join('out','models'),
                    help='root directory for checkpoint storage')
parser.add_argument('--summary_dir', type=str, default=os.path.join('out','summary'),
                    help='root directory for summary storage')

args = parser.parse_args()
# TODO write args to config file and store together with summaries (https://pypi.python.org/pypi/ConfigArgParse)
args.train = not args.eval

# set random seed
if args.seed:
  set_global_seeds(args.seed)

# Disable Tensorflow WARNING log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Specify model type
if args.lstm:
  dir_name = '-'.join([args.experiment_id, 'lstm', 'lr%0.5f'%(args.lr)])
else:
  dir_name = '-'.join([args.experiment_id, 'lr%0.5f'%(args.lr)])

ckpt_path = os.path.join(args.save_dir, dir_name)
summary_type = 'train' if args.train else 'eval'
summary_base = os.path.join(args.summary_dir, dir_name, summary_type)

# save model every 10 iterations when on debug mode
if args.debug:
  args.max_to_keep = 5
  args.save_iters = 1


def _save_if_training(agent, summary_writer):
  if args.train:
    agent.save(ckpt_path)
    summary_writer.flush()
    sys.stdout.flush()


def main():
    # Get token for Slacker
    token = os.environ['SLACK_API_TOKEN']
    slack = Slacker(token=token)

    # Create subdirs for each run in experiment
    if os.path.exists(summary_base):
        run_dirs = [os.path.join(summary_base, d) for d in os.listdir(summary_base)]
        latest_run = int(max(run_dirs, key=os.path.getmtime).split('-')[-1])
        summary_path = os.path.join(summary_base, 'run-%d'%(latest_run + 1))
    else:
        summary_path = os.path.join(summary_base, 'run-1')

    # Overwrite experiment summaries
    if args.train and args.ow:
      shutil.rmtree(ckpt_path, ignore_errors=True)
      shutil.rmtree(summary_path, ignore_errors=True)

    # Create envs
    size_px = (args.res, args.res)
    env_args = dict(
        map_name=args.map,
        step_mul=args.step_mul,
        game_steps_per_episode=0,
        screen_size_px=size_px,
        minimap_size_px=size_px)
    vis_env_args = env_args.copy()
    vis_env_args['visualize'] = args.vis
    num_vis = min(args.envs, args.max_windows)
    env_fns = [partial(make_sc2env, **vis_env_args)] * num_vis
    num_no_vis = args.envs - num_vis
    if num_no_vis > 0:
      env_fns.extend([partial(make_sc2env, **env_args)] * num_no_vis)
    envs = SubprocVecEnv(env_fns)

    # Start tensorflow session
    if not args.lstm:
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.475)
      sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
      sess = tf.Session()
    if args.tfdbg:
      sess = tfdbg.LocalCLIDebugWrapperSession(sess)

    summary_writer = tf.summary.FileWriter(summary_path)
    network_data_format = 'NCHW' if args.nchw else 'NHWC'  # XXX NHWC -> NCHW

    # Create A2CAgent instance
    agent = A2CAgent(
        sess=sess,
        debug=args.debug,
        network_data_format=network_data_format,
        value_loss_weight=args.value_loss_weight,
        entropy_weight=args.entropy_weight,
        learning_rate=args.lr,
        max_to_keep=args.max_to_keep,
        lstm=args.lstm)

    # Setup A2CAgent runner
    runner = A2CRunner(
        envs=envs,
        agent=agent,
        slack=slack,
        train=args.train,
        summary_writer=summary_writer,
        discount=args.discount,
        n_steps=args.steps_per_batch)

    # Build A2CAgent graphs
    static_shape_channels = runner.preproc.get_input_channels()
    agent.build(static_shape_channels, resolution=args.res)

    # Load the latest ckpt
    if os.path.exists(ckpt_path):
      agent.load(ckpt_path)
      load = True
    else:
      agent.init()
      load = False

    runner.reset(nenvs=args.envs, res=args.res)

    # Start Train/Eval
    i = agent.train_step if load else 0
    total_frames = 0
    try:
      while True:
        write_summary = args.train and i % args.summary_iters == 0

        if i > 0 and i % args.save_iters == 0:
          _save_if_training(agent, summary_writer)

        result, total_frames = runner.run_batch(total_frames, train_summary=write_summary, lstm=args.lstm)

        # Debug return
        if args.debug and result == None:
          warning = 'Bad numerics detected by Tensorflow API!' + \
                    'Stopping the environment...'
          send_notification(slack, message=warning, channel='#sc2')
          break

        if write_summary:
          agent_step, loss, summary = result
          summary_writer.add_summary(summary, global_step=agent_step)
          print('iter %d: loss = %f' % (agent_step, loss))

        #if args.train and isnan(loss):
        #  warning = 'NaN output detected from loss!' + \
        #            'Stopping the SC2 environment...'
        #  print(warning)
        #  send_notification(slack, message=warning, channel='#sc2')
        #  break

        i += 1

        if 0 <= args.num_timesteps <= total_frames:
          log = 'Agent has finished training! Saving and closing the SC2 environment..'
          print(log)
          send_notification(slack, message=log, channel='#sc2')
          break

    except KeyboardInterrupt:
        pass

    # Save the model ckpt
    _save_if_training(agent, summary_writer)

    envs.close()
    summary_writer.close()

    print('mean score: %f' % runner.get_mean_score())


if __name__ == "__main__":
    main()
