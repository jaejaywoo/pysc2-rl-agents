import numpy as np
import tensorflow as tf


def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def tf_print(tensor, name):
  return tf.Print(tensor, [tensor],
      summarize=10000,
      message='{} tensor:\n'.format(name))


def safe_div(numerator, denominator, name="value"):
  """Computes a safe divide which returns 0 if the denominator is zero.
  Note that the function contains an additional conditional check that is
  necessary for avoiding situations where the loss is zero causing NaNs to
  creep into the gradient computation.
  Args:
    numerator: An arbitrary `Tensor`.
    denominator: `Tensor` whose shape matches `numerator` and whose values are
      assumed to be non-negative.
    name: An optional name for the returned op.
  Returns:
    The element-wise value of the numerator divided by the denominator.
  """
  return tf.where(
      tf.greater(denominator, 0),
      tf.div(numerator, tf.where(
          tf.equal(denominator, 0),
          tf.ones_like(denominator), denominator)),
      tf.zeros_like(numerator),
      name=name)


def safe_log(x):
  """Computes a safe logarithm which returns 0 if x is zero."""
  result = tf.where(
      tf.equal(x, 0),
      tf.zeros_like(x),
      tf.log(tf.maximum(1e-12, x)))
  #result = tf.cond(
  #    tf.reduce_any(tf.is_nan(result)),
  #    true_fn=lambda: tf.Print(result, [result], message="NaN detected at 'pysc2-rl-agents/rl/util.py/safe_log'\n"),
  #    false_fn=lambda: result)
  #result = tf.Print(result, [result], summarize=16*524, message="safe_log tensors:\n")
  return result


def has_nan_or_inf(datum, tensor):
  """Returns true if the tensor contains NaN or inf."""
  return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))


def send_notification(slack, message, channel):
  """Send notification to Slack channel (i.e. sc2)."""
  slack.chat.post_message(channel=channel, text=message, username="sc2 bot")
