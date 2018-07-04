import numpy as np
import tensorflow as tf


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
  return tf.where(
      tf.equal(x, 0),
      tf.zeros_like(x),
      tf.log(tf.maximum(1e-12, x)))


def has_nan_or_inf(datum, tensor):
  """Returns true if the tensor contains NaN or inf."""
  return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))


def send_notification(slack, message, channel):
  """Send notification to Slack channel (i.e. sc2)."""
  slack.chat.post_message(channel=channel, text=message, username="sc2 bot")
