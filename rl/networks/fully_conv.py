import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from pysc2.lib import actions
from pysc2.lib import features

from rl.pre_processing import is_spatial_action, NUM_FUNCTIONS, FLAT_FEATURES


class FullyConv():
  """FullyConv network from https://arxiv.org/pdf/1708.04782.pdf.

  Both, NHWC and NCHW data formats are supported for the network
  computations. Inputs and outputs are always in NHWC.
  """

  def __init__(self, data_format='NCHW', lstm=False):
    self.data_format = data_format
    self.lstm = lstm

  def embed_obs(self, x, spec, embed_fn, name):
    feats = tf.split(x, len(spec), -1)
    out_list = []
    for i, s in enumerate(spec):
      f = feats[s.index]
      if s.type == features.FeatureType.CATEGORICAL:
        dims = np.round(np.log2(s.scale)).astype(np.int32).item()
        dims = max(dims, 1)
        indices = tf.one_hot(tf.to_int32(tf.squeeze(f, -1)), s.scale)
        out = embed_fn(indices, dims, name, i)
      elif s.type == features.FeatureType.SCALAR:
        out = self.log_transform(f, s.scale)
      else:
        raise NotImplementedError
      out_list.append(out)
    return tf.concat(out_list, -1)

  def log_transform(self, x, scale):
    return tf.log(x + 1.)

  def embed_spatial(self, x, dims, name, i):
      x = self.from_nhwc(x)
      out = layers.conv2d(
          x, dims,
          kernel_size=1,
          stride=1,
          padding='SAME',
          activation_fn=tf.nn.relu,
          data_format=self.data_format,
          scope="embed-%s/conv_%d"%(name,i))
      return self.to_nhwc(out)

  def embed_flat(self, x, dims, name, i):
      return layers.fully_connected(
          x, dims,
          activation_fn=tf.nn.relu,
          scope="embed-%s/fc_%d"%(name,i))

  def input_conv(self, x, name):
      conv1 = layers.conv2d(
          x, 16,
          kernel_size=5,
          stride=1,
          padding='SAME',
          activation_fn=tf.nn.relu,
          data_format=self.data_format,
          scope="%s/conv1"%name)
      conv2 = layers.conv2d(
          conv1, 32,
          kernel_size=3,
          stride=1,
          padding='SAME',
          activation_fn=tf.nn.relu,
          data_format=self.data_format,
          scope="%s/conv2"%name)
      return conv2

  def non_spatial_output(self, x, channels, name):
      logits = layers.fully_connected(
          x, channels,
          activation_fn=None,
          scope="non-spatial-%s/fc"%name)
      return tf.nn.softmax(logits)

  def spatial_output(self, x, name):
      logits = layers.conv2d(
          x, 1,
          kernel_size=1,
          stride=1,
          activation_fn=None,
          data_format=self.data_format,
          scope="spatial-%s/conv"%name)
      logits = layers.flatten(
          self.to_nhwc(logits),
          scope="spatial-%s/flatten"%name)
      return tf.nn.softmax(logits)

  def concat2d(self, lst):
    if self.data_format == 'NCHW':
        return tf.concat(lst, axis=1)
    return tf.concat(lst, axis=3)

  def broadcast_along_channels(self, flat, size2d):
    if self.data_format == 'NCHW':
        return tf.tile(tf.expand_dims(tf.expand_dims(flat, 2), 3),
                       tf.stack([1, 1, size2d[0], size2d[1]]))
    return tf.tile(tf.expand_dims(tf.expand_dims(flat, 1), 2),
                   tf.stack([1, size2d[0], size2d[1], 1]))

  def to_nhwc(self, map2d):
    if self.data_format == 'NCHW':
        return tf.transpose(map2d, [0, 2, 3, 1])
    return map2d

  def from_nhwc(self, map2d):
    if self.data_format == 'NCHW':
        return tf.transpose(map2d, [0, 3, 1, 2])
    return map2d

  def build(self, screen_input, minimap_input, flat_input, lstm_state_in=None):
    """Build A2C policy model.

    Args:
        screen_input: [None, 32, 32, 17]
        minimap_input: [None, 32, 32, 7]
        flat_input: [None, 11]
        """
    size2d = tf.unstack(tf.shape(screen_input)[1:3])
    screen_emb = self.embed_obs(screen_input, features.SCREEN_FEATURES,  # NHWC: [None, 32, 32, 35]
                                self.embed_spatial, "screen")
    minimap_emb = self.embed_obs(minimap_input, features.MINIMAP_FEATURES,  # NHWC: [None, 32, 32, 12]
                                 self.embed_spatial, "minimap")
    flat_emb = self.embed_obs(flat_input, FLAT_FEATURES, self.embed_flat, "flat")  # NHWC: [None, 11]

    # conv/spatial obs
    screen_out = self.input_conv(self.from_nhwc(screen_emb), 'screen')     # NCHW: [None, 32, 32, 32]
    minimap_out = self.input_conv(self.from_nhwc(minimap_emb), 'minimap')  # NCHW: [None, 32, 32, 32]

    # broadcast/non-spatial obs
    broadcast_out = self.broadcast_along_channels(flat_emb, size2d)        # NCHW: [None, 11, None, None]

    # spatial + non-spatial
    state_out = self.concat2d([screen_out, minimap_out, broadcast_out])    # NCHW: [None, 75, 32, 32]

    # convolutional LSTM
    if self.lstm:
      lstm_in = tf.reshape(self.to_nhwc(state_out), [1, -1, 32, 32, 75])
      lstm = tf.contrib.rnn.Conv2DLSTMCell(
        input_shape=[32, 32, 1],
        kernel_shape=[3, 3],
        output_channels=75,
        name='conv_lstm')
      c_init = np.zeros([1, 32, 32, 75], np.float32)
      h_init = np.zeros([1, 32, 32, 75], np.float32)
      self.state_init = [c_init, h_init]

      # TODO: step_size of 16? What is the sequence here?
      step_size = tf.shape(state_out)[:1]  # Get step_size from input dimension
      c_in, h_in = lstm_state_in
      state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
      self.step_size = tf.placeholder(tf.float32, [1])

      outputs, state = tf.nn.dynamic_rnn(
          cell=lstm,
          inputs=lstm_in,
          sequence_length=step_size,  # XXX Maybe optional?
          initial_state=state_in,
          time_major=False,
          dtype=tf.float32)

      lstm_c, lstm_h = state
      lstm_state_out = (lstm_c[:1, :], lstm_h[:1, :])
      flat_out = tf.reshape(outputs, [-1, 76800], name="state/flatten")  # 32*32*75
      lstm_out = tf.reshape(outputs, [-1, 32, 32, 75])
    else:
      flat_out = layers.flatten(self.to_nhwc(state_out), scope="state/flatten")
    fc = layers.fully_connected(flat_out, 256, activation_fn=tf.nn.relu,
                                scope="state/fc")   # [None, 256]

    # fc/value estimate
    value = layers.fully_connected(fc, 1, activation_fn=None,
                                   scope="value/fc")                       # [None, 1]
    value = tf.reshape(value, [-1], name="value")                          # [None]

    # fc/action_id
    fn_out = self.non_spatial_output(fc, NUM_FUNCTIONS, "action_id")       # [None, 524]

    # arguments
    args_out = dict()
    for arg_type in actions.TYPES:
      if is_spatial_action[arg_type]:
        if self.lstm:
          arg_out = self.spatial_output(self.from_nhwc(lstm_out),
                                        arg_type.name)                     # [None, 1024]
        else:
          arg_out = self.spatial_output(state_out, arg_type.name)          # [None, 1024]
      else:
        arg_out = self.non_spatial_output(fc, arg_type.sizes[0],
                                          arg_type.name)                   # [None, 2/4/5/10/500]
      args_out[arg_type] = arg_out

    policy = (fn_out, args_out)

    if self.lstm:
      return policy, value, lstm_state_out
    return policy, value
