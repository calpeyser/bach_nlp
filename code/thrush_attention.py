import tensorflow as tf
import os
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


class AttentionCell(Layer):

    def __init__(self, config=None, **kwargs):
        super(AttentionCell, self).__init__(**kwargs)
        self.loaded_from_config = False
        if config:
          self.loaded_from_config = True
          self.W_a = tf.Variable(config['W_a'])
          self.U_a = tf.Variable(config['U_a'])
          self.V_a = tf.Variable(config['V_a'])

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        if not self.loaded_from_config:
          self.W_a = self.add_weight(name='W_a',
                                    shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                    initializer='uniform',
                                    trainable=True)
          self.U_a = self.add_weight(name='U_a',
                                    shape=tf.TensorShape((input_shape[1][1], input_shape[0][2])),
                                    initializer='uniform',
                                    trainable=True)
          self.V_a = self.add_weight(name='V_a',
                                    shape=tf.TensorShape((input_shape[0][2], 1)),
                                    initializer='uniform',
                                    trainable=True)

        super(AttentionCell, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)
            print('W_a>', self.W_a.shape)
            print('U_a>', self.U_a.shape)
            print('V_a>', self.V_a.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)
            """

            assert_msg = "States must be an iterable. Got {} of type {}".format(
                states, type(states))
            assert isinstance(states, list) or isinstance(
                states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch size * en_seq_len * latent_dim
            W_a_dot_s = K.dot(encoder_out_seq, self.W_a)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(
                K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>', U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)
            if verbose:
                print('Ws+Uh>', Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """

            assert_msg = "States must be an iterable. Got {} of type {}".format(
                states, type(states))
            assert isinstance(states, list) or isinstance(
                states, tuple), assert_msg

            e_i = inputs[0]
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * tf.expand_dims(e_i, axis=-1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        fake_state_c = K.sum(encoder_out_seq, axis=1)
        # <= (batch_size, enc_seq_len, latent_dim
        fake_state_e = K.sum(encoder_out_seq, axis=2)

        last_out, e_outputs = energy_step(decoder_out_seq, [fake_state_e])
        last_out, c_outputs = context_step(e_outputs, [fake_state_c])

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape(
                (input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape(
                (input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

    def get_config(self):
        return {
            'W_a': self.W_a.numpy(),
            'U_a': self.U_a.numpy(),
            'V_a': self.V_a.numpy()
        }

    @classmethod
    def from_config(cls, config):
        return cls(config=config)


class DecoderLayer(Layer):

  def __init__(self, units, constants, teacher_forcing_prob=1.0, config=None):
    super(DecoderLayer, self).__init__()
    self.units = units
    self.teacher_forcing_prob=teacher_forcing_prob
    self.constants = constants
    self.lstm_input_concat_cell = keras.layers.Concatenate(name='decoder_lstm_input_concat')
    self.attn_concat_cell = keras.layers.Concatenate(name='decoder_attn_concat')

    if not config:
      self.lstm_cell_1 = keras.layers.LSTMCell(units=units)
      self.lstm_cell_2 = keras.layers.LSTMCell(units=units)
      self.lstm_cell_3 = keras.layers.LSTMCell(units=units)
      self.attn_cell = AttentionCell(name='decoder_attention')
      self.dense_cell = keras.layers.Dense(units=constants['Y_DIM'], activation=None, name='decoder_dense')
    else:
      self.lstm_cell_1 = keras.layers.LSTMCell.from_config(config['lstm_cell_1'])
      self.lstm_cell_2 = keras.layers.LSTMCell.from_config(config['lstm_cell_2'])
      self.lstm_cell_3 = keras.layers.LSTMCell.from_config(config['lstm_cell_3'])
      self.attn_cell = AttentionCell.from_config(config['attn_cell'])
      self.dense_cell = keras.layers.Dense.from_config(config['dense_cell'])

    lstm_state_sizes = [
        units, units, units, units, units, units       
    ]

    self.state_size = [
                       tf.TensorShape(constants['Y_DIM']),
                       lstm_state_sizes,
                       tf.TensorShape(constants['MAX_CHORALE_LENGTH']),
                       tf.TensorShape((constants['MAX_CHORALE_LENGTH'], units))]
    self.output_size = units

  def call(self, inputs, states):
    dense_in, lstm_states, attention_energies_in, encoder_out = states
    
    inputs_and_dense = self.lstm_input_concat_cell([inputs, dense_in, attention_energies_in])
    dense_and_dense =  self.lstm_input_concat_cell([dense_in, dense_in, attention_energies_in])

    lstm_input = tf.cond(tf.random.uniform(shape=(), minval=0, maxval=1) < self.teacher_forcing_prob,
                         true_fn=lambda: inputs_and_dense,
                         false_fn=lambda: dense_and_dense)

    lstm_in_state_h_1, lstm_in_state_h_2, lstm_in_state_h_3, lstm_in_state_c_1, lstm_in_state_c_2, lstm_in_state_c_3 = lstm_states
    lstm_out_1, [lstm_out_state_h_1, lstm_out_state_c_1] = self.lstm_cell_1(lstm_input, [lstm_in_state_h_1, lstm_in_state_c_1])
    lstm_out_2, [lstm_out_state_h_2, lstm_out_state_c_2] = self.lstm_cell_2(lstm_out_1, [lstm_in_state_h_2, lstm_in_state_c_2])
    lstm_out, [lstm_out_state_h_3, lstm_out_state_c_3] = self.lstm_cell_3(lstm_out_2, [lstm_in_state_h_3, lstm_in_state_c_3])
    lstm_out_states = [
        lstm_out_state_h_1,
        lstm_out_state_h_2,
        lstm_out_state_h_3,
        lstm_out_state_c_1,
        lstm_out_state_c_2,
        lstm_out_state_c_3,
    ]

    attention_context_out, attention_energies_out = self.attn_cell([encoder_out, lstm_out])
    attention_context_out = attention_context_out[0]
    attention_energies_out = attention_energies_out[0]
    concat_out = self.attn_concat_cell([attention_context_out, lstm_out])
    dense_out = self.dense_cell(concat_out)

    return [dense_out, attention_energies_out], [dense_out, lstm_out_states, attention_energies_out, encoder_out]

  def get_config(self):
      return {
          'units': self.units,
          'teacher_forcing_prob': self.teacher_forcing_prob,
          'constants': self.constants,
          'lstm_cell_1': self.lstm_cell_1.get_config(),
          'lstm_cell_2': self.lstm_cell_2.get_config(),
          'lstm_cell_3': self.lstm_cell_3.get_config(),
          'attn_cell': self.attn_cell.get_config(),
          'dense_cell': self.dense_cell.get_config()
      }

  @classmethod
  def from_config(cls, config):
      return cls(units=config['units'], constants=config['constants'], teacher_forcing_prob=config['teacher_forcing_prob'], config=config)
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
