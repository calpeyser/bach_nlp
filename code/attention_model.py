import os, pathlib

import numpy as np
import random

import tensorflow as tf
from tensorflow import keras 

from feature_extractors import RNAChord, ChoraleChord
from scoring import levenshtein, EQUALITY_FNS
from thrush_attention import DecoderLayer

K = keras.backend

LATENT_DIM = 32

ROOT='/content/drive/My Drive/bach_nlp'

MODEL_PATH = ROOT + '/model'

COMPONENTS_IN_ORDER = ['key', 'mode', 'degree', 'inversion', 'quality', 'measure', 'beat', 'is_terminal']

# This would make more sense as a dict of lambdas, but independant statments are
# required for AutoGraph magic in TF.
def KEY_SLICE(x): return x[:,:,:12],
def MODE_SLICE(x): return K.expand_dims(x[:,:,12]),
def DEGREE_SLICE(x): return x[:,:,13:21],
def INVERSION_SLICE(x): return x[:,:,21:25],
def QUALITY_SLICE(x): return x[:,:,25:30],
def MEASURE_SLICE(x): return K.expand_dims(x[:,:,30]),
def BEAT_SLICE(x): return x[:,:,31:33],
def IS_TERMINAL_SLICE(x): return K.expand_dims(x[:,:,33]), 

# This function consumes the outputs of the final dense layer in the model and
# splits its up according to parts of the roman numeral analysis.  This should
# be at the end of the model.
def make_output_components(dense_outputs):
    return {
      'key': keras.layers.Lambda(KEY_SLICE, name='key_l')(dense_outputs)[0],
      'mode': keras.layers.Lambda(MODE_SLICE, name='mode_l')(dense_outputs)[0],
      'degree': keras.layers.Lambda(DEGREE_SLICE, name='degree_l')(dense_outputs)[0],
      'inversion': keras.layers.Lambda(INVERSION_SLICE, name='inversion_l')(dense_outputs)[0],
      'quality': keras.layers.Lambda(QUALITY_SLICE, name='quality_l')(dense_outputs)[0],
      'measure': keras.layers.Lambda(MEASURE_SLICE, name='measure_l')(dense_outputs)[0],
      'beat': keras.layers.Lambda(BEAT_SLICE, name='beat_l')(dense_outputs)[0], 
      'is_terminal': keras.layers.Lambda(IS_TERMINAL_SLICE, name='is_terminal_l')(dense_outputs)[0],   
    }

# Converts outputs of make_output_components by adding softmax layers to
# appropriate slices
def _convert_dense_to_output_components(dense_outputs):
  # Split out different parts of the dense representation
  output_components = make_output_components(dense_outputs)

  # Apply softmax to dense components modeling probabilities
  for comp in ['key', 'degree', 'inversion', 'quality']:
    output_components[comp] = keras.layers.Softmax(name=comp + '_s')(output_components[comp])
  return output_components

# Creates a dictionary for the model's multitask loss, by assigning an
# appropriate loss to each slice of the output.
def create_losses(mask_value):

  xentropy_fn = keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
  mse_fn = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
  # A special loss for is_terminal, which especially penalizes missing the terminal
  # -1.0
  def IsTerminalLoss(y_true, y_pred):
    mse_loss = mse_fn(y_true, y_pred)
    seq_len = tf.reduce_sum(tf.cast(tf.math.logical_not(K.all(K.equal(y_true, -1.0), axis=-1)), dtype=tf.float32)) 
    res = tf.where(tf.squeeze(tf.equal(y_true, [-1.0])), mse_loss * seq_len/20, mse_loss)
    return res
  is_terminal_fn = IsTerminalLoss

  def _create_masked_loss(slice_fn, loss_fn, debug=False):
    def _l(y_true, y_pred):
      mask = tf.math.logical_not(K.all(K.equal(y_true, mask_value), axis=-1))
      y_true = slice_fn(y_true)[0]
      loss = loss_fn(y_true, y_pred)
      masked_loss = tf.boolean_mask(loss, mask)
      avg_masked_loss = tf.reduce_mean(masked_loss)
      if debug:
        tf.print(y_pred, output_stream=sys.stdout, summarize=10000)
        tf.print(y_true, output_stream=sys.stdout, summarize=10000)
        tf.print(loss, output_stream=sys.stdout, summarize=10000)
      return avg_masked_loss
    return _l

  def debug_loss(y_true, y_pred):
    #tf.print(y_pred, output_stream=sys.stdout, summarize=10000)
    return 0

  LOSSES = {
      'key': _create_masked_loss(KEY_SLICE, xentropy_fn),
      'mode': _create_masked_loss(MODE_SLICE, mse_fn),
      'degree': _create_masked_loss(DEGREE_SLICE, xentropy_fn),
      'inversion': _create_masked_loss(INVERSION_SLICE, xentropy_fn),
      'quality': _create_masked_loss(QUALITY_SLICE, xentropy_fn),
      'measure': _create_masked_loss(MEASURE_SLICE, mse_fn),
      'beat': _create_masked_loss(BEAT_SLICE, mse_fn),
      'is_terminal': _create_masked_loss(IS_TERMINAL_SLICE, mse_fn, debug=False),
      'DEBUG': debug_loss,
  }
  return LOSSES

LOSS_WEIGHTS = {
    'key': 1.0,
    'mode': 1.0,
    'degree': 1.0,
    'inversion': 1.0,
    'quality': 1.0,
    'measure': 1.0,
    'beat': 1.0,
    'is_terminal': 1.0,
}

# Builds the attention model for training.
def _build_model(constants):
    encoder_inputs = keras.Input(shape=(constants['MAX_CHORALE_LENGTH'], constants['X_DIM']), name='chorale_input')
    chorale_masking_layer = keras.layers.Masking(mask_value=constants['MASK_VALUE'], name='chorale_masking')
    masked_encoder_inputs = chorale_masking_layer(encoder_inputs)
    
    encoder_1 = keras.layers.LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='encoder_lstm_1')
    encoder_2 = keras.layers.LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='encoder_lstm_2')
    encoder_3 = keras.layers.LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='encoder_lstm_3')

    encoder_outputs_1, _, _ = encoder_1(masked_encoder_inputs)
    encoder_outputs_2, _, _ = encoder_2(encoder_outputs_1)
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_3(encoder_outputs_2)

    decoder_inputs = keras.Input(shape=(constants['MAX_ANALYSIS_LENGTH'], constants['Y_DIM']), name='rna_input')
    rna_masking_layer = keras.layers.Masking(mask_value=constants['MASK_VALUE'], name='rna_masking')
    masked_decoder_inputs = rna_masking_layer(decoder_inputs)

    decoder_layer = DecoderLayer(units=LATENT_DIM, constants=constants, teacher_forcing_prob=0.0)
    decoder_recurrent = keras.layers.RNN(decoder_layer, return_sequences=True, return_state=True, name='decoder_layer')

    # hack to obtain zero vector as a Keras tensor
    initial_dense_ins = keras.layers.Lambda(lambda x: x - x - 5)(tf.reduce_sum(decoder_inputs, axis=1))
    initial_attn_energies = keras.layers.Lambda(lambda x: x - x - 5)(tf.reduce_sum(encoder_outputs, axis=-1))
    initial_lstm_states = [
        encoder_state_h,
        encoder_state_h,
        encoder_state_h,
        encoder_state_c,
        encoder_state_c,
        encoder_state_c,
    ]
    initial_decoder_state = [initial_dense_ins, initial_lstm_states, initial_attn_energies, encoder_outputs]
    dense_outputs, attn_energies, _, lstm_out_states, _, _ = decoder_recurrent(masked_decoder_inputs, initial_state=initial_decoder_state)
    output_components = _convert_dense_to_output_components(dense_outputs)
    output_components['DEBUG'] = {
        'dbg': lstm_out_states[0]
    }

    m = keras.Model([encoder_inputs, decoder_inputs], output_components)
    return m

class NBatchLogger(keras.callbacks.Callback):
    def __init__(self,display=10):
        '''
        display: Number of batches to wait before outputting loss
        '''
        self.seen = 0
        self.display = display

    def on_epoch_end(self,epoch,logs={}):
        self.seen += epoch
        if self.seen % self.display == 0:
            print('Epoch %s, Metrics: %s' % (str(epoch), str(logs)))


def train():
    #tf.debugging.set_log_device_placement(True)

    train_data, test_data, constants = DATASET
    encoder_input_data, decoder_input_data, decoder_target_data = train_data

    # Add slices here to train on only a subset of the data
    encoder_input_data = encoder_input_data
    decoder_input_data = decoder_input_data
    decoder_target_data = decoder_target_data

    model = _build_model(constants)
    model.summary(line_length=200)

    l = create_losses(constants['MASK_VALUE'])
    o = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.8)
    model.compile(optimizer=o, loss=l, loss_weights=LOSS_WEIGHTS)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH + '/thrush_attn_32_bs256_istermmse_lr001B09_luongattn_big/{epoch:02d}',
        save_weights_only=False,
        save_freq=150)

    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=256,
            epochs=500,
            validation_split=0.05,
            shuffle=True,
            verbose=1,
            callbacks=[model_checkpoint_callback])

def predict():
    train_data, test_data, constants = DATASET
    encoder_input_data, decoder_input_data, decoder_target_data = test_data

    # Add slices here to test only a subset of the data
    # encoder_input_data = encoder_input_data[:1]
    # decoder_input_data = decoder_input_data[:1]
    # decoder_target_data = decoder_target_data[:1]


    # model = keras.models.load_model(MODEL_PATH + '/thrush_attn_32_bs256_istermmse_lr001B09_luongattn_big_TF0_3/510',
    #                                 custom_objects={
    #                                     'DecoderLayer': DecoderLayer,
    #                                     'AttentionCell': AttentionCell,
    #                                 },
    #                                 compile=False)
    model = _build_model(constants)
    #model.load_weights(MODEL_PATH + '/thrush_attn_32_bs256_istermmse_lr001B09_luongattn_big_TF075_4/660/variables/variables')
    #model.load_weights(MODEL_PATH + '/thrush_attn_32_bs256_istermmse_lr001B09_luongattn_big_TF05_4/720/variables/variables')
    model.load_weights(MODEL_PATH + '/thrush_attn_32_bs256_attnin_lr001B09_luongattn_big_TF09_4/1520/variables/variables')
    #model.load_weights(MODEL_PATH + '/thrush_attn_64_bs256_attnin_lr001B09_luongattn_big_TF09_dropout/700/variables/variables')

    def _get_layers(layer_type):
      return [l for l in model.layers if layer_type in str(type(l))]
    
    # We have to do a weird thing here to make decoding work: We need to create
    # new Keras model which instead of consuming an entire chorale/RNA sequence,
    # consumes only a chorale sequence, and produces the RNA sequence one step
    # at a time.  To do this, we extract the relevant layers of the training
    # model one by one and then piece them back together.  I got the idea from
    # https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html 

    # Extract encoder from graph
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = _get_layers('LSTM')[-1].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, [encoder_states, encoder_outputs])

    # Extract decoder from graph
    decoder_inputs = keras.Input(shape=(1, constants['Y_DIM']), name='rna_input_inference')
    dense_out_inputs = keras.Input(shape=(constants['Y_DIM']), name='dense_out_inputs_inference')
    # Inputs for the last decoder state
    decoder_state_inputs = [
                            keras.Input(shape=(LATENT_DIM,), name='decoder_state_h_1_inference'),
                            keras.Input(shape=(LATENT_DIM,), name='decoder_state_h_2_inference'),
                            keras.Input(shape=(LATENT_DIM,), name='decoder_state_h_3_inference'),
                            keras.Input(shape=(LATENT_DIM,), name='decoder_state_c_1_inference'),
                            keras.Input(shape=(LATENT_DIM,), name='decoder_state_c_2_inference'),
                            keras.Input(shape=(LATENT_DIM,), name='decoder_state_c_3_inference'),
    ]
    decoder_state_input_attn_energies = keras.Input(shape=(constants['MAX_CHORALE_LENGTH']), name='decoder_state_input_attn_energies')
    encoder_output_input = keras.Input(shape=(constants['MAX_CHORALE_LENGTH'], LATENT_DIM), name="encoder_output_input")

    decoder_recurrent = _get_layers('recurrent.RNN')[0]
    
    decoder_states_inputs = [dense_out_inputs, decoder_state_inputs, decoder_state_input_attn_energies, encoder_output_input]
    dense_outputs, _, _, decoder_state, attention_energies, _ = decoder_recurrent(decoder_inputs, initial_state=decoder_states_inputs)

    output_components = _convert_dense_to_output_components(dense_outputs)

    ins = [decoder_inputs]
    ins.extend(decoder_states_inputs)
    outs = [output_components, dense_outputs]
    outs.extend([decoder_state, attention_energies])
    decoder_model = keras.Model(ins, outs)

    # Returns true when the "is_terminal" output is set to -1, meaning the RNA
    # is finished.
    def _terminate(toks):
        return (toks[-1] <= -0.5)
    
    # Cut off analysis at minimum value of is_terminal
    def _cut_off(result, attn_energies, output_tokens):
      is_terms = [r[0][0][-1] for r in output_tokens]
      terminal_chord_ind = np.argmin(is_terms)
      for i, term in enumerate(is_terms):
        if term < 0:
          terminal_chord_ind = i
          break
      return result[:terminal_chord_ind + 1], attn_energies[:terminal_chord_ind + 1]


    # Decode a single chorale.
    def decode(input_seq, decoder_input=None):
        (prev_decoder_state_h, prev_decoder_state_c), encoder_output_values = encoder_model.predict(np.array([input_seq]))
        prev_decoder_states = [
                               prev_decoder_state_h,
                               prev_decoder_state_h,
                               prev_decoder_state_h,
                               prev_decoder_state_c,
                               prev_decoder_state_c,
                               prev_decoder_state_c
        ]

        prev_attention_energies = tf.zeros_like(tf.reduce_sum(encoder_output_values, axis=-1))

        target_seq = np.ones((1, 1, constants['Y_DIM'])) * -5.
        prev_dense = np.ones((1, constants['Y_DIM'])) * -5.

        result = []
        output_tokens_list = []
        attn_energies = []
        for k in range(constants['MAX_ANALYSIS_LENGTH'])[:-1]:
            ins = [target_seq]
            ins.extend([prev_dense, prev_decoder_states, prev_attention_energies, encoder_output_values])
            ins.append(encoder_output_values)
            output_components, dense_outs, prev_decoder_states, prev_attention_energies = decoder_model.predict(ins)
            attn_energies.append(prev_attention_energies)
            output_tokens_after_softmax = np.concatenate([output_components[key] for key in COMPONENTS_IN_ORDER], axis=-1)

            # translate to RNAChord during decoding
            rna_chord = RNAChord(encoding=output_tokens_after_softmax[0][0])
            output_tokens_from_chord = [[rna_chord.encode()]]
            output_tokens_list.append(output_tokens_after_softmax)
            result.append(output_tokens_from_chord)

            target_seq = np.ones((1, 1, constants['Y_DIM'])) * dense_outs
            prev_dense = np.ones((1, constants['Y_DIM'])) * dense_outs
            prev_dense = np.squeeze(prev_dense, axis=1)

        result, attn_energies = _cut_off(result, attn_energies, output_tokens_list)
        return result, attn_energies

    def cut_off_ground_truth(ground_truth):
        res = []
        for g in ground_truth:
            if _terminate(g):
                return res
            res.append(g)
        print("Ground truth does not terminate! Returning large RNA.")


    err_rates = []
    len_diffs = []
    attn_energy_matrixes = []
    chorale_inds = list(range(len(encoder_input_data)))
    random.shuffle(chorale_inds)
    for chorale_ind in chorale_inds[:20]:
        print("Eval for chorale " + str(chorale_ind))

        decoded, attn_energies = decode(encoder_input_data[chorale_ind], decoder_input_data[chorale_ind])
        attn_energy_matrixes.append(attn_energies)
        decoded_rna_chords = [RNAChord(encoding=decoded[i][0][0]) for i in range(len(decoded))]

        ground_truth = cut_off_ground_truth(decoder_target_data[chorale_ind])
        ground_truth_chords = [RNAChord(encoding=ground_truth[i]) for i in range(len(ground_truth))]

        err_rates = collections.defaultdict(list)
        for fn_name in EQUALITY_FNS.keys():
          errs = levenshtein(ground_truth_chords, decoded_rna_chords, equality_fn=EQUALITY_FNS[fn_name], substitution_cost=1, left_deletion_cost=0, right_deletion_cost=1)
          err_rates[fn_name].append(float(errs / len(decoded_rna_chords)))

        print("Ground Truth: %s, Decoded: %s" % (len(ground_truth_chords), len(decoded_rna_chords)))

        # Uncomment these lines to see the ground truth RNA sequence together
        # with the decoded prediction.
        # print("--------------------- GROUND TRUTH  ------------------")
        # for c in ground_truth_chords:
        #     print(c)
        # print("---------------------  PREDICTION  -------------------")
        # for c in decoded_rna_chords:
        #     print(c)
        # print("-------------------- ANALYSIS COMPLETE ---------------")

    for fn_name in EQUALITY_FNS.keys():
      print("Error Name: " + fn_name + " Error Rate: " + str(np.mean(err_rates[fn_name])))
    return attn_energy_matrixes

train()
attn_energy_matrixes = predict()

