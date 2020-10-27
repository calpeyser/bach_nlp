import os, pathlib

import feature_extractors
import scoring

import numpy as np
import random

import tensorflow as tf
from tensorflow import keras 
K = keras.backend

LATENT_DIM = 128

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

  def _create_masked_loss(slice_fn, loss_fn):
    def _l(y_true, y_pred):
      mask = tf.math.logical_not(K.all(K.equal(y_true, mask_value), axis=-1))
      y_true = slice_fn(y_true)[0]
      loss = loss_fn(y_true, y_pred)
      masked_loss = tf.boolean_mask(loss, mask)
      avg_masked_loss = tf.reduce_mean(masked_loss)
      return avg_masked_loss
    return _l
  LOSSES = {
      'key': _create_masked_loss(KEY_SLICE, xentropy_fn),
      'mode': _create_masked_loss(MODE_SLICE, mse_fn),
      'degree': _create_masked_loss(DEGREE_SLICE, xentropy_fn),
      'inversion': _create_masked_loss(INVERSION_SLICE, xentropy_fn),
      'quality': _create_masked_loss(QUALITY_SLICE, xentropy_fn),
      'measure': _create_masked_loss(MEASURE_SLICE, mse_fn),
      'beat': _create_masked_loss(BEAT_SLICE, mse_fn),
      'is_terminal': _create_masked_loss(IS_TERMINAL_SLICE, mse_fn),
  }
  return LOSSES

# Builds the attention model for training.
def _build_model(constants):
    encoder_inputs = keras.Input(shape=(constants['MAX_SEQ_LEN'], constants['X_DIM']))

    masking_layer = keras.layers.Masking(mask_value=constants['MASK_VALUE'])
    masked_inputs = masking_layer(encoder_inputs)
    
    encoder = keras.layers.LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder(masked_inputs)

    encoder_states = [state_h, state_c]

    decoder_inputs = keras.Input(shape=(constants['MAX_SEQ_LEN'], constants['Y_DIM']))

    decoder_lstm = keras.layers.LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    decoder_dense = keras.layers.Dense(constants['Y_DIM'], activation=None)

    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    
    attn = keras.layers.Attention()
    attention_outputs = attn([decoder_outputs, encoder_outputs])

    dense_outputs = decoder_dense(keras.layers.concatenate([decoder_outputs, attention_outputs]))
    output_components = _convert_dense_to_output_components(dense_outputs)

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
    train_data, test_data, constants = feature_extractors.load_dataset()
    encoder_input_data, decoder_input_data, decoder_target_data = train_data

    # Add slices here to train on only a subset of the data
    encoder_input_data = encoder_input_data[:50]
    decoder_input_data = decoder_input_data[:50]
    decoder_target_data = decoder_target_data[:50]

    model = _build_model(constants)

    l = create_losses(constants['MASK_VALUE'])
    o = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=o, loss=l)
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=64,
            epochs=1,
            validation_split=0.05,
            shuffle=True,
            callbacks=[NBatchLogger()],
            verbose=0)
    model.save('attn_256_multitask')

def predict():
    train_data, test_data, constants = feature_extractors.load_dataset()

    # To eval on the train set, specify "train_data" here
    encoder_input_data, decoder_input_data, decoder_target_data = test_data

    model = keras.models.load_model('attn_256_multitask', compile=False)

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
    encoder_outputs, state_h_enc, state_c_enc = _get_layers('LSTM')[0].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, [encoder_states, encoder_outputs])

    # Extract decoder from graph
    decoder_inputs = model.input[1]
    decoder_state_input_h = keras.Input(shape=(LATENT_DIM,), name="input_3_1")
    decoder_state_input_c = keras.Input(shape=(LATENT_DIM,), name="input_4_1")
    encoder_output_input = keras.Input(shape=(constants['MAX_SEQ_LEN'], LATENT_DIM), name="encoder_output_input")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


    decoder_lstm = _get_layers('LSTM')[1]
    decoder_attn = _get_layers('Attention')[0]
    decoder_concat = _get_layers('Concatenate')[0]
    decoder_dense = _get_layers('Dense')[0]

    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    attention_outputs = decoder_attn([decoder_outputs, encoder_output_input])
    dense_outputs = decoder_dense(decoder_concat([decoder_outputs, attention_outputs]))
    output_components = _convert_dense_to_output_components(dense_outputs)

    ins = [decoder_inputs]
    ins.extend(decoder_states_inputs)
    ins.append(encoder_output_input)
    outs = [output_components]
    outs.extend(decoder_states)
    decoder_model = keras.Model(ins, outs)

    # Retruns true when the "is_terminal" output is set to -1, meaning the RNA
    # is finished.
    def _terminate(toks):
        return (toks[-1] < 0)
    
    # Decode a single chorale.
    def decode(input_seq):
        states_value, encoder_outputs = encoder_model.predict(np.array([input_seq]))
        target_seq = np.ones((1, 1, constants['Y_DIM'])) * -5.

        result = []
        for k in range(constants['MAX_SEQ_LEN']):
            ins = [target_seq]
            ins.extend(states_value)
            ins.append(encoder_outputs)
            output_components, h, c = decoder_model.predict(ins)
            output_tokens = np.concatenate([output_components[key] for key in COMPONENTS_IN_ORDER], axis=-1)
            result.append(output_tokens)
            if _terminate(output_tokens[0][0]):
                return result

            target_seq = np.ones((1, 1, constants['Y_DIM'])) * output_tokens
            states_value = [h, c]
        print("Decoding did not terminate! Returning large RNA.")
        return result

    def cut_off_ground_truth(ground_truth):
        res = []
        for g in ground_truth:
            if _terminate(g):
                return res
            res.append(g)
        print("Ground truth does not terminate! Returning large RNA.")


    err_rates = []
    len_diffs = []
    # Eval the first 10 chorales
    for chorale_ind in range(len(encoder_input_data))[:10]:
        print("Eval for chorale " + str(chorale_ind))
        decoded = decode(encoder_input_data[chorale_ind])
        decoded_rna_chords = [feature_extractors.RNAChord(encoding=decoded[i][0][0]) for i in range(len(decoded))]

        ground_truth = cut_off_ground_truth(decoder_target_data[chorale_ind])
        ground_truth_chords = [feature_extractors.RNAChord(encoding=ground_truth[i]) for i in range(len(ground_truth))]

        errs = scoring.levenshtein(ground_truth_chords, decoded_rna_chords, equality_fn=scoring.EQUALITY_FNS['key_enharmonic'])
        print(len(ground_truth_chords) - len(decoded_rna_chords))
        len_diffs.append(len(ground_truth_chords) - len(decoded_rna_chords))
        err_rates.append(float(errs / len(ground_truth_chords)))
        # Uncomment these lines to see the ground truth RNA sequence together
        # with the decoded prediction.
        # print("--------------------- GROUND TRUTH  ------------------")
        # for c in ground_truth_chords:
        #     print(c)
        # print("---------------------  PREDICTION  -------------------")
        # for c in decoded_rna_chords:
        #     print(c)

    print("Error rate: " + str(np.mean(err_rates)))
    print("Len diff: " + str(np.mean(len_diffs)))




#train()
predict()