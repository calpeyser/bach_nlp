import os, pathlib

import dummy_data_generator
import feature_extractors
import scoring

import numpy as np
import random

import tensorflow as tf
from tensorflow import keras 
K = keras.backend

LATENT_DIM = 256

def _build_model(constants):
    encoder_inputs = keras.Input(shape=(constants['MAX_SEQ_LEN'], constants['X_DIM']))

    masking_layer = keras.layers.Masking(mask_value=constants['MASK_VALUE'])
    masked_inputs = masking_layer(encoder_inputs)
    
    encoder = keras.layers.LSTM(LATENT_DIM, return_state=True)
    _, state_h, state_c = encoder(masked_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = keras.Input(shape=(constants['MAX_SEQ_LEN'], constants['Y_DIM']))

    decoder_lstm = keras.layers.LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    decoder_dense = keras.layers.Dense(constants['Y_DIM'], activation=None)

    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    # decoder_outputs, _, _ = decoder_lstm(decoder_inputs)
    decoder_outputs = decoder_dense(decoder_outputs)
    #decoder_outputs = tf.compat.v1.Print(decoder_outputs, [decoder_outputs], summarize=100)


    m =  keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return m

def train():
    train_data, test_data, constants = feature_extractors.load_dataset()
    encoder_input_data, decoder_input_data, decoder_target_data = train_data
    model = _build_model(constants)

    l = keras.losses.MeanSquaredError()
    o = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=o, loss=l)
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=32,
            epochs=100,
            validation_split=0.05)
    model.save('chorale_model_256')

def prediction_model():
    train_data, test_data, constants = feature_extractors.load_dataset()
    encoder_input_data, decoder_input_data, decoder_target_data = test_data

    model = keras.models.load_model('chorale_model_256')
    
    # Extract encoder from graph
    encoder_inputs = model.input[0]
    _, state_h_enc, state_c_enc = model.layers[3].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    # Extract decoder from graph
    decoder_inputs = model.input[1]
    decoder_state_input_h = keras.Input(shape=(LATENT_DIM,), name="input_3")
    decoder_state_input_c = keras.Input(shape=(LATENT_DIM,), name="input_4")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = model.layers[-2]
    decoder_dense = model.layers[-1]

    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    def _terminate(toks):
        return np.isclose(np.mean(-1 - toks), 0.0, atol=0.5)
    
    def decode(input_seq):
        states_value = encoder_model.predict(np.array([input_seq]))
        target_seq = np.ones((1, 1, constants['Y_DIM'])) * -1.

        result = []
        stop = False
        for _ in range(100):
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)
            if _terminate(output_tokens):
                return result
            result.append(output_tokens)

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
    for chorale_ind in range(len(encoder_input_data))[:15]:
        print("Eval for chorale " + str(chorale_ind))
        decoded = decode(encoder_input_data[chorale_ind])
        decoded_rna_chords = [feature_extractors.RNAChord(encoding=decoded[i][0][0]) for i in range(len(decoded))]

        ground_truth = cut_off_ground_truth(decoder_target_data[chorale_ind])
        ground_truth_chords = [feature_extractors.RNAChord(encoding=ground_truth[i]) for i in range(len(ground_truth))]

        errs = scoring.levenshtein(ground_truth_chords, decoded_rna_chords, equality_fn=scoring.EQUALITY_FNS['key_enharmonic'])
        print(len(ground_truth_chords) - len(decoded_rna_chords))
        len_diffs.append(len(ground_truth_chords) - len(decoded_rna_chords))
        err_rates.append(float(errs / len(ground_truth_chords)))
    print("Error rate: " + str(np.mean(err_rates)))
    print("Len diff: " + str(np.mean(len_diffs)))
    # for c in ground_truth_chords:
    #     print(c)
    # print("-----------------------------------------------------")
    # for c in decoded_rna_chords:
    #     print(c)




#train()
predict()