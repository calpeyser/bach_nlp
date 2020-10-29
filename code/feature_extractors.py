import os, pathlib, sys

import music21
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

X_FILE = (pathlib.Path(__file__).parent / ('../data/preprocessed_data/x.npy')).resolve()
Y_FILE = (pathlib.Path(__file__).parent / ('../data/preprocessed_data/y.npy')).resolve()

MAX_CHORALE_LENGTH = 391
MAX_ANALYSIS_LENGTH = 229

CHORALE_EMBEDDING_SIZE=79
ANALYSIS_EMBEDDING_SIZE=34

# Optional, if you want to view chorales
music21.environment.set('musicxmlPath', '/usr/bin/musescore3')

def _one_hot(index, depth):
    res = np.zeros(depth)
    res[index] = 1
    return res

class RNAChord(object):

    def __init__(self, romantext_chord='', encoding=[], is_terminal=False):
        assert not ((romantext_chord != '') and (encoding != []))
        if romantext_chord != '':
            self.key = romantext_chord.key
            self.degree = romantext_chord.scaleDegree
            self.inversion = romantext_chord.inversion()
            self.quality = romantext_chord.impliedQuality
            self.measure = romantext_chord.measureNumber
            self.start_beat = romantext_chord.offset
            self.end_beat = romantext_chord.offset + romantext_chord.duration.quarterLength
            self.is_terminal=is_terminal
        elif encoding != []:
            assert len(encoding == ANALYSIS_EMBEDDING_SIZE)
            self._decode_key(encoding[:13])
            self._decode_degree(encoding[13:21])
            self._decode_inversion(encoding[21:25])
            self._decode_quality(encoding[25:30])
            self._decode_measure(encoding[30])
            self._decode_beat(encoding[31:33])
        else:
            raise Exception("RNAChord requires either romantext_chord or encoding")


    def __str__(self):
        return "RNAChord with: \n\tKey: %s\n\tDegree: %s\n\tInversion: %s\n\tQuality: %s\n\tMeasure: %s\n\tStart Beat: %s\n\tEnd Beat: %s\n\t" % (
            self.key, self.degree, self.inversion, self.quality, self.measure, self.start_beat, self.end_beat
        ) 

    def _encode_key(self):
        tonic = self.key.tonic
        tonic.octave = -1
        tonic_num = tonic.midi

        mode = self.key.mode
        if mode == 'major':
            mode_num = 0
        elif mode == 'minor':
            mode_num = 1
        else:
            raise Exception('Mode ' + mode + ' not known.')
        encoding = np.concatenate([_one_hot(tonic_num, 12), [mode_num]])
        return encoding

    def _decode_key(self, encoding):
        assert len(encoding) == 13
        tonic = music21.pitch.Pitch(np.argmax(encoding[:12]))
        if encoding[12] < 0.5:
            self.key = music21.key.Key(tonic, mode='major')
        else:
            self.key = music21.key.Key(tonic, mode='minor')

    def _encode_degree(self):
        encoding = _one_hot(self.degree, 8)
        return encoding

    def _decode_degree(self, encoding):
        self.degree = np.argmax(encoding)

    def _encode_inversion(self):
        encoding = _one_hot(self.inversion, 4)
        return encoding

    def _decode_inversion(self, encoding):
        self.inversion = np.argmax(encoding)

    def _encode_quality(self):
        if self.quality == 'major' or self.quality == '':
            quality_num = 0
        elif self.quality == 'minor':
            quality_num = 1
        elif self.quality == 'diminished':
            quality_num = 2
        elif self.quality == 'half-diminished':
            quality_num = 3
        elif self.quality == 'augmented':
            quality_num = 4
        else:
            raise Exception('Quality ' + self.quality + ' not known.')
        encoding = _one_hot(quality_num, 5)
        return encoding

    def _decode_quality(self, encoding):
        quality_num = np.argmax(encoding)
        if quality_num == 0:
            self.quality = 'major'
        elif quality_num == 1:
            self.quality = 'minor'
        elif quality_num == 2:
            self.quality = 'diminished'
        elif quality_num == 3:
            self.quality = 'half-diminished'
        elif quality_num == 4:
            self.quality = 'augmented'
        else:
            raise Exception('Bad quality decode')

    def _encode_measure(self):
        encoding = np.array([self.measure])
        return encoding

    def _decode_measure(self,encoding):
        self.measure = np.round(encoding)

    def _encode_beat(self):
        encoding = np.array([self.start_beat, self.end_beat])
        return encoding

    def _decode_beat(self, encoding):
        self.start_beat = np.round(encoding[0])
        self.end_beat = np.round(encoding[1])

    def _encode_terminal(self):
        if self.is_terminal:
            return np.array([-1.0])
        else:
            return np.array([1.0])

    def encode(self):
        return np.concatenate([
            self._encode_key(),
            self._encode_degree(),
            self._encode_inversion(),
            self._encode_quality(),
            self._encode_measure(),
            self._encode_beat(),
            self._encode_terminal(),
        ])

class ChoraleChord(object):

    def __init__(self, score_chord):
        self.pitches = [n.pitch.simplifyEnharmonic() for n in score_chord.notes]
        self.pitch1 = None; self.pitch2 = None; self.pitch3 = None; self.pitch4 = None
        self.octave1 = None; self.octave2 = None; self.octave3 = None; self.octave4 = None
        if len(self.pitches) >= 1:
            self.octave1 = self.pitches[0].octave
            self.pitches[0].octave = -1
            self.pitch1 = self.pitches[0].midi
        if len(self.pitches) >= 2:
            self.octave2 = self.pitches[1].octave
            self.pitches[1].octave = -1
            self.pitch2 = self.pitches[1].midi
        if len(self.pitches) >= 3:
            self.octave3 = self.pitches[2].octave
            self.pitches[2].octave = -1
            self.pitch3 = self.pitches[2].midi
        if len(self.pitches) == 4:
            self.octave4 = self.pitches[3].octave
            self.pitches[3].octave = -1
            self.pitch4 = self.pitches[3].midi
        if len(self.pitches) > 4:
            raise Exception("More than four notes in chord: " + str(self.pitches))

        self.measure = score_chord.measureNumber
        self.start_beat = score_chord.offset
        self.end_beat = score_chord.offset + score_chord.duration.quarterLength

    def __str__(self):
        return "ChoraleChord with: \n\Pitches: %s\n\tMeasure: %s\n\tStart Beat: %s\n\tEnd Beat: %s" % (
            self.pitches, self.measure, self.start_beat, self.end_beat
        )

    def _encode_notes(self):
        def _encode_note(pitch, octave):
            assert (pitch == None) == (octave == None)
            if pitch == None:
                return np.zeros(19)
            else:
                return np.concatenate([_one_hot(pitch, 12), _one_hot(octave, 7)])
        return np.concatenate([
            _encode_note(self.pitch1, self.octave1),
            _encode_note(self.pitch2, self.octave2),
            _encode_note(self.pitch3, self.octave3),
            _encode_note(self.pitch4, self.octave4),
        ])

    def _encode_measure(self):
        encoding = np.array([self.measure])
        return encoding

    def _encode_beat(self):
        encoding = np.array([self.start_beat, self.end_beat])
        return encoding

    def encode(self):
        return np.concatenate([
            self._encode_notes(),
            self._encode_measure(),
            self._encode_beat(),
        ])

def process_rna(number, transposition_interval=None):
    analysis_file = (pathlib.Path(__file__).parent / ('../data/analysis/riemenschneider%s.txt' % number)).resolve()
    rna = music21.converter.parse(analysis_file, format="romantext")
    chords = []
    for element in rna.recurse():
        if (type(element).__name__ == 'RomanNumeral'):
            if (transposition_interval):
                element.key = element.key.transpose(transposition_interval)
            chords.append(RNAChord(romantext_chord=element))
    # for c in chords:
    #     print(c)
    # The last chorale should have the terminal bit set
    chords[-1].is_terminal = True
    return chords

def process_chorale(number, transposition_interval=None):
    chorale_file = (pathlib.Path(__file__).parent / ('../data/chorales/riemenschneider%s.xml' % number)).resolve()
    chorale = music21.converter.parse(chorale_file)
    if (transposition_interval):
        chorale = chorale.transpose(transposition_interval)
    # chorale.show()
    chordified = chorale.chordify()
    chords = []
    for element in chordified.recurse():
        if (type(element).__name__ == 'Chord'):
            chords.append(ChoraleChord(element))
    return chords

def create_dataset():
    x = []
    y = []
    for i in range(372)[1:]:
        for transposition_interval in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]:
            if i == 150: continue # Chorale 150 has five parts
            print("Processing Chorale %s in transposition %s" % (i, str(transposition_interval)))
            if i < 10:
                ind = "00%s" % i
            elif i < 100:
                ind = "0%s" % i
            else:
                ind = "%s" %i
            encoded_chorale_chords = []
            chorale_chords = process_chorale(ind, transposition_interval=transposition_interval)
            for chord in chorale_chords:
                enc = chord.encode()
                encoded_chorale_chords.append(enc)
            # Padding
            for _ in range(MAX_CHORALE_LENGTH)[len(encoded_chorale_chords):]:
                encoded_chorale_chords.append(-1. * np.ones(CHORALE_EMBEDDING_SIZE))
            encoded_chorale_chords = np.stack(encoded_chorale_chords)
            x.append(encoded_chorale_chords)
            
            encoded_rna_chords = []
            rna_chords = process_rna(ind, transposition_interval=transposition_interval)
            for chord in rna_chords:
                enc = chord.encode()
                encoded_rna_chords.append(enc)
            # Padding
            for _ in range(MAX_ANALYSIS_LENGTH)[len(encoded_rna_chords):]:
                encoded_rna_chords.append(-1. * np.ones(ANALYSIS_EMBEDDING_SIZE))
            encoded_rna_chords = np.stack(encoded_rna_chords)
            y.append(encoded_rna_chords)

    x = np.stack(x)
    np.save(X_FILE, x)

    y = np.stack(y)
    np.save(Y_FILE, y)

def load_dataset():
    x = np.load(X_FILE, allow_pickle=True)
    y = np.load(Y_FILE, allow_pickle=True)
    y_target = np.roll(y, -5, axis=1)

    print("Loaded x array of shape " + str(np.shape(x)))
    print("Loaded y array of shape " + str(np.shape(y)))
    print("Computed y_target array of shape " + str(np.shape(y_target)))

    split = int(len(x) * 9/10)
    
    x_train = x[:split]
    y_train = y[:split]
    y_target_train = y_target[:split]
    x_test = x[split+1:]
    y_test = y[split+1:]
    y_target_test = y_target[split+1:]

    return (x_train, y_train, y_target_train), (x_test, y_test, y_target_test), {
        'DATASET_SIZE': len(x_train),
        'MAX_CHORALE_LENGTH': MAX_CHORALE_LENGTH,
        'MAX_ANALYSIS_LENGTH': MAX_ANALYSIS_LENGTH,
        'MASK_VALUE': -1,
        'X_DIM': CHORALE_EMBEDDING_SIZE,
        'Y_DIM': ANALYSIS_EMBEDDING_SIZE,
    }


#create_dataset()
#print(load_dataset())