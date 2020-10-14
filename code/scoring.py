import os, pathlib

import music21
import numpy as np


def levenshtein(seq_a, seq_b, equality_fn=None, substitution_cost=1, deletion_cost=1):
    assert(equality_fn is not None, 'equality_fn must be set')

    memo = [[None for _ in range(len(seq_b) + 1)] for _ in range(len(seq_a) + 1)]

    def _levenshtein(inner_seq_a, inner_seq_b):
        if memo[len(inner_seq_a)][len(inner_seq_b)] != None:
            return memo[len(inner_seq_a)][len(inner_seq_b)]

        if len(inner_seq_b) == 0:
            res = len(inner_seq_a)
        elif len(inner_seq_a) == 0:
            res = len(inner_seq_b)
        else:
            reduce_a = deletion_cost + _levenshtein(inner_seq_a[:-1], inner_seq_b)
            reduce_b = deletion_cost + _levenshtein(inner_seq_a, inner_seq_b[:-1])

            incurred_substitution_penalty = substitution_cost
            if equality_fn(inner_seq_a[-1], inner_seq_b[-1]):
                incurred_substitution_penalty = 0
            reduce_both = incurred_substitution_penalty + _levenshtein(inner_seq_a[:-1], inner_seq_b[:-1])

            res = min(reduce_a, reduce_b, reduce_both)

        memo[len(inner_seq_a)][len(inner_seq_b)] = res
        return res

    return _levenshtein(seq_a, seq_b)

def _direct_equality_fn(a, b):
    return a == b

def _key_equality_fn(chord_a, chord_b):
    return chord_a.key == chord_b.key

def _key_forgive_enharmonic_equality_fn(chord_a, chord_b):
    if chord_a.key == chord_b.key:
        return True
    else:
        return chord_a.key == chord_b.key.relative

EQUALITY_FNS = {
    'direct': _direct_equality_fn,
    'key': _key_equality_fn,
    'key_enharmonic': _key_forgive_enharmonic_equality_fn
}

if __name__ == '__main__':
    print(levenshtein('sitten', 'ittin', equality_fn=EQUALITY_FNS['direct']))
