from collections import Sequence


def flatten(sequence, seq_type=Sequence):
    return list(flatten_(sequence, seq_type))


def flatten_(sequence, seq_type):
    for seq in sequence:
        if isinstance(seq, seq_type):
            yield from flatten_(seq, seq_type)
        else:
            yield seq