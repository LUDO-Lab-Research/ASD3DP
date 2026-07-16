import numpy as np


def fixed_10s_sample(wave, audio_length=160000):
    if len(wave) >= audio_length:
        return wave[:audio_length]

    pad_width = [(0, audio_length - len(wave))]
    pad_width.extend([(0, 0) for _ in range(wave.ndim - 1)])
    return np.pad(wave, pad_width, "constant")
