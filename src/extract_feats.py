import os, sys
import numpy as np
from scipy.io import wavfile
import utils

import matplotlib.pyplot as plt
import librosa.display


if __name__ == '__main__':
    sr, data = wavfile.read('/home/adam/Desktop/00001.wav', False)
    data = data.astype(np.float32) / (-np.iinfo(data.dtype).min)
    data, sr = utils.preproc_audio(data, sr)
    f = utils.extract_fbanks(data, sr, 25, 15, 128)
    # f = utils.extract_mfcc(data, sr, 25, 15, 20)
    # librosa.display.specshow(np.pad(f, ((0, 0), (10, 10)), mode='reflect'))
    # librosa.display.specshow(f - np.mean(f, axis=1)[:, np.newaxis])
    librosa.display.specshow(utils.mean_normalize(f, 48000))
    plt.show()
    print(f.shape)
