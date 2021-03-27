import numpy as np
import librosa as rosa


def preproc_audio(data, sr):
    if sr != 16000:
        data = rosa.core.resample(data, sr, 16000, res_type='kaiser_best')
    data = rosa.effects.preemphasis(data, coef=0.97)
    return data, 16000


def extract_fbanks(data, sr, frame_size, frame_step, n_mels):
    win_l = int(np.round(0.001 * frame_size * sr))
    hop_l = int(np.round(0.001 * frame_step * sr))
    n_fft = int(2 ** np.ceil(np.log2(win_l)))
    power = rosa.feature.melspectrogram(data, sr, n_fft=n_fft, win_length=win_l, hop_length=hop_l, n_mels=n_mels)
    return rosa.power_to_db(np.abs(power))


def extract_mfcc(data, sr, frame_size, frame_step, n_mfcc):
    fbanks = extract_fbanks(data, sr, frame_size, frame_step, 128)
    return rosa.feature.mfcc(data, sr, S=fbanks, n_mfcc=n_mfcc, dct_type=2, norm='ortho')


def mean_normalize(x, win_size):
    step = win_size // 2
    integral = np.cumsum(x, axis=1)
    y = np.empty_like(x)
    for i in range(y.shape[1]):
        a = max(0, i - step)
        b = min(i + step, y.shape[1] - 1)
        local_mean = ((integral[:, b] - integral[:, a]) / (b - a))
        y[:, i] = x[:, i] - local_mean
    return y
