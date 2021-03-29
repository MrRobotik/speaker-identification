import numpy as np
import librosa


def preproc_audio(data, sr):
    if data.ndim > 1:
        data = librosa.to_mono(data)
    if sr != 16000:
        data = librosa.resample(data, sr, 16000, res_type='kaiser_best')
    data = librosa.effects.preemphasis(data, coef=0.97)
    return 16000, data


def extract_fbanks(data, sr, frame_size, frame_step, n_mels, use_vad=True):
    win_l = int(np.round(0.001 * frame_size * sr))
    hop_l = int(np.round(0.001 * frame_step * sr))
    n_fft = int(2 ** np.ceil(np.log2(win_l)))
    mel_f = librosa.filters.mel(sr, n_fft, n_mels, norm='slaney')
    frames = np.abs(librosa.stft(data, n_fft, hop_l, win_l, window='hann', center=True))
    if use_vad:
        frames = energy_based_vad(frames)
    power = mel_f.dot(frames ** 2)
    return 20 * np.log(power)


def extract_mfcc(data, sr, frame_size, frame_step, n_mfcc, use_vad=True, fbanks=None):
    if fbanks is None:
        fbanks = extract_fbanks(data, sr, frame_size, frame_step, 128, use_vad)
    return librosa.feature.mfcc(None, sr, fbanks, n_mfcc, dct_type=2, norm='ortho')


def mean_normalize(frames, win_size):
    step = win_size // 2
    integral = np.cumsum(frames, axis=1)
    y = np.empty_like(frames)
    n = frames.shape[1]
    for i in range(n):
        a = max(0, i - step)
        b = min(i + step, n - 1)
        local_mean = ((integral[:, b] - integral[:, a]) / (b - a))
        y[:, i] = frames[:, i] - local_mean
    return y


def energy_based_vad(frames, context=5):
    energy = 10 * np.log(np.sum(frames, axis=0))
    emin, emax = np.min(energy), np.max(energy)
    lower_tr = emin + (emax - emin)/4
    upper_tr = emin + (emax - emin)/2
    mask = energy > upper_tr
    for i in np.nonzero(energy > lower_tr)[0]:
        if mask[i]:
            continue
        if np.count_nonzero(mask[i-context:i+context]):
            mask[i] = 1
    return frames[:, np.nonzero(mask)[0]]

