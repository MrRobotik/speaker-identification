import numpy as np
import librosa
import collections


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
        # simple energy based VAD enhanced with hysteresis:
        frames = energy_based_vad(frames)
    eps = np.finfo(np.float32).eps
    power = 20 * np.log(mel_f.dot(frames ** 2) + eps)
    return power


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


def energy_based_vad(frames, context=10):
    eps = np.finfo(np.float32).eps
    energy = 10 * np.log(np.sum(frames, axis=0) + eps)
    emin, emax = np.min(energy), np.max(energy)
    lower_m = energy > (emin + (emax - emin) * 0.25)
    upper_m = energy > (emin + (emax - emin) * 0.50)
    speech_inds = list(np.nonzero(upper_m)[0])
    queue = collections.deque()
    for i in speech_inds:
        queue.append(i)
    while len(queue) > 0:
        i = queue.popleft()
        a = max(0, i - context)
        b = min(i + context, len(energy) - 1)
        for j in np.nonzero(lower_m[a:b])[0] + a:
            if not upper_m[j]:
                queue.append(j)
                speech_inds.append(j)
                upper_m[j] = True
    return frames[:, sorted(speech_inds)]
