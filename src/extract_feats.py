import sys, os
import numpy as np
from scipy.io import wavfile
import utils

HELP = \
    "Extract Mel filter-banks and MFCC features from *.wav files.\n" \
    "Usage: extract_feats.py in_dir out_dir\n"


def extract_feats(filepath):
    sr, data = wavfile.read(filepath, False)
    if data.dtype != np.float32:
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 127.0 - 1.0
        else:
            data = data.astype(np.float32) / (-np.iinfo(data.dtype).min)
    sr, data = utils.preproc_audio(data, sr)
    frame_size = 25
    frame_step = 15
    fbanks = utils.extract_fbanks(data, sr, frame_size, frame_step, n_mels=128)
    mfcc = utils.extract_mfcc(data, sr, frame_size, frame_step, n_mfcc=24, fbanks=fbanks)
    norm_win_size = 200
    fbanks = utils.mean_normalize(fbanks, norm_win_size)
    mfcc = utils.mean_normalize(mfcc, norm_win_size)
    return fbanks, mfcc


def main():
    if len(sys.argv) != 3:
        print(HELP), exit(0)
    src_root = sys.argv[1]
    dst_root = sys.argv[2]
    counter = 1
    for speaker_id in os.listdir(src_root):
        src_dir = src_root + os.sep + speaker_id
        dst_dir = dst_root + os.sep + speaker_id
        try:
            os.mkdir(dst_dir)
        except FileExistsError as e:
            print(e, file=sys.stderr)
            continue
        for source_id in os.listdir(src_dir):
            src_subdir = src_dir + os.sep + source_id
            dst_subdir = dst_dir + os.sep + source_id
            try:
                os.mkdir(dst_subdir)
            except FileExistsError as e:
                print(e, file=sys.stderr)
                continue
            for fn in os.listdir(src_subdir):
                i_filepath = src_subdir + os.sep + fn
                o_filepath = dst_subdir + os.sep + fn
                print('processing: {}'.format(i_filepath))
                fbanks, mfcc = extract_feats(i_filepath)
                o_filepath = o_filepath[:o_filepath.rfind(os.extsep)]
                np.save(o_filepath + '-fbanks.npy', fbanks)
                np.save(o_filepath + '-mfcc.npy',   mfcc)
                print('done ({})'.format(counter))
                counter += 1


if __name__ == '__main__':
    main()

