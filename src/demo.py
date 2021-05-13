import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import sounddevice as sd
import utils
import numpy as np
import math
import torch
from nn_models import XVectors


def extract_mfcc(data):
    if data.dtype != np.float32:
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 127.0 - 1.0
        else:
            data = data.astype(np.float32) / (np.iinfo(data.dtype).max + 1)
    sr, data = utils.preproc_audio(data, 16000)
    frame_size = 25
    frame_step = 15
    fbanks = utils.extract_fbanks(data, sr, frame_size, frame_step, n_mels=128)
    mfcc = utils.extract_mfcc(data, sr, frame_size, frame_step, n_mfcc=24, fbanks=fbanks)
    norm_win_size = 200
    fbanks = utils.mean_normalize(fbanks, norm_win_size)
    mfcc = utils.mean_normalize(mfcc, norm_win_size)
    return mfcc


class DemoApp(QApplication):

    def __init__(self):
        super().__init__(sys.argv)
        self.w = QWidget()
        self.w.setFixedSize(300, 200)
        self.w.setWindowTitle('Demo')
        self.layout = QVBoxLayout(self.w)
        self.button = QPushButton(self.w)
        self.button.setText('Record')
        self.button.setStyleSheet('font-size: 20px;')
        self.button.clicked.connect(self.record_start)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.button, 0)
        self.layout.addWidget(self.label, 1)
        
        sd.default.samplerate = 16000
        sd.default.channels = 1
        self.buffer = []
        self.audio1 = None
        self.audio2 = None
        
        path = '../trained_models/xvectors-A-softmax.pt'
        self.model = XVectors()
        self.model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
        self.model = self.model.eval()
        self.EER = 0.7446393072605133

    def start(self):
        self.w.show()
        sys.exit(app.exec_())

    def record_start(self):
        self.button.setText('Stop')
        self.button.clicked.disconnect(self.record_start)
        self.button.clicked.connect(self.record_stop)
        self.istream = sd.InputStream(callback=self.callback)
        self.istream.start()

    def record_stop(self):
        self.button.setText('Record')
        self.button.clicked.disconnect(self.record_stop)
        self.button.clicked.connect(self.record_start)
        self.istream.stop()
        if self.audio1 is None:
            self.audio1 = np.concatenate(self.buffer)
            self.buffer = []
        else:
            self.audio2 = np.concatenate(self.buffer)
            self.buffer = []

            mfcc1 = extract_mfcc(self.audio1.ravel())
            mfcc2 = extract_mfcc(self.audio2.ravel())
            mfcc1 = torch.FloatTensor(np.expand_dims(mfcc1, axis=0))
            mfcc2 = torch.FloatTensor(np.expand_dims(mfcc2, axis=0))
            try:
                with torch.no_grad():
                    y = self.model([mfcc1, mfcc2])
                    embedding1 = y[0, :]
                    embedding2 = y[1, :]
                numer = float(torch.dot(embedding1, embedding2))
                denom = float(torch.norm(embedding1) * torch.norm(embedding2))
                cosine_similarity = np.clip(numer / denom, -1, +1)
                
                result_num = '%.2f\n' % (cosine_similarity * 100)
                if cosine_similarity > self.EER:
                    result_txt = 'Same person'
                    self.label.setStyleSheet('font-size: 20px; color: white; background-color: green')
                else:
                    result_txt = 'Not same person'
                    self.label.setStyleSheet('font-size: 20px; color: white; background-color: red')
                
                self.label.setText('Similarity: ' + result_num + '\n' + result_txt)

            except Exception as e:
                print(e, file=sys.stderr)
                self.label.setText('Similarity: \n?')

            self.audio1 = None
            self.audio2 = None

    def callback(self, indata, frames, time, status):
        self.buffer.append(indata.copy())


if __name__ == '__main__':
    app = DemoApp()
    app.start()

