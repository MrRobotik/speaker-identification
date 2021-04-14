from os import listdir
import torch
import numpy as np


class DatasetBase:

    def __init__(self, data_folder, feats_type):
        self.data = []
        self.speaker_data = []
        suffix = feats_type + '.npy'
        speaker_label = 0
        for speaker_folder in listdir(data_folder):
            self.speaker_data.append([])
            for session_folder in listdir(data_folder / speaker_folder):
                for features_file in listdir(data_folder / speaker_folder / session_folder):
                    if features_file.rpartition('-')[2] == suffix:
                        path = data_folder / speaker_folder / session_folder / features_file
                        self.data.append((path, speaker_label))
                        self.speaker_data[speaker_label].append(len(self.data) - 1)
            speaker_label += 1

    def __len__(self):
        return len(self.data)
    
    def speakers_count(self):
        return len(self.speaker_data)


class DatasetClasses(DatasetBase):

    def __init__(self, data_folder, feats_type, batch_size, device, min_sample_length):
        super().__init__(data_folder, feats_type)
        self.batch_size = batch_size
        self.device = device
        self.min_sample_length = min_sample_length
        self.cross_val_indices = []
        for utters in self.speaker_data:
            i = utters[0]
            self.cross_val_indices.append(i)

        cvis = set(self.cross_val_indices)
        self.train_indices = [i for i in range(len(self.data)) if i not in cvis]
        self.cross_val_indices = np.asarray(self.cross_val_indices)
        self.train_indices = np.asarray(self.train_indices)

    def __iter__(self):
        indices = np.random.permutation(self.train_indices)
        for batch_start in range(0, len(indices), self.batch_size):
            batch_end = batch_start + self.batch_size
            inputs = []
            labels = []
            for i in indices[batch_start:batch_end]:
                path, t = self.data[i]
                try:
                    x = torch.FloatTensor(np.expand_dims(np.load(path), axis=0))
                except Exception:
                    continue
                if x.size()[2] < self.min_sample_length:
                    continue
                inputs.append(x.to(self.device))
                labels.append(t)
            yield inputs, labels

    def cross_val(self):
        labels = []
        inputs = []
        for i in self.cross_val_indices:
            path, t = self.data[i]
            try:
                x = torch.FloatTensor(np.expand_dims(np.load(path), axis=0))
            except Exception:
                continue
            if x.size()[2] < self.min_sample_length:
                continue
            inputs.append(x.to(self.device))
            labels.append(t)
        return inputs, labels


class DatasetTriplets(DatasetBase):

    def __init__(self, data_folder, feats_type, batch_size, device, min_sample_length):
        super().__init__(data_folder, feats_type)
        self.batch_size = batch_size
        self.device = device
        self.min_sample_length = min_sample_length
        self.cross_val_indices = []
        for utters in self.speaker_data:
            i = utters[0]
            j = utters[1]
            self.cross_val_indices.append((i, j))

        cvis = set(self.cross_val_indices)
        self.train_indices = [i for i in range(len(self.data)) if i not in cvis]
        self.cross_val_indices = np.asarray(self.cross_val_indices)
        self.train_indices = np.asarray(self.train_indices)

    def __iter__(self):
        indices = np.random.permutation(self.train_indices)
        for batch_start in range(0, len(indices), self.batch_size):
            batch_end = batch_start + self.batch_size
            anchors = []
            positives = []
            negatives = []

            for i in indices[batch_start:batch_end]:
                anc_path, t_pos = self.data[i]
                t_neg = np.random.randint(0, len(self.speaker_data))
                while t_neg == t_pos:
                    t_neg = np.random.randint(0, len(self.speaker_data))

                pos_utters = self.speaker_data[t_pos]
                neg_utters = self.speaker_data[t_neg]
                pos_path = self.data[pos_utters[np.random.randint(0, len(pos_utters))]]
                neg_path = self.data[neg_utters[np.random.randint(0, len(neg_utters))]]
                try:
                    x_anc = torch.FloatTensor(np.expand_dims(np.load(anc_path), axis=0))
                    x_pos = torch.FloatTensor(np.expand_dims(np.load(pos_path), axis=0))
                    x_neg = torch.FloatTensor(np.expand_dims(np.load(neg_path), axis=0))
                except Exception:
                    continue
                if x_anc.size()[2] < self.min_sample_length:
                    continue
                if x_pos.size()[2] < self.min_sample_length:
                    continue
                if x_neg.size()[2] < self.min_sample_length:
                    continue

                anchors.append(x_anc)
                positives.append(x_pos)
                negatives.append(x_neg)

            yield anchors, positives, negatives

    def cross_val(self):
        pass  # TODO: implement me
