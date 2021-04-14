from os import listdir


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
