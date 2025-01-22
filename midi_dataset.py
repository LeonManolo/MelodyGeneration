import os
import glob
import pretty_midi
import numpy as np
from torch.utils.data import Dataset


class MidiDataset(Dataset):
    def __init__(self, folder_path, seq_length=16):
        self.seq_length = seq_length
        self.pitch_to_onehot = self.pitch_to_one_hot()

        # Hier sammeln wir **alle** Sequenzen
        # Jede Sequenz hat die Form (seq_length, 128) wenn 128 One-hot-Spalten.
        # plus ein Target (128) oder ein Index, je nachdem, wie du es willst.
        self.sequences = []
        self.targets = []

        # (A) Alle MIDI-Dateien laden
        midi_files = glob.glob(os.path.join(folder_path, "*.mid"))

        for file_path in midi_files:
            print(file_path)
            # (B) Noten (pitch) aus MIDI extrahieren
            note_sequence = self._parse_midi(file_path)  # z.B. Liste von {'pitch': ...}

            # (C) One-hot umwandeln
            one_hot_list = []
            for note in note_sequence:
                pitch = note['pitch']
                # hole den 128-dimensionalen Vektor
                one_hot_vec = self.pitch_to_onehot[pitch]
                one_hot_list.append(one_hot_vec)

            # (D) In Sequenzen aufteilen
            # => z.B. (16 Eingabeschritte) -> 1 Zielschritt
            for i in range(len(one_hot_list) - seq_length):
                seq_input = one_hot_list[i: i + seq_length]  # 16 x 128
                seq_target = one_hot_list[i + seq_length]  # 128

                self.sequences.append(seq_input)
                self.targets.append(seq_target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Hier geben wir (input, target) als numpy arrays oder Tensors zurück
        x = np.array(self.sequences[idx], dtype=np.float32)  # shape: (seq_length, 128)
        y = np.array(self.targets[idx], dtype=np.float32)  # shape: (128,)
        return x, y

    @staticmethod
    def _parse_midi(file_path):
        # Holt die Noten aus dem MIDI
        midi_data = pretty_midi.PrettyMIDI(file_path)
        instruments = midi_data.instruments
        note_seq = []
        for inst in instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    note_seq.append({
                        'pitch': note.pitch,
                        'start': note.start,
                        'end': note.end,
                        'velocity': note.velocity
                    })
        # sortieren
        note_seq.sort(key=lambda n: n['start'])
        return note_seq

    @staticmethod
    def pitch_to_one_hot():
        one_hot_map = {}
        for pitch in range(128):
            vec = [0] * 128
            vec[pitch] = 1
            one_hot_map[pitch] = vec
        return one_hot_map

# [0,0,1] A2 [[0,0,1],[0,0,1]] 23
# [0,1,0] A3
# [1,0,0] A4