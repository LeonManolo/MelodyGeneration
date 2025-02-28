import os
import glob
import pretty_midi
import numpy as np
from torch.utils.data import Dataset

import constants


def quantize_duration(duration, interval=0.125):
    return min(int(round(duration / interval)), 15) # für werte > 1.875


def map_velocity_to_value(velocity):
    velocity_values = [49, 64, 80, 96, 112]
    if velocity in velocity_values:
        return velocity_values.index(velocity) # returns the index of the velocity value
    else:
        return 2 # 80 most popular velocity value (around 80%)


def note_to_one_hot(note):
    one_hot_encoded_note = [0] * constants.MODEL_INPUT_SIZE
    pitch = note['pitch'] # value between 0 - 127, representing the pitch in midi
    #velocity = note['velocity'] # value between 0 - 127, representing the velocity in midi
    velocity = map_velocity_to_value(note['velocity'])
    one_hot_encoded_note[pitch] = 1 # first 128 values represent the pitch
    one_hot_encoded_note[constants.NUM_OF_NOTE_VALUES +  velocity] = 1 # next 128 values represent the velocity

    note_duration = note['end'] - note['start']
    note_duration_interval_idx = quantize_duration(note_duration)
    one_hot_encoded_idx = constants.NUM_OF_NOTE_VALUES + constants.NUM_OF_VELOCITY_VALUES + note_duration_interval_idx
    one_hot_encoded_note[one_hot_encoded_idx] = 1
    return one_hot_encoded_note


class MidiDataset(Dataset):
    def __init__(self, folder_path, seq_length=16):
        self.seq_length = seq_length

        # Hier sammeln wir **alle** Sequenzen
        # plus ein Target
        self.sequences = []
        self.targets = []

        midi_files = glob.glob(os.path.join(folder_path, "*.mid"))

        for file_path in midi_files:
            print(file_path)
            # Noten (pitch) aus MIDI extrahieren
            note_sequence = self._parse_midi(file_path)

            # One-hot umwandeln
            one_hot_list = []
            for note in note_sequence:
                one_hot_encoded_note = note_to_one_hot(note)
                one_hot_list.append(one_hot_encoded_note) # 256

            # In Sequenzen aufteilen
            # (16 Eingabeschritte) -> 1 Zielschritt
            for i in range(len(one_hot_list) - seq_length):
                seq_input = one_hot_list[i: i + seq_length]  # sequence x output_size
                seq_target = one_hot_list[i + seq_length]  # output size

                self.sequences.append(seq_input)
                self.targets.append(seq_target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
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

        note_seq.sort(key=lambda n: n['start'])
        return note_seq


# [0,0,1] A2 [[0,0,1],[0,0,1]] 23
# [0,1,0] A3
# [1,0,0] A4