import torch
import torch.nn as nn
import pretty_midi

import constants
from midi_dataset import MidiDataset  # zum Seed-Laden oder Notenformat
from model import MusicRNN

# ----------------------------------------------------------
# 1) Device festlegen
# ----------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# ----------------------------------------------------------
# 2) Modell laden
# ----------------------------------------------------------
model = MusicRNN(input_size=constants.MODEL_INPUT_SIZE,
                 hidden_size=constants.MODEL_HIDDEN_SIZE,
                 output_size=constants.MODEL_OUTPUT_SIZE)
print(model)
loaded_state_dict = torch.load("saved_model.pth", map_location=device)
model.load_state_dict(loaded_state_dict)
model.to(device)
model.eval()

def map_value_to_velocity(velocity_index):
    velocity_values = [49, 64, 80, 96, 112]
    if velocity_index < len(velocity_values):
        return velocity_values[velocity_index] # returns the index of the velocity value
    else:
        print("Error finding correct velocity, out of bounds!")
        return 80 # 80 most popular velocity value (around 80%)

# ----------------------------------------------------------
# 3) Funktionen zum Generieren & MIDI-Erstellen
# ----------------------------------------------------------
def generate_notes(model, seed_sequence, num_notes=50):
    """
    Erzeugt 'num_notes' neue Noten (One-hot) basierend auf einem Seed (1, seq_len, 128).
    """
    model.eval()
    device = next(model.parameters()).device

    # Sicherstellen, dass seed_sequence ein Tensor auf dem richtigen Device ist
    if not torch.is_tensor(seed_sequence):
        seed_sequence = torch.tensor(seed_sequence, dtype=torch.float32)
    seed_sequence = seed_sequence.unsqueeze(0) if seed_sequence.ndim == 2 else seed_sequence
    # Jetzt hat seed_sequence min. shape (1, seq_len, 128)
    seed_sequence = seed_sequence.to(device)

    generated_notes = []

    # Hidden init (batch=1)
    batch_size = 1
    hidden = model.init_hidden(batch_size)
    h0, c0 = hidden[0].to(device), hidden[1].to(device)
    hidden = (h0, c0)

    current_seq = seed_sequence  # shape (1, seq_len, 128)

    with torch.no_grad():
        for _ in range(num_notes):
            outputs, hidden = model(current_seq, hidden)  # (1,128)
            predicted_pitch_idx = torch.argmax(outputs[:, :128], dim=1).item()
            predicted_velocity_idx = torch.argmax(outputs[:, 128:133], dim=1).item()
            predicted_note_duration_idx = torch.argmax(outputs[:, 133:149], dim=1).item()

            # One-hot
            new_note = torch.zeros(constants.MODEL_INPUT_SIZE, dtype=torch.float32, device=device)
            new_note[predicted_pitch_idx] = 1.0
            new_note[128 + predicted_velocity_idx] = 1.0
            new_note[133 + predicted_note_duration_idx] = 1.0

            generated_notes.append(new_note.cpu())

            # Fenster verschieben: wir nehmen current_seq[:, 1:] + new_note
            keep_part = current_seq[:, 1:, :]  # (1, seq_len-1, 128)
            new_note_expanded = new_note.unsqueeze(0).unsqueeze(0)  # (1,1,128)
            current_seq = torch.cat([keep_part, new_note_expanded], dim=1)

    return generated_notes


# Wandelt eine Liste von One-hot-Noten in ein MIDI-File. Jede Note bekommt feste Dauer 'note_duration'.
def notes_to_midi(one_hot_notes, filename="generated.mid"):
    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano_track = pretty_midi.Instrument(program=piano_program)

    current_time = 0.0
    for one_hot_vec in one_hot_notes:
        pitch_one_hot_vec = one_hot_vec[:128]  # 256 values and the first 128 values represent the pitch
        velocity_one_hot_vec = one_hot_vec[128:133]  # 256 values and the last 128 values represent the velocity
        note_duration_one_hot_vec = one_hot_vec[133:149]  # 256 values and the last 128 values represent the velocity
        pitch_idx = torch.argmax(pitch_one_hot_vec).item()
        velocity_idx = torch.argmax(velocity_one_hot_vec).item()
        note_duration_idx = torch.argmax(note_duration_one_hot_vec).item()

        # probabilities = torch.softmax(output, dim=1) # in softmax reinschauen, notiz: (Variation autoencoder in den Folien, das aufteilen anschauen)
        # pitch_idx = torch.multinomial(probabilities, 1).item()

        note_duration = note_duration_idx * 0.125

        note = pretty_midi.Note(
            velocity=map_value_to_velocity(velocity_idx),
            pitch=pitch_idx,
            start=current_time,
            end=current_time + note_duration
        )
        piano_track.notes.append(note)
        current_time += note_duration

    pm.instruments.append(piano_track)
    pm.write(filename)
    print(f"MIDI-Datei '{filename}' gespeichert (Notenanzahl: {len(one_hot_notes)}).")


# ----------------------------------------------------------
# 4) Seed-Sequenz laden & Noten generieren
# ----------------------------------------------------------
# TOdo: Einfach leeres Dataset anlegen, um ggf. an seeds zu kommen


dataset = MidiDataset(folder_path="data/folk_music/evaluation", seq_length=constants.SEQUENCE_LENGTH)
if len(dataset) == 0:
    print("Warnung: Dataset leer. Keine Seeds verfügbar.")
    # Evtl. einfach Zufalls-Seed generieren oder beenden
    # Hier beenden wir für's Beispiel.
    exit(0)
else:
    seed_x, _ = dataset[5]  # shape (16,256)
    # Generiere 50 neue Noten

    # custom
    # one_hot_map = MidiDataset.pitch_to_one_hot()
    # custom_notes = [57, 60, 53, 62, 55, 58, 52, 50, 62, 58, 60, 57, 55, 53, 52, 50]
    # seed_x_2 = [one_hot_map[pitch] for pitch in custom_notes]

    generated = generate_notes(model, seed_x, num_notes=100)
    notes_to_midi(generated, filename="output.mid")
