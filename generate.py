import torch
import torch.nn as nn
import pretty_midi
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
model = MusicRNN(input_size=128, hidden_size=256, output_size=128)
print(model)
exit(1)
loaded_state_dict = torch.load("final_model_lofi.pth", map_location=device, weights_only=True)
model.load_state_dict(loaded_state_dict)
model.to(device)
model.eval()


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
            predicted_idx = torch.argmax(outputs, dim=1).item()

            # One-hot
            new_note = torch.zeros(128, dtype=torch.float32, device=device)
            new_note[predicted_idx] = 1.0

            generated_notes.append(new_note.cpu())

            # Fenster verschieben: wir nehmen current_seq[:, 1:] + new_note
            keep_part = current_seq[:, 1:, :]  # (1, seq_len-1, 128)
            new_note_expanded = new_note.unsqueeze(0).unsqueeze(0)  # (1,1,128)
            current_seq = torch.cat([keep_part, new_note_expanded], dim=1)

    return generated_notes


# Wandelt eine Liste von One-hot-Noten in ein MIDI-File. Jede Note bekommt feste Dauer 'note_duration'.
def notes_to_midi(one_hot_notes, filename="generated.mid", note_duration=0.4):
    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano_track = pretty_midi.Instrument(program=piano_program)

    current_time = 0.0
    # [[0,0,0,0,0,0.89,...,0,0], [0,0,0.55,0,0,0.89,...,0,0]]
    for one_hot_vec in one_hot_notes:
        pitch_idx = torch.argmax(one_hot_vec).item()
        # probabilities = torch.softmax(output, dim=1) # in softmax reinschauen, notiz: (Variation autoencoder in den Folien, das aufteilen anschauen)
        # pitch_idx = torch.multinomial(probabilities, 1).item()
        note = pretty_midi.Note(
            velocity=64,
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
try:
    dataset = MidiDataset(folder_path="data/raw", seq_length=16)
    if len(dataset) == 0:
        print("Warnung: Dataset leer. Keine Seeds verfügbar.")
        # Evtl. einfach Zufalls-Seed generieren oder beenden
        # Hier beenden wir für's Beispiel.
        exit(0)
    else:
        #seed_x, _ = dataset[1]  # shape (16,128)
        # Generiere 50 neue Noten

        # custom
        one_hot_map = MidiDataset.pitch_to_one_hot()
        custom_notes = [57, 60, 53, 62, 55, 58, 52, 50, 62, 58, 60, 57, 55, 53, 52, 50]
        seed_x_2 = [one_hot_map[pitch] for pitch in custom_notes]
        generated = generate_notes(model, seed_x_2, num_notes=100)
        notes_to_midi(generated, filename="my_new_song_2.midi", note_duration=0.5)

except Exception as e:
    print("Fehler beim Laden des Seeds:", e)
    print("Generiere kein neues MIDI.")
