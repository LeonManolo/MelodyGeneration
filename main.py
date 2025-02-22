import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import constants
from midi_dataset import MidiDataset
from model import MusicRNN

# ----------------------------------------------------------
# 1) Device festlegen (Apple Silicon MPS, CUDA, CPU)
# ----------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# ----------------------------------------------------------
# 2) Dataset & DataLoader
# ----------------------------------------------------------
dataset = MidiDataset(folder_path="data/folk_music/train", seq_length=constants.SEQUENCE_LENGTH)
# batch size = anzahl an traings sequenzen (seq_length) pro batch
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # shuffle=True üblich beim Training

if len(dataset) == 0:
    print("Keine Daten vorhanden! Überprüfe 'data/raw' oder seq_length.")
    exit(0)

# ----------------------------------------------------------
# 3) Modell instanzieren und auf Device schicken
# ----------------------------------------------------------
model = MusicRNN(input_size=constants.MODEL_INPUT_SIZE,
                 hidden_size=constants.MODEL_HIDDEN_SIZE,
                 output_size=constants.MODEL_OUTPUT_SIZE)
model.to(device)
model.train()

criterion = nn.CrossEntropyLoss()  # Klassifizierungsproblem
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------------------------------------
# 4) Trainingsloop
# ----------------------------------------------------------
print("Starting training")
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)  # (batch_size, seq_length, 256)
        targets = targets.to(device)  # (batch_size, 256) (One-Hot)

        # Hidden State
        batch_size = inputs.size(0)
        hidden = model.init_hidden(batch_size)
        hidden = (hidden[0].to(device), hidden[1].to(device))

        outputs, hidden = model(inputs, hidden)
        # outputs -> (batch_size, 256)

        # 1) Zerlegen in Pitch/Velocity-Logits
        pitch_logits = outputs[:, :128]
        velocity_logits = outputs[:, 128:256]
        note_duration_logits = outputs[:, 256:272]

        # 2) Targets in Indizes umwandeln
        pitch_one_hot = targets[:, :128]
        velocity_one_hot = targets[:, 128:256]
        note_duration_one_hot = targets[:, 256:272]
        pitch_target = torch.argmax(pitch_one_hot, dim=1)
        velocity_target = torch.argmax(velocity_one_hot, dim=1)
        note_duration_target = torch.argmax(note_duration_one_hot, dim=1)

        # 3) Verluste berechnen
        pitch_loss = criterion(pitch_logits, pitch_target)
        velocity_loss = criterion(velocity_logits, velocity_target)
        note_duration_loss = criterion(note_duration_logits, note_duration_target)
        loss = pitch_loss + velocity_loss + note_duration_loss

        # 4) Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    # ----------------------------------------------------------
    # 5) Speichere das trainierte Modell
    # ----------------------------------------------------------
    model_path = "final_model_lofi.pth"  # 1,7897 loss
    torch.save(model.state_dict(), model_path)

print(f"Training abgeschlossen. Modell gespeichert unter: {model_path}")
