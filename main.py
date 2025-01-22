import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


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
dataset = MidiDataset(folder_path="data/raw/lofi", seq_length=16)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # shuffle=True üblich beim Training

if len(dataset) == 0:
    print("Keine Daten vorhanden! Überprüfe 'data/raw' oder seq_length.")
    exit(0)

# ----------------------------------------------------------
# 3) Modell instanzieren und auf Device schicken
# ----------------------------------------------------------
model = MusicRNN(input_size=128, hidden_size=256, output_size=128)
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
        # -> Daten auf das Device
        inputs = inputs.to(device)
        targets = targets.to(device)

        batch_size = inputs.size(0)
        # Hidden States
        hidden = model.init_hidden(batch_size)
        h0, c0 = hidden[0].to(device), hidden[1].to(device)
        hidden = (h0, c0)

        # Forward
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    # ----------------------------------------------------------
    # 5) Speichere das trainierte Modell
    # ----------------------------------------------------------
    model_path = "final_model_lofi.pth"  # 1,7897 loss
    torch.save(model.state_dict(), model_path)


print(f"Training abgeschlossen. Modell gespeichert unter: {model_path}")
