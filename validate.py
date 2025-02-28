import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import constants
from midi_dataset import MidiDataset
from model import MusicRNN

# Gerätedefinition
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)


test_dataset = MidiDataset(folder_path="data/folk_music/evaluation", seq_length=constants.SEQUENCE_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, shuffle=False)

if len(test_dataset) == 0:
    print("Keine Testdaten vorhanden!")
    exit(0)

# Modell laden und in den Evaluationsmodus versetzen
model = MusicRNN(input_size=constants.MODEL_INPUT_SIZE,
                 hidden_size=constants.MODEL_HIDDEN_SIZE,
                 output_size=constants.MODEL_OUTPUT_SIZE)
model.load_state_dict(torch.load("saved_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss()

# Mtriken
total_loss = 0.0
total_pitch_correct = 0
total_velocity_correct = 0
total_duration_correct = 0
total_samples = 0


with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        batch_size = inputs.size(0)
        hidden = model.init_hidden(batch_size)
        hidden = (hidden[0].to(device), hidden[1].to(device))

        outputs, hidden = model(inputs, hidden)

        # Aufteilung der Ausgaben in die jeweiligen Bereiche
        pitch_logits = outputs[:, :128]
        velocity_logits = outputs[:, 128:133]
        duration_logits = outputs[:, 133:149]


        pitch_target = torch.argmax(targets[:, :128], dim=1)
        velocity_target = torch.argmax(targets[:, 128:133], dim=1)
        duration_target = torch.argmax(targets[:, 133:149], dim=1)

        # Verlustberechnung
        pitch_loss = criterion(pitch_logits, pitch_target)
        velocity_loss = criterion(velocity_logits, velocity_target)
        duration_loss = criterion(duration_logits, duration_target)
        loss = pitch_loss + velocity_loss + duration_loss
        total_loss += loss.item() * batch_size


        pitch_pred = torch.argmax(pitch_logits, dim=1)
        velocity_pred = torch.argmax(velocity_logits, dim=1)
        duration_pred = torch.argmax(duration_logits, dim=1)

        total_pitch_correct += (pitch_pred == pitch_target).sum().item()
        total_velocity_correct += (velocity_pred == velocity_target).sum().item()
        total_duration_correct += (duration_pred == duration_target).sum().item()
        total_samples += batch_size

# Durchschnittlichen Verlust und Genauigkeiten berechnen
avg_loss = total_loss / total_samples
pitch_accuracy = total_pitch_correct / total_samples
velocity_accuracy = total_velocity_correct / total_samples
duration_accuracy = total_duration_correct / total_samples

print(f"Test Loss: {avg_loss:.4f}")
print(f"Pitch Accuracy: {pitch_accuracy:.4f}")
print(f"Velocity Accuracy: {velocity_accuracy:.4f}")
print(f"Duration Accuracy: {duration_accuracy:.4f}")


# Metriken und ihre Werte (hier: durchschnittlicher Loss und Genauigkeiten)
metrics = ['Pitch Accuracy', 'Velocity Accuracy', 'Duration Accuracy']
values = [ pitch_accuracy, velocity_accuracy, duration_accuracy]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values)


for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')

plt.title("Evaluation Metrics")
plt.ylim(0, max(values)*1.2)
plt.ylabel("Wert")
plt.show()
