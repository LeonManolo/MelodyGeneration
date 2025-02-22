import os
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt


def collect_note_durations(directory):
    """
    Geht rekursiv durch den angegebenen Ordner, lädt alle MIDI-Dateien
    und sammelt die Dauer (in Sekunden) jeder Note.
    """
    durations = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                file_path = os.path.join(root, file)
                try:
                    pm = pretty_midi.PrettyMIDI(file_path)
                    for instrument in pm.instruments:
                        for note in instrument.notes:
                            duration = note.end - note.start
                            durations.append(duration)
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {file_path}: {e}")
    return durations


def compute_statistics(durations):
    """
    Berechnet grundlegende Statistiken über die Notendauern:
    Minimum, Maximum, Mittelwert, Median, Standardabweichung und Perzentile.
    """
    durations = np.array(durations)
    return {
        'min': np.min(durations),
        'max': np.max(durations),
        'mean': np.mean(durations),
        'median': np.median(durations),
        'std': np.std(durations),
        'percentiles': np.percentile(durations, [25, 50, 75])
    }


def plot_histogram(durations, bins=50):
    """
    Erstellt ein Histogramm der Notendauern.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel('Notendauer (Sekunden)')
    plt.ylabel('Anzahl Noten')
    plt.title('Histogramm der Notendauern')
    plt.show()


# Passe hier den Pfad zu deinem MIDI-Dataset an
dataset_directory = "/Users/manolo/Downloads/nottingham-dataset-master/MIDI/melody"

# Notendauern sammeln
durations = collect_note_durations(dataset_directory)

# Statistiken berechnen
stats = compute_statistics(durations)

print("Statistik der Notendauern (in Sekunden):")
print(f"Minimum: {stats['min']:.3f}")
print(f"Maximum: {stats['max']:.3f}")
print(f"Mittelwert: {stats['mean']:.3f}")
print(f"Median: {stats['median']:.3f}")
print(f"Standardabweichung: {stats['std']:.3f}")
print(f"25. Perzentil: {stats['percentiles'][0]:.3f}")
print(f"50. Perzentil: {stats['percentiles'][1]:.3f}")
print(f"75. Perzentil: {stats['percentiles'][2]:.3f}")

# Histogramm anzeigen
plot_histogram(durations, bins=50)
