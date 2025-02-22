import os
import pretty_midi


def count_notes_and_duration(directory):
    total_notes = 0
    total_duration = 0.0  # in Sekunden

    # Gehe rekursiv durch alle Dateien im Ordner
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                file_path = os.path.join(root, file)
                try:
                    pm = pretty_midi.PrettyMIDI(file_path)
                    # Zähle alle Noten in allen Instrumenten
                    file_notes = sum(len(instr.notes) for instr in pm.instruments)
                    total_notes += file_notes
                    # pm.get_end_time() gibt die Gesamtdauer (in Sekunden) der MIDI-Datei zurück
                    total_duration += pm.get_end_time()
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {file_path}: {e}")

    return total_notes, total_duration


notes, duration_sec = count_notes_and_duration("/Users/manolo/Downloads/nottingham-dataset-master/MIDI/melody")
duration_min = duration_sec / 60.0

print(f"Gesamtanzahl der Noten: {notes}")
print(f"Gesamtdauer des Datasets: {duration_min:.2f} Minuten")


