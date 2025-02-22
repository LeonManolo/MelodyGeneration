import os
import pretty_midi
from collections import Counter

def get_time_signature(midi_file):
    """
    Liest eine MIDI-Datei ein und gibt den ersten Taktwechsel als String zurück.
    Falls keine Taktwechsel vorhanden sind, wird standardmäßig "4/4" angenommen.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {midi_file}: {e}")
        return None

    

    if midi_data.time_signature_changes:
        # Ersten Taktwechsel als repräsentativen Takt verwenden
        ts = midi_data.time_signature_changes[0]
        ts_str = f"{ts.numerator}/{ts.denominator}"
    else:
        print("NOT FOUND")
        ts_str = "4/4"
    return ts_str

def analyze_directory(start_directory):
    ts_counter = Counter()
    for root, dirs, files in os.walk(start_directory):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                file_path = os.path.join(root, file)
                ts = get_time_signature(file_path)
                print("Takt: " + ts + " filepath: " + file_path)
                if ts is not None:
                    ts_counter[ts] += 1
    return ts_counter

def analyze_midi_signatures():
    # Passe den Pfad zum Ordner an: Hier wird der Ordner "MeineMIDIs" im Downloads-Ordner verwendet.
    start_directory = os.path.expanduser("~/Downloads/maestro-v3.0.0")
    ts_counter = analyze_directory(start_directory)

    if ts_counter:
        most_common_ts, count = ts_counter.most_common(1)[0]
        print(f"Am häufigsten vorkommender Takt: {most_common_ts} ({count} Vorkommen)")
        print("\nÜbersicht aller gefundenen Takte:")
        for ts, cnt in ts_counter.items():
            print(f"{ts}: {cnt}")
    else:
        print("Keine MIDI-Dateien gefunden.")

analyze_midi_signatures()