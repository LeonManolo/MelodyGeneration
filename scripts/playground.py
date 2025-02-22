import pretty_midi


def remove_chords_exact(notes):
    """
    Gruppiert Noten anhand ihres exakten Startzeitpunkts (ohne Toleranz)
    und behält aus jeder Gruppe die Note mit dem höchsten Pitch.

    Parameter:
      notes: Liste von PrettyMIDI-Note-Objekten.

    Rückgabe:
      Liste gefilterter Note-Objekte.
    """
    notes_by_start = {}
    for note in notes:
        key = note.start  # exakter Startzeitpunkt
        if key not in notes_by_start:
            notes_by_start[key] = []
        notes_by_start[key].append(note)

    filtered_notes = []
    for key in sorted(notes_by_start.keys()):
        group = notes_by_start[key]
        best_note = max(group, key=lambda n: n.pitch)
        filtered_notes.append(best_note)

    return filtered_notes


def show_first_10_notes(midi_file_path):
    """
    Liest eine MIDI-Datei mit PrettyMIDI ein, gruppiert Noten, die exakt denselben
    Startzeitpunkt haben, und gibt die ersten 10 gefilterten Noten aus.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_file_path)
    except Exception as e:
        print(f"Fehler beim Laden der MIDI-Datei: {e}")
        return

    all_notes = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        all_notes.extend(instrument.notes)

    # Sortiere alle Noten nach Startzeit
    all_notes.sort(key=lambda n: n.start)

    # Akkorde entfernen: Nur bei exakt gleichen Startzeiten wird die höchste Note behalten.
    filtered_notes = remove_chords_exact(all_notes)

    # Ausgabe der ersten 10 gefilterten Noten
    for i, note in enumerate(filtered_notes[:10]):
        note_name = pretty_midi.note_number_to_name(note.pitch)
        print(f"Note {i + 1}: Pitch={note.pitch} ({note_name}), "
              f"Start={note.start:.3f}, End={note.end:.3f}, Velocity={note.velocity}")




