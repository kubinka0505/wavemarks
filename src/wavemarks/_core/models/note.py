"""
Pitch / key name conversion utilities.

MIDI note numbering: C0 = 12 (standard convention, matches certain software's tlst encoding).
Octave range supported: C0 through G10 (MIDI 12 through 127).
"""

#-=-=-=-#

_NOTE_OFFSETS = {
	"C":  0,
	"C#": 1, "DB": 1,
	"D":  2,
	"D#": 3, "EB": 3,
	"E":  4,
	"F":  5,
	"F#": 6, "GB": 6,
	"G":  7,
	"G#": 8, "AB": 8,
	"A":  9,
	"A#": 10, "BB": 10,
	"B":  11,
}

_REVERSE_NOTE_NAMES = {
	0: "C",
	1: "C#",
	2: "D",
	3: "D#",
	4: "E",
	5: "F",
	6: "F#",
	7: "G",
	8: "G#",
	9: "A",
	10: "A#",
	11: "B",
}

MIDI_MIN = 12  # C0
MIDI_MAX = 127 # G10

class Note:
	"""
	Note-name ↔ MIDI-number conversion.

	Usage:
		Note.encode("C0")  -> 12
		Note.encode("E4")  -> 64
		Note.encode("Gb1") -> 30    (same as F#1)
		Note.decode(64)    -> "E4"
		Note.decode(30)    -> "F#1" (sharps preferred over flats on decode (A#1 is more readable than Bb1))
	"""

	@staticmethod
	def encode(name: str) -> int:
		"""
		Converts a note name to a MIDI note number.

		Uses the convention C0 = 12 (matching certain software's tlst root_note encoding).

		Accepts sharps (#) and flats (b/B) as accidentals.

		Enharmonic equivalents resolve to the same integer, e.g. "Gb1" and "F#1" both return 30.

		Parameters
		----------
			name (str):
				A note name string consisting of:
				- letter (A–G),
				- optional accidental (# <sharp> or b <flat>)
				- octave number (0–10)

				e.g. "C0", "F#4", "Gb1", "A#10"

		Returns
		-------
			int:
				MIDI note number in the range 12 (C0) to 127 (G10).

		Raises
		------
			ValueError:
				If name is:
				- malformed
				- uses an unknown note letter
				- has an invalid octave
				- maps to a MIDI number outside the supported range (C0–G10 / 12–127).
		"""
		if not name or len(name) < 2:
			raise ValueError(f"Invalid note name: {name!r}")

		name = name.strip()

		# split letter+accidental from octave digits
		i = 1
		if len(name) > 1 and name[1] in "#bB" and not name[1].isdigit():
			i = 2

		note_part   = name[:i].upper()
		octave_part = name[i:]

		if note_part not in _NOTE_OFFSETS:
			raise ValueError(f"Unknown note letter: {name!r}")

		try:
			octave = int(octave_part)
		except ValueError:
			raise ValueError(f"Invalid octave in note name: {name!r}")

		# MIDI: C-1 = 0 standard, but certain software/tlst uses C0 = 12 (offset +1 octave)
		midi = (octave + 1) * 12 + _NOTE_OFFSETS[note_part]

		if not (MIDI_MIN <= midi <= MIDI_MAX):
			raise ValueError(
				f"Note {name!r} (MIDI {midi}) is out of supported range "
				f"C0–G10 ({MIDI_MIN}–{MIDI_MAX})"
			)

		return midi

	@staticmethod
	def decode(midi: int) -> str:
		"""
		Converts a MIDI note number to a note name string.

		Uses sharps in preference to flats for enharmonic notes (e.g. 30 -> "F#1" rather than "Gb1").

		Parameters
		----------
			midi (int):
				A MIDI note number in the range 12 (C0) to 127 (G10).

		Returns
		-------
			str:
				The note name string, e.g. "E4", "F#1", "C0".

		Raises
		------
			ValueError:
				If midi is outside the supported range 12–127 (C0–G10).
		"""
		if not (MIDI_MIN <= midi <= MIDI_MAX):
			raise ValueError(
				f"MIDI note {midi} is out of supported range C0–G10 "
				f"({MIDI_MIN}–{MIDI_MAX})"
			)

		octave = (midi // 12) - 1
		note   = _REVERSE_NOTE_NAMES[midi % 12]

		return f"{note}{octave}"

#-=-=-=-#

def _populate_key_constants() -> None:
	"""
	Populates Note class attributes with MIDI note constants at import time.

	Iterates over all valid MIDI note numbers and creates a class attribute
	for every supported note spelling.

	Sharps are represented with "s" instead of "#" to produce valid
	identifiers (e.g. "Note.Cs4" for C#4), while flats use "b" directly
	(e.g. "Note.Db4" for Db4).

	Enharmonic spellings are exposed as wavemarks that reference the same MIDI
	note, allowing lookups such as "Note.Cs4 == Note.Db4".
	"""
	for midi in range(MIDI_MIN, MIDI_MAX + 1):
		octave = midi // 12 - 1
		note = midi % 12

		for name, offset in _NOTE_OFFSETS.items():
			if offset != note:
				continue

			# C#4 -> Cs4, Db4 stays Db4
			identifier = name.title().replace("#", "s") + str(octave)
			setattr(Note, identifier, midi)

_populate_key_constants()