from ..helpers import _read_chunk_header, _pad

from ..models.type import MarkerType
from ..models.entry import Entry
from ..models.tempo import TempoMode
from ..models.note import Note

import io
import re
import struct
from typing import List, Tuple, Union, Optional

#-=-=-=-#
# Utils

_TIME_SIG_RE = re.compile(r"^(\d+)/(\d+)$")

_DEFAULT_ACID = dict(
	type_of_file      = TempoMode.FREE, # 0x04, defaults to Free unless overridden
	root_note         = 60,             # middle C
	unknown1          = 0,
	unknown2          = 0,
	num_beats         = 0,
	meter_denominator = 4,
	meter_numerator   = 4,
	tempo             = 120.0,
	extra             = b"",
)

def _parse_time_signature(ts: str) -> Tuple[int, int]:
	"""
	Parses a "N/D" time signature string into (numerator, denominator).

	Parameters
	----------
		ts (str):
			Time signature as "numerator/denominator", e.g. "4/4".

	Returns
	-------
		Tuple[int, int]:
			(numerator, denominator)

	Raises
	------
		ValueError:
			If `ts` doesn't match the "N/D" pattern, or either part is
			zero (a zero denominator/numerator isn't a valid signature).
	"""
	match = _TIME_SIG_RE.match(ts.strip())

	if not match:
		raise ValueError(f'Invalid time_signature {ts!r}; expected format like "4/4"')

	numerator, denominator = int(match.group(1)), int(match.group(2))

	if numerator == 0 or denominator == 0:
		raise ValueError(f"time_signature numerator/denominator must be nonzero, got {ts!r}")

	return numerator, denominator

def _apply_acid_overrides(
	fields: dict,
	*,

	synced:         Optional[bool]            = None,
	note_root:      Optional[Union[int, str]] = None,
	time_signature: Optional[str]             = None,
	tempo:          Optional[float]           = None,
	num_beats:      Optional[int]             = None,
) -> dict:
	"""
	Applies override kwargs to an acid-fields dict in place and returns it.

	Shared by _build_chunk_acid (at write time) and MarkerFile.save()'s
	post-write state sync, so the override logic only lives in one place.
	"""
	if synced is not None:
		fields["type_of_file"] = (fields["type_of_file"] & ~0x01) if synced else (fields["type_of_file"] | 0x01)
		fields["synced"] = synced

	if note_root is not None:
		fields["root_note"] = _resolve_note_root(note_root)

	if time_signature is not None:
		fields["meter_numerator"], fields["meter_denominator"] = _parse_time_signature(time_signature)

	if tempo is not None:
		fields["tempo"] = float(tempo)

	if num_beats is not None:
		fields["num_beats"] = int(num_beats)

	return fields

def _resolve_note_root(note_root: Union[int, str]) -> int:
	"""
	Normalizes a note_root argument to a raw MIDI-style int for the acid
	chunk's root_note field.

	Parameters
	----------
		note_root (Union[int, str]):
			Either an already-encoded int (e.g. Note.Cb6, if that constant
			is an int) or a note name string (e.g. "Cb6"), which is run
			through Note.encode.

	Returns
	-------
		int:
			The encoded root note value.
	"""
	if isinstance(note_root, str):
		return Note.encode(note_root)

	return int(note_root)

#-=-=-=-#
# Main

def _build_chunk_cue(markers: List[Entry]) -> bytes:
	"""
	Builds a complete cue chunk (header + body) from a list of markers.

	Loop entries are excluded, since loops are represented in
	the `smpl` chunk rather than the cue chunk.

	Parameters
	----------
		markers (List[Entry]):
			All Entries belonging to the file.

	Returns
	-------
		bytes:
			The fully encoded "cue " chunk, including its 8-byte header and even-padded body.
	"""
	cue_markers = [m for m in markers if not m.is_loop]

	body = struct.pack("<I", len(cue_markers))

	for m in cue_markers:
		body += struct.pack(
			"<II4sIII",
			m.id,
			m.start,
			b"data",
			0,
			0,
			m.start
		)

	if len(body) % 2:
		body += b"\x00"

	return b"cue " + struct.pack("<I", len(body)) + body

def _build_chunk_acid(
	acid: Optional[dict],
	*,

	synced:         Optional[bool]            = None,
	note_root:      Optional[Union[int, str]] = None,
	time_signature: Optional[str]             = None,
	tempo:          Optional[float]           = None,
	num_beats:      Optional[int]             = None,
) -> bytes:
	"""
	(docstring unchanged from before)
	"""
	if not acid and not any(
		v is not None for v in (synced, note_root, time_signature, tempo, num_beats)
	):
		return b""

	fields = dict(_DEFAULT_ACID) if not acid else dict(acid)
	fields = _apply_acid_overrides(
		fields,
		synced         = synced,
		note_root      = note_root,
		time_signature = time_signature,
		tempo          = tempo,
		num_beats      = num_beats,
	)

	body = struct.pack(
		"<IHHIIHHf",
		fields["type_of_file"],
		fields["root_note"],
		fields["unknown1"],
		fields["unknown2"],
		fields["num_beats"],
		fields["meter_denominator"],
		fields["meter_numerator"],
		fields["tempo"],
	)

	body += fields.get("extra", b"")

	if len(body) % 2:
		body += b"\x00"

	return b"acid" + struct.pack("<I", len(body)) + body

def _build_chunk_adtl(markers: List[Entry]) -> bytes:
	"""
	Builds a complete LIST/adtl chunk from a list of markers.

	Emits:
	- labl sub-chunk for every entry's name
	- note sub-chunk for any entry with a comment,
	- ltxt sub-chunk for non-loop entries
		- only ones that require one (regions, and point markers with a non-default type)
	- tlst sub-chunk for any entry with a root_note set.
 
	Parameters
	----------
		markers (List[Entry]):
			All Entries belonging to the file.
 
	Returns
	-------
		bytes:
			The fully encoded "LIST" chunk with "adtl" type,
			including its 8-byte header and even-padded body.
	"""
 
	adtl = b"adtl"

	for m in markers:
		# labl
		text      = m.name.encode("latin-1") + b"\x00"
		labl_body = struct.pack("<I", m.id) + text

		if len(labl_body) % 2:
			labl_body += b"\x00"

		adtl += b"labl" + struct.pack("<I", len(labl_body)) + labl_body

		# commment (only if set)
		if m.comment:
			info_text = m.comment.encode("latin-1") + b"\x00"
			info_body = struct.pack("<I", m.id) + info_text

			if len(info_body) % 2:
				info_body += b"\x00"

			adtl += b"note" + struct.pack("<I", len(info_body)) + info_body

		# ltxt
		if not m.is_loop and m.needs_ltxt():
			ltxt_body = struct.pack("<II4sHHHH",
				m.id, m.length, MarkerType.encode(m.type), 0, 0, 0, 0)

			if len(ltxt_body) % 2:
				ltxt_body += b"\x00"

			adtl += b"ltxt" + struct.pack("<I", len(ltxt_body)) + ltxt_body

		# tlst (only if note is set, only for non-loop regions)
		if m.note is not None and not m.is_loop:
			# cue_id
			tlst_body = struct.pack("<I", m.id)

			# chunk type reference
			tlst_body += b"cue "

			# unknown_1, unknown_2
			tlst_body += struct.pack("<II", 0, 1)
			
			# encoded note, constant
			tlst_body += bytes([m.note - 12, 0xFF])

			# constant + padding
			tlst_body += struct.pack("<HII", 0x9000, 0, 0)

			if len(tlst_body) % 2:
				tlst_body += b"\x00"

			adtl += b"tlst" + struct.pack("<I", len(tlst_body)) + tlst_body

	if len(adtl) % 2:
		adtl += b"\x00"

	return b"LIST" + struct.pack("<I", len(adtl)) + adtl

def _build_chunk_smpl(loops: List[Entry], raw_header: bytes) -> bytes:
	"""
	Builds a complete smpl chunk from a list of loop entries.
 
	Preserves the original smpl header fields
	(manufacturer, product, sample period, MIDI note)
	when available, updating only the loop count and loop records.
 
	Parameters
	----------
		loops (List[Entry]):
			Loop entries to encode as smpl loop records.
 
		raw_header (bytes):
			The original 36-byte smpl header, as returned by
			_parse_chunk_smpl, used to preserve non-loop metadata.

			If shorter than 36 bytes, a minimal default header is
			generated instead.
 
	Returns
	-------
		bytes:
			The fully encoded "smpl" chunk, including its 8-byte header and even-padded body.
	"""
	if len(raw_header) >= 36:
		# patch num_loops in place, keep everything else
		header = bytearray(raw_header[:36])
		struct.pack_into("<I", header, 28, len(loops))
		header = bytes(header)
	else:
		# no original smpl header - build a minimal one
		# (sample_period 0 is safe; DAW will use fmt chunk for actual rate)
		header = struct.pack(
			"<IIIIiIIII",
			0,          # manufacturer
			0,          # product
			0,          # sample_period
			60,         # midi_unity_note (middle C)
			0,          # midi_pitch_fraction
			0,          # smpte_format
			0,          # smpte_offset
			len(loops), # num_sample_loops
			0,          # sampler_data
		)

	body = header
	for loop in loops:
		body += struct.pack(
			"<IIIIII",
			loop.id,
			loop.loop_type,
			loop.start,
			loop.end - 1, # back to inclusive (certain software convention)
			0,            # fraction
			0,            # play_count  (0 = infinite)
		)

	if len(body) % 2:
		body += b"\x00"

	return b"smpl" + struct.pack("<I", len(body)) + body

#-=-=-=-#

def _build_file(
	raw_file:        bytes,
	markers:         List[Entry],
	smpl_raw_header: bytes = b"",
	acid:            Optional[dict]            = None,
	synced:          Optional[bool]            = None,
	note_root:       Optional[Union[int, str]] = None,
	time_signature:  Optional[str]             = None,
	tempo:           Optional[float]           = None,
	num_beats:       Optional[int]             = None,
) -> bytes:
	"""
	Rebuilds a complete WAV file with updated marker, region, and loop data.
 
	All existing cue, LIST/adtl, and smpl chunks are stripped from
	the source file and replaced with freshly serialized versions
	reflecting the current marker state.

	All other chunks (fmt, data, and any unrelated metadata) are preserved unchanged.
 
	Parameters
	----------
		raw_file (bytes):
			The original WAV file bytes to use as a base.
 
		markers (List[Entry]):
			The current set of Entries to write.
 
		smpl_raw_header (bytes):
			The original smpl chunk header, used to preserve sample
			period and MIDI metadata when loops are present or when
			the source file already had a smpl chunk. Defaults to
			empty bytes if no smpl chunk existed.
 
		acid (Optional[dict]):
			Parsed acid chunk fields from _parse_chunk_acid, used to
			preserve tempo/root-note/meter metadata on rewrite. If None,
			no acid chunk is written even if the source file had one.

		synced (Optional[bool]):
			Overrides the acid chunk's tempo-mode bit on write (True =
			Synced, False = Free). None preserves whatever `acid`
			already contains.

			Has no effect if `acid` is None.
 
	Returns
	-------
		bytes:
			A complete, valid WAV file as raw bytes, ready to be
			written to disk or returned to the caller.
	"""
	f = io.BytesIO(raw_file)
	f.read(12) # RIFF????WAVE
 
	kept = bytearray()
 
	while True:
		try:
			fourcc, size = _read_chunk_header(f)
		except EOFError:
			break
 
		body = f.read(_pad(size))
 
		# strip old marker/loop chunks - all rewritten below
		if fourcc == b"cue ":
			continue
		if fourcc == b"LIST" and body[:4] == b"adtl":
			continue
		if fourcc == b"smpl":
			continue
 
		kept += fourcc + struct.pack("<I", size) + body
 
	loops       = [m for m in markers if m.is_loop]
	cue_markers = [m for m in markers if not m.is_loop]
 
	# smpl - write if there are loops OR if the file originally had a smpl chunk
	# (preserve it so DAW metadata like sample period / MIDI note isn't lost)
	if loops or smpl_raw_header:
		kept += _build_chunk_smpl(loops, smpl_raw_header)
 
	# cue + adtl - write if there are any non-loop entries
	if cue_markers:
		kept += _build_chunk_cue(cue_markers)

		# adtl covers ALL entries (loops get labl too)
		kept += _build_chunk_adtl(markers)
 
	elif markers:
		# only loops - still need adtl for their labl names
		kept += _build_chunk_adtl(markers)
 
	acid_chunk = _build_chunk_acid(
		acid,
		synced         = synced,
		note_root      = note_root,
		time_signature = time_signature,
		tempo          = tempo,
		num_beats      = num_beats,
	)

	if acid_chunk:
		kept += acid_chunk

	return b"RIFF" + struct.pack("<I", 4 + len(kept)) + b"WAVE" + bytes(kept)