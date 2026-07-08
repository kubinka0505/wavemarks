from ..helpers import _read_chunk_header, _pad

import io
import struct
from typing import List, Tuple

#-=-=-=-#
# Main

def _parse_chunk_smpl(data: bytes) -> Tuple[List[dict], bytes]:
	"""
	Parses the body of a smpl chunk into loop records and a preserved header.

	Parameters
	----------
		data (bytes):
			The raw smpl chunk body, excluding the 8-byte chunk header.

	Returns
	-------
		Tuple[List[dict], bytes]:
			A list of loop dicts, each containing:
			- id
			- loop_type,
			- end (is converted from the on-disk inclusive value to an exclusive sample index)
			- the original 36-byte smpl header:
				- manufacturer
				- product
				- sample period
				- MIDI fields

			Preserved verbatim for round-trip rewriting.

		([], b""):
			If data is shorter than the 36-byte header.
	"""
	if len(data) < 36:
		return [], b""

	raw_header = data[:36]
	num_loops  = struct.unpack_from("<I", data, 28)[0]

	loops  = []
	offset = 36 # smpl header is 36 bytes

	for _ in range(num_loops):
		if offset + 24 > len(data):
			break

		cue_id, type_, start, end, _, _ = \
			struct.unpack_from("<IIIIII", data, offset)

		loops.append(dict(
			cue_id    = cue_id,
			loop_type = type_,
			start     = start,
			end       = end + 1, # some software stores inclusive end → convert to exclusive
		))

		offset += 24

	return loops, raw_header

def _parse_chunk_acid(data: bytes) -> dict:
	"""
	Parses the body of an acid chunk.

	The standard ACID chunk body is 24 bytes:
		type_of_file      (uint32) - bitfield; bit 0 distinguishes tempo mode
	                      (0 = Synced, 1 = Free) as observed on
	                      round-tripped files. Other bits are
	                      preserved verbatim, not interpreted.
	    root_note         (uint16)
	    unknown1          (uint16) - reserved, preserved verbatim
	    unknown2          (uint32) - reserved, preserved verbatim
	    num_beats         (uint32)
	    meter_denominator (uint16)
	    meter_numerator   (uint16)
	    tempo             (float32)

	Parameters
	----------
		data (bytes):
			The raw acid chunk body, excluding the 8-byte chunk header.

	Returns
	-------
		dict:
			Parsed fields:
			- type_of_file
			- root_note
			- num_beats
			- meter_denominator
			- meter_numerator
			- tempo
			- synced

			...plus "unknown1", "unknown2" and "extra"
			(any trailing bytes beyond the standard 24,
			preserved for round-trip fidelity).

		{}:
			If data is shorter than the 24-byte body.
	"""
	if len(data) < 24:
		return {}

	(
		type_of_file, root_note, unknown1, unknown2,
		num_beats, meter_denominator, meter_numerator, tempo
	) = \
		struct.unpack_from("<IHHIIHHf", data, 0)

	return dict(
		type_of_file      = type_of_file,
		root_note         = root_note,
		unknown1          = unknown1,
		unknown2          = unknown2,
		num_beats         = num_beats,
		meter_denominator = meter_denominator,
		meter_numerator   = meter_numerator,
		tempo             = tempo,
		synced            = (type_of_file & 0x01) == 0, # bit 0 clear = Synced
		extra             = data[24:], # any bytes beyond the standard body, preserved as-is
	)

def _parse_chunk_cue(data: bytes) -> List[dict]:
	"""
	Parses the body of a cue chunk into a list of cue-point dicts.

	Parameters
	----------
		data (bytes):
			The raw cue chunk body, excluding the 8-byte chunk header.
 
	Returns
	-------
		List[dict]:
			One dict per cue point, each containing id (the cue point
			identifier) and sample_offset (the cue's sample position).
		
		[]:
			If data is too short to contain a valid cue count.
	"""
	if len(data) < 4:
		return []

	num    = struct.unpack_from("<I", data, 0)[0]
	cues   = []
	offset = 4

	for _ in range(num):
		if offset + 24 > len(data):
			break

		id_, _, _, _, _, sample_offset = \
			struct.unpack_from("<II4sIII", data, offset)

		cues.append(dict(
			id            = id_,
			sample_offset = sample_offset,
		))

		offset += 24

	return cues

def _parse_chunk_adtl(data: bytes):
	"""
	Parses the body of a LIST/adtl chunk into its constituent sub-chunks.
 
	Parameters
	----------
		data (bytes):
			The raw LIST chunk body, including the leading "adtl" type identifier.
 
	Returns
	-------
		Tuple[dict, dict, dict, dict]:
			labels     – {cue_id: label_text} from labl sub-chunks.
			comments   – {cue_id: comment_text} from note sub-chunks.
			ltxt_info  – {cue_id: (purpose, length)} from ltxt sub-chunks,
					where purpose is a 4-character marker type code
					and length is the region's sample length.
			root_notes – {cue_id: midi_note} from tlst sub-chunks.

			All four dicts are empty if data does not begin with "adtl".
	"""
	labels:    dict[int, str]   = {}
	comments:  dict[int, str]   = {}
	ltxt_info: dict[int, tuple] = {}
	notes:     dict[int, int]   = {}

	if len(data) < 4 or data[:4] != b"adtl":
		return labels, comments, ltxt_info, notes

	offset = 4
	while offset + 8 <= len(data):
		sub_id   = data[offset:offset + 4]
		sub_size = struct.unpack_from("<I", data, offset + 4)[0]
		sub_body = data[offset + 8: offset + 8 + sub_size]
		offset  += 8 + _pad(sub_size)

		if sub_id == b"labl" and len(sub_body) >= 4:
			cue_id         = struct.unpack_from("<I", sub_body, 0)[0]
			labels[cue_id] = sub_body[4:].rstrip(b"\x00").decode("latin-1")

		elif sub_id == b"note" and len(sub_body) >= 4:
			cue_id           = struct.unpack_from("<I", sub_body, 0)[0]
			comments[cue_id] = sub_body[4:].rstrip(b"\x00").decode("latin-1")

		elif sub_id == b"ltxt" and len(sub_body) >= 12:
			cue_id            = struct.unpack_from("<I", sub_body, 0)[0]
			length            = struct.unpack_from("<I", sub_body, 4)[0]
			purpose           = sub_body[8:12].decode("latin-1")
			ltxt_info[cue_id] = (purpose, length)

		elif sub_id == b"tlst" and len(sub_body) >= 18:
			cue_id        = struct.unpack_from("<I", sub_body, 0)[0]
			encoded       = sub_body[16] # midi_note - 12
			notes[cue_id] = encoded + 12 # back to standard MIDI

	return labels, comments, ltxt_info, notes

#-=-=-=-#

def _parse_file_bytes(raw_file: bytes):
	"""
	Parses raw WAV file bytes into marker, region, and loop metadata.
 
	Parameters
	----------
		raw_file (bytes):
			The complete contents of a WAV file, including the RIFF
			header and all chunks.
 
	Returns
	-------
		Tuple:
			raw_cues        – list of cue-point dicts from the cue chunk.
			labels          – {id: label_text} from adtl/labl.
			comments        – {id: comment_text} from adtl/note.
			ltxt_info       – {id: (purpose, length)} from adtl/ltxt.
			root_notes      – {id: midi_note} from adtl/tlst.
			smpl_loops      – list of loop dicts from the smpl chunk.
			smpl_raw_header – raw 36-byte smpl header, preserved for
						 round-trip rewriting.
			acid            – dict of parsed acid chunk fields (see
						 _parse_chunk_acid), or {} if no acid
						 chunk was present.
			raw_file        – the original input bytes, returned
						 unchanged for use in serialization.
 
	Raises
	------
		ValueError:
			If raw_file does not begin with a valid RIFF/WAVE header.
	"""
	f      = io.BytesIO(raw_file)
	header = f.read(12)

	if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
		raise ValueError("Not a valid WAV file: missing RIFF/WAVE header")

	raw_cues        = []
	labels          = {}
	comments        = {}
	ltxt_info       = {}
	notes           = {}
	smpl_loops      = []
	smpl_raw_header = b""
	acid            = {}

	while True:
		try:
			fourcc, size = _read_chunk_header(f)
		except EOFError:
			break

		body = f.read(_pad(size))

		if fourcc == b"cue ":
			raw_cues = _parse_chunk_cue(body[:size])
		elif fourcc == b"LIST" and body[:4] == b"adtl":
			labels, comments, ltxt_info, notes = _parse_chunk_adtl(body[:size])
		elif fourcc == b"smpl":
			smpl_loops, smpl_raw_header = _parse_chunk_smpl(body[:size])
		elif fourcc == b"acid":
			acid = _parse_chunk_acid(body[:size])

	return raw_cues, labels, comments, ltxt_info, notes, smpl_loops, smpl_raw_header, acid, raw_file