from typing import Optional

#-=-=-=-#

class TempoMode:
	"""
	First byte of the ACID chunk's 4-byte type-of-file field.

	These values (0x04 / 0x05) happen to coincide with the ASCII control
	codes EOT / ENQ — used here as memorable names, not because they carry
	any control-character meaning in this context.
	"""
	SYNCED = 0x04 # EOT
	FREE   = 0x05 # ENQ

def _find_acid_chunk_data_offset(raw: bytes) -> int:
	"""
	Locate the start of the `acid` chunk's data within a raw WAV byte
	buffer and return the file offset of its first data byte (i.e. the
	tempo-mode flag byte).

	Raises ValueError if no `acid` chunk is present — callers should treat
	this as "file has no ACID tempo metadata to modify."
	"""
	idx = raw.find(b"acid")

	if idx == -1:
		raise ValueError("No `acid` chunk found in file; cannot set tempo mode.")

	# layout: b'acid' (4 bytes) + chunk size (4 bytes, LE uint32) + data...
	data_offset = idx + 8

	if data_offset >= len(raw):
		raise ValueError("Malformed `acid` chunk: no data bytes after header.")

	return data_offset

def set_acid_tempo_flag(raw: bytes, synced: bool) -> bytes:
	"""
	Return a copy of `raw` with the ACID chunk's tempo-mode flag byte set.

	Parameters
	----------
		raw (bytes):
			Full raw WAV file bytes.

		synced (bool):
			True  -> TempoMode.SYNCED (0x04 / EOT)
			False -> TempoMode.FREE   (0x05 / ENQ)

	Returns
	-------
		bytes:
			Modified copy of the file bytes.
	"""
	offset = _find_acid_chunk_data_offset(raw)

	buf = bytearray(raw)
	buf[offset] = TempoMode.SYNCED if synced else TempoMode.FREE

	return bytes(buf)

def get_acid_tempo_flag(raw: bytes) -> Optional[bool]:
	"""
	Read the current tempo-mode flag from a raw WAV byte buffer.

	Returns
	-------
		True if Synced, False if Free, None if no `acid` chunk
		is present or the byte doesn't match either known value.
	"""
	try:
		offset = _find_acid_chunk_data_offset(raw)
	except ValueError:
		return None

	val = raw[offset]

	if val == TempoMode.SYNCED:
		return True
	elif val == TempoMode.FREE:
		return False

	return None # unrecognized value; don't guess