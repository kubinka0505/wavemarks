from .models.region import Region, Position

import struct
from typing import List, Optional, Union

#-=-=-=-#

# Region

def _types_normalize(types: Union[str, List[str], None]) -> Optional[set]:
	"""
	Normalises a marker type filter argument into a set for membership testing.

	Parameters
	----------
		types (str | List[str] | None):
			A single marker type string, a list of marker type strings,
			or None to indicate no filtering should occur.

	Returns
	-------
		Optional[set]:
			A set of marker type strings, or None if types was None
			(meaning the caller should treat the filter as "match any").
	"""
	if types is None:
		return None

	if isinstance(types, str):
		return {types}

	return set(types)

def _position_coerce(pos: Position) -> Union[int, Region]:
	"""
	Normalises any accepted position input into either a sample index or a Region.

	Parameters
	----------
		pos (Position):
			One of: an int (point position), a Region, a `range`,
			or a 2-tuple/list of (start, end) sample indices.
 
	Returns
	-------
		Union[int, Region]:
			The position as a plain int if it represents a single sample,
			or as a Region if it represents a start/end span.
 
	Raises
	------
		TypeError:
			If pos is not one of the accepted input types.
	"""
	if isinstance(pos, int):
		return pos

	if isinstance(pos, Region):
		return pos

	if isinstance(pos, range):
		return Region(pos.start, pos.stop)

	if isinstance(pos, (tuple, list)) and len(pos) == 2:
		return Region(int(pos[0]), int(pos[1]))

	raise TypeError(
		f"Position must be an int, Region, range, or 2-tuple; got {type(pos).__name__!r}"
	)

# Low-level RIFF helpers

def _read_chunk_header(f):
	"""
	Reads an 8-byte RIFF chunk header from a binary stream.
 
	Parameters
	----------
		f (BinaryIO):
			A file-like object positioned at the start of a chunk header.
 
	Returns
	-------
		Tuple[bytes, int]:
			The 4-byte chunk FourCC identifier and the chunk's declared
			size in bytes (not including the 8-byte header itself).
 
	Raises
	------
		EOFError:
			If fewer than 8 bytes remain in the stream.
	"""
	data = f.read(8)

	if len(data) < 8:
		raise EOFError

	return data[:4], struct.unpack_from("<I", data, 4)[0]


def _pad(size: int) -> int:
	"""
	Computes the padded length of a RIFF chunk body.
 
	RIFF chunks are padded to an even number of bytes; chunks with an
	odd declared size have one trailing padding byte on disk.
 
	Parameters
	----------
		size (int):
			The chunk's declared (unpadded) size in bytes.
 
	Returns
	-------
		int:
			size if already even, otherwise size + 1.
	"""
	return size + (size % 2)