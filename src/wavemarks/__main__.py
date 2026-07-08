from __future__ import annotations

import math
import struct
import warnings
from io import BytesIO

from ._core.exceptions import (
	OutOfBoundsError,
	InvalidRangeError,
	DuplicateBoundsWarning,
	FieldMismatchWarning
)

from ._core.helpers import (
	_types_normalize,
	_position_coerce,
	_read_chunk_header,
	_pad
)

from ._core.models.type import MarkerType
from ._core.models.entry import Entry
from ._core.models.region import Region, Position
from ._core.utils.tempo import detect as detect_tempo

from ._core.operations.parse import _parse_file_bytes
from ._core.operations.serialize import (
	_build_file,
	_apply_acid_overrides,
	_DEFAULT_ACID
)

from typing import List, Tuple, Optional, Union, Iterator

#-=-=-=-#

class MarkerFile:
	"""
	Fluent interface for markers and regions.

	Marker: add_entry(41501, "hit", MarkerType.DOWNBEAT)
            add_entry(Entry(start=41501, name="hit", type=MarkerType.DOWNBEAT))
	Region: add_entry((200, 400), "verse", MarkerType.BASIC)
            add_entry(range(200, 400), "verse", MarkerType.BASIC)
            add_entry(Entry(start=200, end=400, name="verse"))

	Markers and regions may freely overlap.
	All mutating methods return a new class instance; call .save() to write to disk.

	Note: += and -= rebind the variable, not mutate in-place.
          Inside a `with` block, always use: `f = f.add_entry(...)` not `f.add_entry(...)`
	"""

	def __init__(self, source: Union[str, bytes, BytesIO]) -> None:
		"""
		Open a file for marker/region/loop manipulation.

		source: str     - path to a file on disk
	            bytes   - raw file bytes
	            BytesIO - in-memory buffer

		When source is bytes or BytesIO, self.path is set to None
		and save() will require an explicit path argument.
		"""
		if isinstance(source, str):
			self.path     = source
			raw_file_data = open(source, "rb").read()
		elif isinstance(source, (bytes, bytearray)):
			self.path     = None
			raw_file_data = bytes(source)
		elif isinstance(source, BytesIO):
			self.path     = None
			raw_file_data = source.getvalue()
		else:
			raise TypeError(
				f"MarkerFile expects a path (str), bytes, or BytesIO - got {type(source).__name__}"
			)

		self._raw_file = raw_file_data

		(raw_cues, labels, comments, ltxt_info, notes,
		smpl_loops, smpl_raw_header, acid, raw_file) = _parse_file_bytes(raw_file_data)

		self._smpl_raw_header = smpl_raw_header
		self._n_samples       = MarkerFile._get_sample_count(self._raw_file)
		self._sample_rate     = MarkerFile._get_sample_rate(self._raw_file)
		self._acid            = acid
		self._cue_fields: List[Entry] = []

		# regular markers/regions from cue chunk
		for c in raw_cues:
			cid = c["id"]

			if cid in ltxt_info:
				purpose, length = ltxt_info[cid]
				end = c["sample_offset"] + length if length else None
			else:
				purpose, end = MarkerType.BASIC, None

			self._cue_fields.append(
				Entry(
					id      = cid,
					name    = labels.get(cid, ""),
					start   = c["sample_offset"],
					end     = end,
					type    = purpose,
					comment = notes.get(cid) or None,
					note    = notes.get(cid),
				)
			)

		# loops from smpl chunk
		used_ids    = {m.id for m in self._cue_fields}
		loop_labels = [labels[cid] for cid in sorted(labels) if cid not in used_ids]

		for i, loop in enumerate(smpl_loops):
			self._cue_fields.append(
					Entry(
					id        = loop["cue_id"],
					name      = loop_labels[i] if i < len(loop_labels) else "",
					start     = loop["start"],
					end       = loop["end"],
					type      = MarkerType.BASIC,
					loop_type = loop["loop_type"],
				)
			)

		self._cue_fields = sorted(self._cue_fields, key = lambda m: m.start)

	@staticmethod
	def _get_sample_count(raw_file: bytes) -> Optional[int]:
		"""
		Computes the total number of audio samples in a file.

		Reads the fmt chunk to obtain block_align (bytes per sample frame)
		and the data chunk to obtain the audio payload size, then divides
		to produce a sample count.

		Used to validate marker positions in their adding function against the actual audio length.

		Parameters
		----------
			raw_file (bytes):
				The complete file bytes, including RIFF header and all chunks.

		Returns
		-------
			Optional[int]:
				Total number of sample frames, or None
				if the fmt or data chunk could not be found
				(in which case bounds validation is silently skipped by the caller).
		"""
		f = BytesIO(raw_file)
		f.read(12)

		block_align  = None
		data_size    = None

		while True:
			try:
				fourcc, size = _read_chunk_header(f)
			except EOFError:
				break

			body = f.read(_pad(size))

			if fourcc == b"fmt ":
				if len(body) >= 14:
					block_align = struct.unpack_from("<H", body, 12)[0]

			elif fourcc == b"data":
				data_size = size

		if block_align and data_size:
			return data_size // block_align

		return None

	@staticmethod
	def _get_sample_rate(raw_file: bytes) -> Optional[int]:
		"""
		Reads nSamplesPerSec from the fmt chunk.

		Returns
		-------
			integer:
				Found sample rate value from the chunk.

			None:
				If not found.
		"""
		f = BytesIO(raw_file)
		f.read(12)

		while True:
			try:
				fourcc, size = _read_chunk_header(f)
			except EOFError:
				break

			body = f.read(_pad(size))

			if fourcc == b"fmt " and len(body) >= 8:
				return struct.unpack_from("<I", body, 4)[0] # nSamplesPerSec

		return None

	#-=-=-=-#
	# Context manager

	def __enter__(self):
		"""
		Enters the context manager, returning self.

		Mutations inside the with block must still be assigned explicitly
		(f = f.add_entry(...)) since all mutating methods return a new instance
		rather than modifying in place.

		The context manager provides a clear scope and exception-safety guarantee.
		The file on disk is never touched unless .save() is called explicitly.

		Returns
		-------
			MarkerFile:
				This instance, unchanged.
		"""
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> None:
		"""
		Exits the context manager without performing any automatic action.

		No auto-save is performed on exit.

		Callers must call .save() explicitly before the with block ends
		if they wish to write changes to disk.

		Parameters
		----------
			exc_type:
				The exception class, or None if no exception was raised.

			exc_val:
				The exception instance, or None.

			exc_tb:
				The traceback, or None.
		"""
		pass

	#-=-=-=-#
	# Operators

	def __iadd__(self, entry: Entry):
		"""
		Adds a Entry via the += operator.

		Syntactic sugar for f = f.add_entry(entry).

		Because MarkerFile is immutable/fluent, += rebinds
		the variable to a new instance rather than mutating the existing one.

		Parameters
		----------
			entry (Entry):
				The Entry to add.

		Returns
		-------
			MarkerFile:
				A new MarkerFile instance containing the added entry.

		Raises
		------
			ValueError:
				If an identical entry already exists.

			InvalidRangeError:
				If the entry's end is not greater than its start.

			OutOfBoundsError:
				If the entry's start or end exceeds the file's sample count.
		"""
		return self.add_entry(entry)

	def __isub__(self, entry: Entry):
		"""
		Removes a Entry via the -= operator.

		Syntactic sugar for f = f.delete_entry(entry).

		Rebinds the variable to a new instance with the entry removed.

		Parameters
		----------
			entry (Entry):
				The Entry to remove.
				Matched by equality (name, start, type, end, loop_type), not by id.

		Returns
		-------
			MarkerFile:
				A new MarkerFile instance with the entry removed.

		Raises
		------
			KeyError:
				If no matching entry is found.
		"""
		return self.delete_entry(entry)

	def __contains__(self, entry: Entry) -> bool:
		"""
		Tests membership via the in operator.

		Checks whether a Entry equal to entry exists in the current entries list.

		Equality is determined by Entry.__eq__, which compares
		name, start, type, end, and loop_type (not id).

		Parameters
		----------
			entry (Entry):
				The entry to search for.

		Returns
		-------
			bool:
				True if a matching entry is found, False otherwise.
		"""
		return any(m == entry for m in self._cue_fields)

	def __iter__(self) -> Iterator[Entry]:
		"""
		Iterates over all entries sorted by start sample position.

		Enables: for m in f, list(f), and any other iteration context.

		The iteration order reflects the internal sorted _cue_fields list,
		which covers markers, regions, and loops together.

		Returns
		-------
			Iterator[Entry]:
				An iterator over all Entry entries in start-sample order.
		"""
		return iter(self._cue_fields)

	def __len__(self) -> int:
		"""
		Returns the total number of Entries in the file.

		Returns
		-------
			int:
				The number of Entry held in this class object.
		"""
		return len(self._cue_fields)

	def __repr__(self) -> str:
		"""
		Returns a human-readable summary of the MarkerFile and its entries.

		Lists all entries sorted by start sample, annotating each with
		its position (point, start-end range, or loop range), type FourCC,
		comment, root_note, and name.

		Loop entries are prefixed with "(loop)".

		Returns
		-------
			str:
				A multi-line string beginning with a header line of the
				form "MarkerFile(<path>, <count> entries)" followed by
				one indented line per entry.
		"""
		lines = [f"MarkerFile({self.path!r}, {len(self._cue_fields)} entries)"]

		for m in self:
			if m.is_loop:
				pos = f"{m.start}~{m.end}"
			elif m.is_region:
				pos = f"{m.start}-{m.end}"
			else:
				pos = str(m.start)

			extra = ""
			if m.is_loop:
				extra = "(loop) "
			if m.comment:
				extra += f"comment={m.comment!r} "
			if m.note is not None:
				extra += f"note={m.note} "

			lines.append(f"  [{pos:>22}]  type={m.type!r}  {extra}{m.name!r}")

		return "\n".join(lines)

	#-=-=-=-#
	# Internal

	def _copy(self, entries: List[Entry]):
		"""
		Creates a new MarkerFile instance sharing the same raw file data.

		Used internally by all mutating methods to implement the fluent/immutable pattern.

		The new instance shares _raw_file, _smpl_raw_header, and _acid by
		reference (safe since bytes/dicts here are treated as read-only
		snapshots at copy time), and receives the supplied marker list as
		its _cue_fields.

		Parameters
		----------
			entries (List[Entry]):
				The new entry list for the returned instance.

		Returns
		-------
			MarkerFile:
				A new MarkerFile with the same path and raw file data,
				but with _cue_fields replaced by entries.
		"""
		obj                  = object.__new__(MarkerFile)
		obj.path             = self.path
		obj._raw_file        = self._raw_file
		obj._smpl_raw_header = self._smpl_raw_header
		obj._n_samples       = self._n_samples
		obj._sample_rate     = self._sample_rate
		obj._acid            = self._acid
		obj._cue_fields      = sorted(entries, key = lambda m: m.start)

		return obj

	def _next_id(self) -> int:
		"""
		Returns the next available cue point ID.

		Scans the current entry list for all used ID values and
		returns the smallest positive integer not already in use.

		IDs start at 1, matching the RIFF cue chunk convention.

		Returns
		----------
			int:
				The smallest integer >= 1 not currently assigned to
				any entry in _cue_fields.
		"""
		used = {m.id for m in self._cue_fields}
		i = 1

		while i in used:
			i += 1

		return i

	def _find(
		self,
		pos:   Union[int, Region],
		name:  str,
		type_: str,
	) -> Optional[Entry]:
		"""
		Searches for a Entry by position, name, and type.

		Iterates over _cue_fields and returns the first entry whose
		matches() method returns True for the given arguments.

		Used internally by add_entry(), delete_entry(), and rename_entry() to
		locate existing entries before performing mutations.

		Parameters
		----------
			pos (Union[int, Region]):
				The position to match.

				int matches point markers by start sample.
				Region matches ranged entries by both start and end.

			name (str):
				The label to match.

			type_ (str):
				The FourCC marker type string to match.

		Returns
		-------
			Optional[Entry]:
				The first matching entry, or None if no match is found.
		"""
		for m in self._cue_fields:
			if m._matches(pos, name, type_):
				return m

		return None

	#-=-=-=-#
	# Modifiers

	def add_entry(
		self,
		position:  Union[Position, Entry],

		name:      str           = "",
		type:      str           = MarkerType.BASIC,

		loop_type: Optional[int] = None,

		comment:   Optional[str] = None,
		note:      Optional[int] = None,
	):
		"""
		Adds a point marker, region, or loop to the file.

		Accepts either a Entry directly or a raw position with
		accompanying keyword arguments.

		When a Entry is passed, all its fields are carried over
		and the remaining keyword arguments are ignored.

		Parameters
		----------
			position (Union[Position, Entry]):
				The location to add. Accepts:
					int               - point marker at that sample.
					(start, end)      - region or loop spanning those samples.
					range(start, end) - same as a 2-tuple.
					Entry             - all fields taken from the object.

			name (str):
				Label text for the entry. Defaults to empty string.

			type (str):
				FourCC marker type. Defaults to MarkerType.BASIC.

			loop_type (Optional[int]):
				If set, the entry is stored as a loop in
				the smpl chunk with this LoopType loop type.

				Mutually exclusive with comment and note.

			comment (Optional[str]):
				Free-text annotation stored in the adtl/note sub-chunk.

				Only applies to regions and point markers, not loops.

			note (Optional[int]):
				MIDI root note stored in the adtl/tlst sub-chunk, in the range 12 (C0) to 127 (G10).

				Use Note.encode() to convert from a note name.
				Only applies to regions, not loops.

		Returns
		-------
			MarkerFile:
				A new MarkerFile instance containing the added entry.

		Raises
		------
			ValueError:
				If an identical entry (same position, name, and type) already exists.

			InvalidRangeError:
				If end is not strictly greater than start for a ranged entry.

			OutOfBoundsError:
				If start or end exceeds the file's total sample count.

		Warns:
			FieldMismatchWarning:
				If loop_type is set alongside comment or note,
				since those fields are not written for loop entries.

			DuplicateBoundsWarning:
				If a region with the same bounds already exists when
				adding a loop, or vice versa.
		"""
		if isinstance(position, Entry):
			m         = position
			loop_type = m.loop_type
			comment   = m.comment
			note      = m.note
			pos       = _position_coerce((m.start, m.end) if m.is_region or m.is_loop else m.start)
			name      = m.name
			type      = m.type
		else:
			pos = _position_coerce(position)

		is_ranged = isinstance(pos, Region)
		start     = pos.start if is_ranged else pos
		end       = pos.end   if is_ranged else None

		# --- validation: range sanity ---
		if is_ranged and end <= start:
			raise InvalidRangeError(
				f"Region/loop end ({end}) must be greater than start ({start})"
			)

		# --- validation: bounds ---
		if self._n_samples is not None:
			if start < 0 or start > self._n_samples:
				raise OutOfBoundsError(
					f"start={start} exceeds file length ({self._n_samples} samples)"
				)

			if end is not None and end > self._n_samples:
				raise OutOfBoundsError(
					f"end={end} exceeds file length ({self._n_samples} samples)"
				)

		# --- validation: exclusive field mismatch ---
		is_loop_entry = loop_type is not None

		if is_loop_entry and (comment is not None or note is not None):
			warnings.warn(
				f"loop_type is set ({loop_type!r}) but note/pitch were also "
				f"provided - note and pitch only apply to regions, not loops, "
				f"and will be ignored on write",
				FieldMismatchWarning,
			)

		# --- validation: duplicate bounds across kinds ---
		if is_ranged and self._n_samples is not None:
			same_bounds = [
				m for m in self._cue_fields
				if m.start == start and m.end == end
			]

			for m in same_bounds:
				if is_loop_entry and m.is_region:
					warnings.warn(
						f"There's already a region with the same bounds "
						f"({start}–{end}): {m.name!r}",
						DuplicateBoundsWarning,
					)

				elif not is_loop_entry and m.is_loop:
					warnings.warn(
						f"There's already a loop with the same bounds "
						f"({start}–{end}): {m.name!r}",
						DuplicateBoundsWarning,
					)

		# --- existing duplicate-entry check ---
		if self._find(pos, name, type) is not None:
			raise ValueError(f"Already exists: pos={pos}, name={name!r}, type={type!r}")

		new_m = Entry(
			id        = self._next_id(),
			name      = name,
			start     = start,
			end       = end,
			type      = type,
			loop_type = loop_type,
			comment   = comment,
			note      = note,
		)

		return self._copy(self._cue_fields + [new_m])

	def delete_entry(
		self,
		position: Union[Position, Entry],

		name: str = "",
		type: str = MarkerType.BASIC,
	):
		"""
		Removes a marker, region, or loop from the file.

		Parameters
		----------
			position (Union[Position, Entry]):
				The entry to remove.

				Accepts a Entry (matched by equality)
				or a raw position with name and type used to locate the entry via _find().

			name (str):
				Label to match.

				Only used when position is not a Entry.
				Defaults to empty string.

			type (str):
				FourCC type to match.

				Only used when position is not a
				Entry. Defaults to MarkerType.BASIC.

		Returns
		-------
			MarkerFile:
				A new MarkerFile instance with the matched entry removed.

		Raises
		------
			KeyError:
				If no matching entry is found.
		"""
		if isinstance(position, Entry):
			target = next((m for m in self._cue_fields if m == position), None)
		else:
			pos    = _position_coerce(position)
			target = self._find(pos, name, type)

		if target is None:
			raise KeyError(f"Not found: {position!r}")

		return self._copy([m for m in self._cue_fields if m is not target])

	def rename_entry(
		self,
		target:   Union[str, Entry],
		new_name: str,
	):
		"""
		Renames a marker, region, or loop.

		Locates the target entry and returns a new MarkerFile with that entry's name replaced.
		All other fields (type, start, end, loop_type, comment, note) are preserved unchanged.

		Parameters
		----------
			target (Union[str, Entry]):
				The Entry to rename. Accepts:
				- Entry: matched by equality (name, start, type, end, loop_type).
				- str:   matched by name; the first entry whose name equals this string is renamed.

			new_name (str):
				The replacement label text.

		Returns
		-------
			MarkerFile:
				A new MarkerFile instance with the matched entry renamed.

		Raises
		------
			KeyError:
				If no matching Entry is found.
		"""
		if isinstance(target, Entry):
			pred = lambda m: m == target
		else:
			pred = lambda m: m.name == target

		found = next((m for m in self._cue_fields if pred(m)), None)
		if found is None:
			raise KeyError(f"Not found: {target!r}")

		updated = [
			Entry(
				id        = m.id,
				name      = new_name,
				comment   = m.comment,
				start     = m.start,
				end       = m.end,
				type      = m.type,
				loop_type = m.loop_type,
				note      = m.note,
			)

			if m is found else m
			for m in self._cue_fields
		]

		return self._copy(updated)

	def save(
		self,
		path: Optional[str] = None,
		*,

		synced:         Optional[bool]            = None,
		note_root:      Optional[Union[int, str]] = None,
		time_signature: Optional[str]             = None,
		tempo:          Optional[float]           = None,
		num_beats:      Optional[int]             = None,
	) -> None:
		"""
		Writes the current marker state to disk.

		ACID tempo metadata kwargs (synced, note_root, time_signature, tempo,
		num_beats) are optional overrides - any left as None preserve the
		value already present in the loaded file (or a sane default if the
		file had no acid chunk and at least one override is given).

		Parameters
		----------
			path (Optional[str]):
				Output path; defaults to the originally loaded path.

			synced (Optional[bool]):
				True = Synced tempo mode, False = Free tempo mode.

			note_root (Optional[Union[int, str]]):
				Root note, e.g. Note.Cb6 or "Cb6".

			time_signature (Optional[str]):
				e.g. "4/4".

			tempo (Optional[float]):
				BPM to inject to file.

				If -1, detects tempo and saves it into file forcefully.
				If -2, detects tempo and saves it into file if it doesn't have it already.

			num_beats (Optional[int]):
				Loop length in beats.

		Raises
		------
			ValueError:
				If path is None and self.path is also None (i.e. this
				instance was opened from bytes/BytesIO rather than a
				file path) - an explicit path is required in that case.
		"""
		out_path = path or self.path

		if not out_path:
			raise ValueError(
				"No output path given and this MarkerFile has no source path "
				"(opened from bytes/BytesIO) - pass save(path = ...) explicitly."
			)

		if tempo == -1:
			tempo = detect_tempo(self._file, mode = 3)
		if tempo == -2:
			bpm = detect_tempo(self._file, mode = 3)

			if not self._tempo:
				tempo = bpm

		raw_out = _build_file(
			self._raw_file,
			self._cue_fields,

			smpl_raw_header = self._smpl_raw_header,
			acid            = self._acid,
			synced          = synced,
			note_root       = note_root,
			time_signature  = time_signature,
			tempo           = tempo,
			num_beats       = num_beats,
		)

		with open(out_path, "wb") as f:
			f.write(raw_out)

		if not self._acid:
			self._acid = dict(_DEFAULT_ACID)

		self._acid = _apply_acid_overrides(
			dict(self._acid),
			synced         = synced,
			note_root      = note_root,
			time_signature = time_signature,
			tempo          = tempo,
			num_beats      = num_beats,
		)

	def copy(self, dest: str, fixed_time: Optional[bool] = True) -> None:
		"""
		Copies all markers, regions, and loops onto another file.

		Opens dest as a MarkerFile, replays all entries from this instance
		onto it via add_entry(), and saves the result.

		Entries that already exist in dest (matched by equality) are skipped silently.

		Parameters
		----------
			dest (str):
				Path to an existing file to copy the entries onto.

				The file must be valid; its audio data and non-marker chunks are preserved unchanged.

			fixed_time (bool, optional):
				Determines whether Entries should have start/end positions
				recalculated in order to match the dest file sample rate.
		"""
		target = MarkerFile(dest)

		src_rate = self._sample_rate
		dst_rate = target._sample_rate

		rates_differ = src_rate and dst_rate and src_rate != dst_rate

		for m in self._cue_fields:
			try:
				if fixed_time and rates_differ:
					ratio = dst_rate / src_rate

					scaled = Entry(
						name      = m.name,
						start     = math.floor(m.start * ratio),
						end       = math.ceil(m.end * ratio) if m.end is not None else None,
						type      = m.type,
						loop_type = m.loop_type,
						comment   = m.comment,
						note      = m.note
					)

					target = target.add_entry(scaled)
				else:
					target = target.add_entry(m)
			except ValueError:
				pass # already exists, skip

		target.save()  # ← outside the loop

	def export(self, obj: Entry, path: str = None) -> None:
		"""
		Exports the audio slice covered by a region or loop to a new file.

		Reads the audio frames from self.path between obj.start and
		obj.end and writes them to a new file with the same format
		parameters (sample rate, channels, bit depth) as the source.

		Parameters
		----------
			obj (Entry):
				A region or loop Entry whose start and end define the slice to export.

				Must have is_region or is_loop True.

			path (Optional[str]):
				Output file path.

				Basename defaults to `{marker.name}` if `marker.name`
				is non-empty, or "region_{marker.start}_{marker.end}" otherwise.

		Raises
		------
			ValueError:
				If object is a point marker (is_marker is True),
				since point markers have no range to extract audio from.
		"""
		if not (obj.is_region or obj.is_loop):
			raise ValueError(
				f"Entry {obj.name!r} is a marker - no range to export"
			)

		if not path:
			path = (obj.name or f"region_{obj.start}_{obj.end}") + ".wav"

		import wave

		with wave.open(self.path, "rb") as src:
			params = src.getparams()
			src.setpos(obj.start)
			frames = src.readframes(obj.end - obj.start)

		with wave.open(path, "wb") as dst:
			dst.setparams(params)
			dst.writeframes(frames)

	#-=-=-=-#
	# Utility

	def markers_to_regions(self, copy_names: bool = True):
		"""
		Converts consecutive point markers into regions.

		Sorts all point markers by start sample and pairs each one with
		the next as a region, where the current marker's start becomes
		the region start and the next marker's start becomes the region
		end.

		The last point marker is consumed as the  closing boundary
		and does not produce a region of its own.

		Existing regions and loops are preserved unchanged and
		included in the result alongside the newly created regions.

		Parameters
		----------
			copy_names (bool):
				If True (default), a newly created region whose source
				marker had an empty name inherits the name of the next
				marker (the one that defines its end boundary).

				If False, empty names remain empty.

		Returns
		-------
			MarkerFile:
				A new MarkerFile instance with point markers converted to regions.

				The original _cue_fields list is not modified.
		"""
		points  = sorted([m for m in self._cue_fields if m.is_marker], key = lambda m: m.start)
		regions = [m for m in self._cue_fields if m.is_region or m.is_loop]

		new_regions = []
		for i, m in enumerate(points):
			end = points[i + 1].start if i + 1 < len(points) else None

			if end is None:
				# last point is consumed as endpoint, drop it
				pass
			else:
				name = m.name

				if copy_names and not name:
					name = points[i + 1].name

				new_regions.append(
					Entry(
						id      = m.id,
						name    = name,
						comment = m.comment,
						start   = m.start,
						end     = end,
						type    = m.type,
						note    = m.note,
					)
				)

		return self._copy(regions + new_regions)

	def search_markers(self, *, types = None, at = None) -> bool:
		"""
		Checks whether any point markers exist matching the given filters.

		All provided filters must match; omitted filters match any value.

		Parameters
		----------
			types (Union[str, List[str], None]):
				A FourCC type string or list of type strings to filter by.
				Matches any type if None.

			at (Optional[int]):
				A sample position to filter by.
				Matches any position if None.

		Returns
		-------
			bool:
				True if at least one point marker satisfies
				all provided filters, False otherwise.
		"""
		return any(
			(types is None or m.type in _types_normalize(types))
			and (at is None or m.start == at)
			for m in self.markers
		)

	def search_regions(self, *, types = None, start_at = None, end_at = None) -> bool:
		"""
		Checks whether any regions exist matching the given filters.

		All provided filters must match; omitted filters match any value.
		Loop entries are not searched - use search_loops() for those.

		Parameters
		----------
			types (Union[str, List[str], None]):
				A FourCC type string or list of type strings to filter by.
				Matches any type if None.

			start_at (Optional[int]):
				A sample position to match against each region's start.
				Matches any start if None.

			end_at (Optional[int]):
				A sample position to match against each region's end.
				Matches any end if None.

		Returns
		-------
			bool:
				True if at least one region satisfies
				all provided filters, False otherwise.
		"""
		return any(
			(types is None or m.type in _types_normalize(types))
			and (start_at is None or m.start == start_at)
			and (end_at is None or m.end == end_at)
			for m in self.regions
		)

	def search_loops(self, *, start_at = None, end_at = None, loop_type = None) -> bool:
		"""
		Checks whether any loops exist matching the given filters.

		All provided filters must match; omitted filters match any value.

		Parameters
		----------
			start_at (Optional[int]):
				A sample position to match against each loop's start.
				Matches any start if None.

			end_at (Optional[int]):
				A sample position to match against each loop's end.
				Matches any end if None.

			loop_type (Optional[int]):
				A LoopType constant to filter by.
				Matches any loop type if None.

		Returns
		-------
			bool:
				True if at least one loop satisfies
				all provided filters, False otherwise.
		"""
		return any(
			(start_at is None or m.start == start_at)
			and (end_at is None or m.end == end_at)
			and (loop_type is None or m.loop_type == loop_type)
			for m in self.loops
		)

	#-=-=-=-#
	# Inspectors
	def is_empty(self) -> bool:
		"""
		Returns
		-------
			bool:
				True if the file contains no markers, regions, or loops.
		"""
		return not self._cue_fields

	@property
	def markers(self) -> List[Entry]:
		"""
		Returns
		-------
			List[Entry]:
				All point marker entries (is_marker is True),
				sorted by start sample position.
		"""
		return [m for m in self._cue_fields if m.is_marker]

	@property
	def regions(self) -> List[Entry]:
		"""All regions (non-loop), sorted by start."""
		return [m for m in self._cue_fields if m.is_region]

	@property
	def loops(self) -> List[Entry]:
		"""
		Returns
		-------
			List[Entry]:
				All loop entries (is_loop is True),
				sorted by start sample position.

				Stored in the smpl chunk rather than cue / adtl.
		"""
		return [m for m in self._cue_fields if m.is_loop]

	@property
	def all(self) -> List[Entry]:
		"""
		Returns
		-------
			List[Entry]:
				All entries (markers, regions, and loops) in their
				current internal order, sorted by start sample position.
		"""
		return self._cue_fields

	@property
	def synced(self) -> Optional[bool]:
		"""
		Current ACID tempo mode.

		True = Synced,
		False = Free
		None = no acid chunk present.

		Reflects the last save() override if one has been applied.
		Otherwise reflects the value loaded from disk.

		Override per-save via save(synced = bool)
		"""
		return self._acid.get("synced") if self._acid else None

	@property
	def time_signature(self) -> Tuple[int, int] | None:
		"""
		Current ACID time signature.

		Returns
		-------
			[int, int]:
				If loaded time signature is valid.

			None:
				If loaded time signature is invalid.
		"""
		denominator = self._acid.get("meter_denominator") if self._acid else None
		numerator = self._acid.get("meter_numerator") if self._acid else None

		if denominator and numerator:
			return denominator, numerator

		return None

	@property
	def tempo(self) -> int | None:
		"""
		Current ACID tempo.

		Returns
		-------
			int:
				If loaded tempo is valid.

			None:
				If no loaded tempo.
		"""
		return self._acid.get("tempo") if self._acid else None

	# aliases
	ts = time_signature
	mtr = markers_to_regions
	remove = delete_entry