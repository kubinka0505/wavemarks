from .region import Region
from .type import MarkerType

from typing import Optional, Union
from dataclasses import dataclass, field

#-=-=-=-#

@dataclass
class Entry:
	name:      str
	start:     int
	end:       Optional[int] = None
	type:      str           = MarkerType.BASIC
	loop_type: Optional[int] = None
	comment:   Optional[str] = None
	note:      Optional[int] = None
	id:        int           = field(default = 0, repr = False)

	def __eq__(self, other: object) -> bool:
		"""
		Checks equality between two Entry instances.

		Comparison is based on name, start, type, end, and loop_type only.
		The internal id field is excluded, allowing a user-constructed
		Entry (with id = 0) to match a stored one (with an assigned id).

		Parameters
		----------
			other (object):
				The object to compare against.

		Returns
		-------
			bool:
				True if other is a Entry and all compared fields match.
				NotImplemented if other is not a Entry.
		"""
		if not isinstance(other, Entry):
			return NotImplemented

		return (
			self.name          == other.name
			and self.start     == other.start
			and self.type      == other.type
			and self.end       == other.end
			and self.loop_type == other.loop_type # distinguishable
		)

	def __hash__(self):
		"""
		Returns a hash of the Entry based on its identity fields.

		Hashes name, start, type, end, and loop_type - the same fields
		used by __eq__ - so that equal objects produce equal hashes,
		satisfying the data model requirement.

		Returns
		-------
			int:
				Hash value suitable for use in sets and as dict keys.
		"""
		return hash((self.name, self.start, self.type, self.end, self.loop_type))

	#-=-=-=-#
	# Properties

	@property
	def is_marker(self) -> bool:
		"""
		Returns
		-------
			bool:
				True if this Entry is a marker, i.e. end is None and loop_type is None.
				Mutually exclusive with is_region and is_loop.
		"""
		return self.end is None and self.loop_type is None

	@property
	def is_region(self) -> bool:
		"""
		Returns
		-------
			bool:
				True if this entry is a standard region, i.e. end is set and loop_type is None.

				Stored in the cue/adtl chunks.
				Mutually exclusive with is_marker and is_loop.
		"""
		return self.end is not None and self.loop_type is None

	@property
	def is_loop(self) -> bool:
		"""
		Returns
		-------
			bool:
				True if this entry is a loop, i.e. loop_type is set.

				Stored in the smpl chunk rather than the cue chunk.
				Mutually exclusive with is_marker and is_region.
		"""
		return self.loop_type is not None and self.end is not None

	@property
	def length(self) -> int:
		"""
		Returns
		-------
			int:
				The number of samples spanned by a region or loop
				(end - start). Returns 0 for point markers.
		"""
		return (self.end - self.start) if (self.is_region or self.is_loop) else 0

	#-=-=-=-#
	# Internal

	def _pos_key(self):
		"""
		Returns a hashable position key used for internal matching.

		Returns
		-------
			Union[int, Tuple[int, int]]:
				A plain int (start) for point markers, or a (start, end)
				tuple for regions and loops.
		"""
		return (self.start, self.end) if (self.is_region or self.is_loop) else self.start

	def _matches(
		self,
		pos: Union[int, Region],
		name: str,
		type_: str
	) -> bool:

		"""
		Checks whether this Entry matches a given lookup key.

		Used internally by _find() to locate entries by position, name, and type.

		Region position matches only against is_region/is_loop entries
		int position matches only against is_marker entries.

		Parameters
		----------
			pos (Union[int, Region]):
				The position to match.

				int matches point markers by start sample
				Region matches ranged entries by both start and end sample.

			name (str):
				The label to match against this entry's name field.

			type (str):
				The FourCC marker type to match against this entry's type field.

		Returns
		-------
			bool:
				True if all three of pos, name, and type match this
				entry's fields.
		"""
		if isinstance(pos, Region):
			return (
				(self.is_region or self.is_loop)
				and self.name  == name
				and self.type  == type_
				and self.start == pos.start
				and self.end   == pos.end
			)

		return (
			self.is_marker
			and self.name  == name
			and self.type  == type_
			and self.start == pos
		)

	def needs_ltxt(self) -> bool:
		"""
		Determines whether this entry requires an ltxt sub-chunk on write.

		Regions always require an ltxt sub-chunk to carry the sample_length field.

		Point markers require one only for non-default types (any type other than "rgn "),
		matching certain software's convention of omitting ltxt for basic markers.

		Loop entries never emit ltxt since their range is stored in the smpl chunk.

		Returns
		-------
			bool:
				True if an ltxt sub-chunk should be written for this entry.
		"""
		if self.is_region:
			return True

		return self.type not in MarkerType._LABL_ONLY_POINT