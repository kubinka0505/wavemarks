from typing import Union
from dataclasses import dataclass

#-=-=-=-#

@dataclass(frozen = True)
class Region:
	"""
	A half-open sample range [start, end]).

	Accepts:
		- Region(200, 400)
		- Region.from_range(range(200, 400))
	"""
	start: int
	end: int

	def __post_init__(self):
		"""
		Validates the region after dataclass construction.

		Raises
		------
			ValueError:
				If end is not strictly greater than start, since a region
				or loop must span at least one sample.
		"""
		if self.end <= self.start:
			raise ValueError(f"Region end ({self.end}) must be > start ({self.start})")

	@classmethod
	def from_range(cls, r: range) -> "Region":
		"""
		Builds a Region from a `range` object.

		Parameters
		----------
			r (range):
				A range whose start and stop define the region bounds.
				r.stop is treated as the exclusive end sample.

		Returns
		-------
			Region:
				A new Region with start = r.start and end = r.stop.
		"""
		return cls(r.start, r.stop)

	@property
	def length(self) -> int:
		"""
		Returns
		-------
			int:
				The number of samples spanned by the region (end - start).
		"""
		return self.end - self.start

	def __repr__(self):
		"""
		Returns
		-------
			str:
				A human-readable representation of the region, e.g.
				"Region(200, 400)".
		"""
		return f"Region({self.start}, {self.end})"

#-=-=-=-#

class LoopType:
	FORWARD   = 0
	PING_PONG = 1
	REVERSE   = 2

	_ALL = { FORWARD, PING_PONG, REVERSE }

	_NAMES = {
		FORWARD:   "forward",
		PING_PONG: "ping-pong",
		REVERSE:   "reverse",
	}

	@classmethod
	def name(cls, value: int) -> str:
		"""
		Resolves a loop type constant to its human-readable name.

		Parameters
		----------
			value (int):
				A loop type value, typically one of
				LoopType.FORWARD, LoopType.PING_PONG, or LoopType.REVERSE.

		Returns
		-------
			str:
				The lowercase name of the loop type
				(e.g. "forward", "ping-pong", "reverse"), or "unknown(<value>)"
				if the value does not match a known loop type.
		"""
		return cls._NAMES.get(value, f"unknown({value})")

	@classmethod
	def is_valid(cls, value: int) -> bool:
		"""
		Checks whether a value is a recognised loop type.

		Parameters
		----------
			value (int):
				The value to validate.

		Returns
		-------
			bool:
				True if value matches one of the defined loop type constants, otherwise False.
		"""
		return value in cls._ALL

#-=-=-=-#
# accepted by the public API

Position = Union[int, Region, range, tuple, list]