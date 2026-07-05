class WaveMarksError(Exception):
	"""
	Base class for all wavemarks-specific exceptions.
	"""
	pass


class OutOfBoundsError(WaveMarksError):
	"""
	Raised when a sample position exceeds the audio data length.
	"""
	pass


class InvalidRangeError(WaveMarksError):
	"""
	Raised when end <= start for a region or loop.
	"""
	pass


class DuplicateBoundsWarning(UserWarning):
	"""
	Raised as a warning (not exception) when adding an entry whose start/end
	bounds exactly match an existing region or loop of a different kind
	(e.g. adding a loop where a region already exists with the same bounds).
	"""
	pass


class FieldMismatchWarning(UserWarning):
	"""
	Raised when fields exclusive to one Entry kind are set while
	creating a different kind (e.g. loop_type set on a marker,
	or note set on a loop).
	"""
	pass