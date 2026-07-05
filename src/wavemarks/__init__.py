"""Library for managing marker/region-related objects in audio files."""

__author__  = "kubinka0505"
__credits__ = __author__
__version__ = "1.0"
__date__    = "05th July 2026"

#-=-=-=-#

from .__main__ import MarkerFile

from ._core.models.type import MarkerType
from ._core.models.entry import Entry
from ._core.models.region import Region, Position, LoopType
from ._core.models.note import Note