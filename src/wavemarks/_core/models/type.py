class MarkerType:
	"""Common FourCCs marker types."""
	BASIC        = "rgn "
	BEAT         = "beat"
	DOWNBEAT     = "dwnb"
	CD_TRACK     = "trak"
	CD_INDEX     = "indx"
	SINGLE_CYCLE = "wtsc"
	INVALID      = "none"

	beat = BT  = BEAT
	dwnb = DB  = DOWNBEAT
	trak = CDT = CD_TRACK
	indx = IDX = CD_INDEX
	wtsc = SC  = SINGLE_CYCLE
	none = NUL = INVALID

	# "rgn " point-markers are written as labl-only (no ltxt) to match certain software.
	# Regions with any type always get an ltxt (needed for length).
	_LABL_ONLY_POINT = {"rgn "}

	@classmethod
	def encode(cls, purpose: str) -> bytes:
		"""
		Encodes a marker type string into a fixed 4-byte FourCC.

		Parameters
		----------
			purpose (str):
				The marker type identifier (e.g. "rgn ", "dwnb").

				Shorter strings are right-padded with spaces.
				Longer strings are truncated to 4 characters.

		Returns
		-------
			bytes:
				A 4-byte latin-1 encoded FourCC suitable for writing
				into an ltxt sub-chunk's purpose field.
		"""
		return (purpose + " ")[:4].encode("latin-1")

	@classmethod
	def decode(cls, raw: bytes) -> str:
		"""
		Decodes a 4-byte FourCC back into a marker type string.

		Parameters
		----------
			raw (bytes):
				Raw bytes read from an ltxt sub-chunk's purpose field.

				Only the first 4 bytes are used.

		Returns
		-------
			str:
				The latin-1 decoded 4-character marker type string.
		"""
		return raw[:4].decode("latin-1")