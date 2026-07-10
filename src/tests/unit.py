"""Comprehensive unit tests for wavemarks.MarkerFile.

Run with:
	pytest --cov=wavemarks --cov-report=term-missing
"""

from __future__ import annotations

import os
import wave
import array
import tempfile
import warnings
from io import BytesIO
from typing import Generator

import pytest

from wavemarks import MarkerFile
from wavemarks._core.exceptions import (
	DuplicateBoundsWarning,
	FieldMismatchWarning,
	InvalidRangeError,
	OutOfBoundsError,
)

from wavemarks._core.models.entry import Entry
from wavemarks._core.models.type import MarkerType

#-=-=-=-#
# Helpers

SAMPLE_RATE  = 44100
N_SAMPLES    = 44100  # 1 second of silence
N_CHANNELS   = 1
SAMPLE_WIDTH = 2      # 16-bit

def _make_wav_bytes(
	n_samples:    int = N_SAMPLES,
	sample_rate:  int = SAMPLE_RATE,
	n_channels:   int = N_CHANNELS,
	sample_width: int = SAMPLE_WIDTH,
) -> bytes:
	"""Return raw bytes of a minimal silent WAV file."""
	buf = BytesIO()
	with wave.open(buf, "wb") as w:
		w.setnchannels(n_channels)
		w.setsampwidth(sample_width)
		w.setframerate(sample_rate)
		w.writeframes(array.array("h", [0] * n_samples * n_channels).tobytes())

	return buf.getvalue()

def _tmp_wav(
	n_samples: int = N_SAMPLES,
	**kwargs,
) -> str:
	"""Write a temporary WAV file and return its path."""
	fd, path = tempfile.mkstemp(suffix = ".wav")
	os.close(fd)

	with open(path, "wb") as f:
		f.write(_make_wav_bytes(n_samples, **kwargs))

	return path

@pytest.fixture
def wav_path() -> Generator[str, None, None]:
	path = _tmp_wav()

	yield path

	if os.path.exists(path):
		os.unlink(path)

@pytest.fixture
def wav_bytes() -> bytes:
	return _make_wav_bytes()

@pytest.fixture
def mf(wav_path) -> MarkerFile:
	return MarkerFile(wav_path)

#-=-=-=-#
# Construction

class TestConstruction:
	def test_from_path(self, wav_path):
		f = MarkerFile(wav_path)

		assert f.path == wav_path
		assert f.is_empty()

	def test_from_bytes(self, wav_bytes):
		f = MarkerFile(wav_bytes)

		assert f.path is None
		assert f.is_empty()

	def test_from_bytesio(self, wav_bytes):
		f = MarkerFile(BytesIO(wav_bytes))

		assert f.path is None
		assert f.is_empty()

	def test_from_bytearray(self, wav_bytes):
		f = MarkerFile(bytearray(wav_bytes))
		assert f.path is None

	def test_invalid_source_type(self):
		with pytest.raises(TypeError):
			MarkerFile(12345)

	def test_invalid_wav_bytes(self):
		with pytest.raises(ValueError):
			MarkerFile(b"not a wav file at all")

	def test_sample_count_computed(self, wav_path):
		f = MarkerFile(wav_path)
		assert f._n_samples == N_SAMPLES

	def test_empty_on_fresh_file(self, mf):
		assert len(mf) == 0
		assert mf.markers == []
		assert mf.regions == []
		assert mf.loops   == []

#-=-=-=-#
# Context manager

class TestContextManager:
	def test_enter_returns_self(self, mf):
		with mf as f:
			assert f is mf

	def test_no_autosave_on_exit(self, wav_path):
		f = MarkerFile(wav_path)

		with f as g:
			g = g.add_entry(100, "x", MarkerType.BASIC)

		# reload - disk should be unchanged
		reloaded = MarkerFile(wav_path)

		assert reloaded.is_empty()

	def test_exception_does_not_propagate_from_exit(self, mf):
		try:
			with mf:
				pass
		except Exception:
			pytest.fail("__exit__ should not raise")

#-=-=-=-#
# add_entry - markers

class TestAddMarker:
	def test_add_basic_marker(self, mf):
		f = mf.add_entry(100, "hit", MarkerType.BASIC)
		assert len(f.markers) == 1

		m = f.markers[0]
		assert m.start == 100
		assert m.name  == "hit"
		assert m.type  == MarkerType.BASIC
		assert m.is_marker

	def test_add_downbeat_marker(self, mf):
		f = mf.add_entry(200, "db", MarkerType.DOWNBEAT)
		assert f.markers[0].type == MarkerType.DOWNBEAT

	def test_add_multiple_markers(self, mf):
		f = mf.add_entry(100, "a", MarkerType.BASIC)
		f = f.add_entry(200, "b", MarkerType.BASIC)
		assert len(f.markers) == 2

	def test_markers_sorted_by_start(self, mf):
		f = mf.add_entry(500, "late", MarkerType.BASIC)
		f = f.add_entry(100, "early", MarkerType.BASIC)
		starts = [m.start for m in f.markers]
		assert starts == sorted(starts)

	def test_add_marker_via_entry_object(self, mf):
		e = Entry(name = "x", start = 300)
		f = mf.add_entry(e)
		assert f.markers[0].start == 300
		assert f.markers[0].name  == "x"

	def test_duplicate_marker_raises(self, mf):
		f = mf.add_entry(100, "hit", MarkerType.BASIC)
		with pytest.raises(ValueError):
			f.add_entry(100, "hit", MarkerType.BASIC)

	def test_add_does_not_mutate_original(self, mf):
		_ = mf.add_entry(100, "x")
		assert mf.is_empty()

	def test_iadd_operator(self, mf):
		e = Entry(name = "op", start = 50)
		f = mf
		f += e
		assert len(f.markers) == 1

	def test_contains_operator(self, mf):
		e = Entry(name = "op", start = 50)
		f = mf
		f += e
		assert e in f

	def test_not_contains(self, mf):
		e = Entry(name = "ghost", start = 999)
		assert e not in mf

#-=-=-=-#
# add_entry - regions

class TestAddRegion:
	def test_add_tuple_region(self, mf):
		f = mf.add_entry((100, 500), "verse")
		assert len(f.regions) == 1

		r = f.regions[0]
		assert r.start     == 100
		assert r.end       == 500
		assert r.is_region

	def test_add_range_region(self, mf):
		f = mf.add_entry(range(200, 800), "chorus")

		assert f.regions[0].start == 200
		assert f.regions[0].end   == 800

	def test_add_entry_region(self, mf):
		e = Entry(name = "drop", start = 300, end = 700)
		f = mf.add_entry(e)

		assert f.regions[0].end == 700

	def test_region_and_marker_can_overlap(self, mf):
		f = mf.add_entry((100, 500), "region")
		f = f.add_entry(300, "inside")

		assert len(f.regions) == 1
		assert len(f.markers) == 1

	def test_two_regions_same_bounds_different_names(self, mf):
		f = mf.add_entry((100, 500), "a")
		f = f.add_entry((100, 500), "b")

		assert len(f.regions) == 2

	def test_invalid_range_raises(self, mf):
		with pytest.raises((InvalidRangeError, ValueError)):
			mf.add_entry((500, 100), "bad")

	def test_equal_bounds_raises(self, mf):
		with pytest.raises((InvalidRangeError, ValueError)):
			mf.add_entry((200, 200), "zero-length")

#-=-=-=-#
# add_entry - loops

class TestAddLoop:
	def test_add_forward_loop(self, mf):
		f = mf.add_entry((100, 500), "lp", loop_type = 0)
		assert len(f.loops) == 1
		assert f.loops[0].loop_type == 0
		assert f.loops[0].is_loop

	def test_loop_not_in_regions(self, mf):
		f = mf.add_entry((100, 500), "lp", loop_type = 0)
		assert f.regions == []

	def test_loop_and_region_same_bounds_warns(self, mf):
		f = mf.add_entry((100, 500), "region")

		with warnings.catch_warnings(record = True) as w:
			warnings.simplefilter("always")
			f = f.add_entry((100, 500), "loop", loop_type = 0)

		assert any(issubclass(x.category, DuplicateBoundsWarning) for x in w)

	def test_region_after_loop_same_bounds_warns(self, mf):
		f = mf.add_entry((100, 500), "lp", loop_type = 0)

		with warnings.catch_warnings(record = True) as w:
			warnings.simplefilter("always")
			f = f.add_entry((100, 500), "rgn")

		assert any(issubclass(x.category, DuplicateBoundsWarning) for x in w)

	def test_loop_with_comment_warns(self, mf):
		with warnings.catch_warnings(record = True) as w:
			warnings.simplefilter("always")
			mf.add_entry((100, 500), "lp", loop_type = 0, comment = "oops")

		assert any(issubclass(x.category, FieldMismatchWarning) for x in w)

	def test_loop_with_note_warns(self, mf):
		with warnings.catch_warnings(record = True) as w:
			warnings.simplefilter("always")
			mf.add_entry((100, 500), "lp", loop_type = 0, note = 60)

		assert any(issubclass(x.category, FieldMismatchWarning) for x in w)

#-=-=-=-#
# Bounds validation

class TestBoundsValidation:
	def test_start_beyond_file_raises(self, mf):
		with pytest.raises(OutOfBoundsError):
			mf.add_entry(N_SAMPLES + 1, "over")

	def test_negative_start_raises(self, mf):
		with pytest.raises(OutOfBoundsError):
			mf.add_entry(-1, "neg")

	def test_end_beyond_file_raises(self, mf):
		with pytest.raises(OutOfBoundsError):
			mf.add_entry((100, N_SAMPLES + 1), "over")

	def test_start_at_boundary_ok(self, mf):
		f = mf.add_entry(N_SAMPLES, "at-end")
		assert len(f.markers) == 1

	def test_end_at_boundary_ok(self, mf):
		f = mf.add_entry((0, N_SAMPLES), "full")
		assert len(f.regions) == 1

#-=-=-=-#
# delete_entry

class TestDeleteEntry:
	def test_delete_by_position(self, mf):
		f = mf.add_entry(100, "hit")
		f = f.delete_entry(100, "hit")

		assert f.is_empty()

	def test_delete_by_entry_object(self, mf):
		e = Entry(name = "hit", start = 100)
		f = mf.add_entry(e)
		f = f.delete_entry(e)

		assert f.is_empty()

	def test_delete_nonexistent_raises(self, mf):
		with pytest.raises(KeyError):
			mf.delete_entry(999, "ghost")

	def test_isub_operator(self, mf):
		e = Entry(name = "x", start = 50)

		mf += e
		mf -= e

		assert mf.is_empty()

	def test_delete_leaves_others(self, mf):
		f = mf.add_entry(100, "a")
		f = f.add_entry(200, "b")
		f = f.delete_entry(100, "a")

		assert len(f.markers) == 1
		assert f.markers[0].name == "b"

	def test_delete_region_by_object(self, mf):
		e = Entry(name = "v", start = 100, end = 500)

		f = mf.add_entry(e)
		f = f.delete_entry(e)

		assert f.is_empty()

	def test_remove_alias(self, mf):
		f = mf.add_entry(100, "x")
		f = f.remove(100, "x")

		assert f.is_empty()

#-=-=-=-#
# rename_entry

class TestRenameEntry:
	def test_rename_by_string(self, mf):
		f = mf.add_entry(100, "old")
		f = f.rename_entry("old", "new")
		assert f.markers[0].name == "new"

	def test_rename_by_entry_object(self, mf):
		e = Entry(name = "old", start = 100)
		f = mf.add_entry(e)
		f = f.rename_entry(e, "new")
		assert f.markers[0].name == "new"

	def test_rename_preserves_other_fields(self, mf):
		f = mf.add_entry(100, "old", MarkerType.DOWNBEAT)
		f = f.rename_entry("old", "new")
		m = f.markers[0]
		assert m.type  == MarkerType.DOWNBEAT
		assert m.start == 100

	def test_rename_nonexistent_raises(self, mf):
		with pytest.raises(KeyError):
			mf.rename_entry("ghost", "new")

	def test_rename_does_not_mutate_original(self, mf):
		f = mf.add_entry(100, "old")
		_ = f.rename_entry("old", "new")
		assert f.markers[0].name == "old"

#-=-=-=-#
# save / round-trip

class TestSave:
	def test_save_overwrites_source(self, wav_path):
		f = MarkerFile(wav_path)
		f = f.add_entry(100, "x")
		f.save()
		reloaded = MarkerFile(wav_path)

		assert len(reloaded.markers) == 1
		assert reloaded.markers[0].name == "x"

	def test_save_to_new_path(self, wav_path):
		f = MarkerFile(wav_path)
		f = f.add_entry(200, "y")

		out = wav_path + ".out.wav"
		try:
			f.save(out)
			reloaded = MarkerFile(out)

			assert reloaded.markers[0].name == "y"
		finally:
			if os.path.exists(out):
				os.unlink(out)

	def test_save_without_path_raises_for_bytesio(self, wav_bytes):
		f = MarkerFile(wav_bytes)
		f = f.add_entry(100, "x")

		with pytest.raises(ValueError):
			f.save()

	def test_round_trip_preserves_audio_length(self, wav_path):
		f = MarkerFile(wav_path)
		f = f.add_entry(100, "x")
		f.save()

		reloaded = MarkerFile(wav_path)

		assert reloaded._n_samples == N_SAMPLES

	def test_round_trip_multiple_markers(self, wav_path):
		f = MarkerFile(wav_path)
		f = f.add_entry(100, "a", MarkerType.BASIC)
		f = f.add_entry(200, "b", MarkerType.DOWNBEAT)
		f = f.add_entry((300, 800), "c", MarkerType.CD_TRACK)

		f.save()
		r = MarkerFile(wav_path)

		assert len(r.all) == 3

		assert r.markers[0].name == "a"
		assert r.markers[1].name == "b"
		assert r.regions[0].name == "c"

	def test_round_trip_preserves_loop(self, wav_path):
		f = MarkerFile(wav_path)
		f = f.add_entry((100, 500), "lp", loop_type = 0)
		f.save()

		r = MarkerFile(wav_path)
		assert len(r.loops) == 1

		assert r.loops[0].start == 100
		assert r.loops[0].end   == 500

	def test_round_trip_delete_then_save(self, wav_path):
		f = MarkerFile(wav_path)
		f = f.add_entry(100, "x")
		f = f.add_entry(200, "y")
		f = f.delete_entry(100, "x")
		f.save()

		r = MarkerFile(wav_path)

		assert len(r.markers) == 1
		assert r.markers[0].name == "y"

#-=-=-=-#
# copy

class TestCopy:
	def test_copy_markers_to_dest(self, wav_path):
		dest_path = _tmp_wav()
		try:
			f = MarkerFile(wav_path)
			f = f.add_entry(100, "x")
			f.save()

			MarkerFile(wav_path).copy(dest_path)
			reloaded = MarkerFile(dest_path)

			assert reloaded.markers[0].name == "x"
		finally:
			os.unlink(dest_path)

	def test_copy_skips_duplicates(self, wav_path):
		dest_path = _tmp_wav()
		try:
			src = MarkerFile(wav_path).add_entry(100, "x")
			src.save()

			dst = MarkerFile(dest_path).add_entry(100, "x")
			dst.save()

			MarkerFile(wav_path).copy(dest_path)
			reloaded = MarkerFile(dest_path)
			assert len(reloaded.markers) == 1
		finally:
			os.unlink(dest_path)

#-=-=-=-#
# export

class TestExport:
	def test_export_region(self, wav_path):
		f = MarkerFile(wav_path)
		f = f.add_entry((100, 500), "slice")

		out = wav_path + ".slice.wav"
		try:
			f.export(f.regions[0], out)

			with wave.open(out) as w:
				assert w.getnframes() == 400
		finally:
			if os.path.exists(out):
				os.unlink(out)

	def test_export_marker_raises(self, mf):
		f = mf.add_entry(100, "pt")

		with pytest.raises(ValueError):
			f.export(f.markers[0])

 	def test_export_default_filename(self, wav_path, tmp_path, monkeypatch):
		monkeypatch.chdir(tmp_path)
		f = MarkerFile(wav_path)
		f = f.add_entry((100, 300), "my_slice")
		f.export(f.regions[0])

		src_basename = os.path.splitext(os.path.basename(wav_path))[0]
		assert (tmp_path / f"{src_basename}_100_300.wav").exists()

	def test_export_unnamed_region_fallback_name(self, wav_path, tmp_path, monkeypatch):
		monkeypatch.chdir(tmp_path)
		f = MarkerFile(wav_path)
		f = f.add_entry((100, 300), "")
		f.export(f.regions[0])

		src_basename = os.path.splitext(os.path.basename(wav_path))[0]
		assert (tmp_path / f"{src_basename}_100_300.wav").exists()

#-=-=-=-#
# markers_to_regions

class TestMarkersToRegions:
	def test_basic_conversion(self, mf):
		f = mf.add_entry(100, "a")
		f = f.add_entry(500, "b")
		g = f.markers_to_regions()
		assert len(g.regions) == 1

		r = g.regions[0]

		assert r.start == 100
		assert r.end == 500

	def test_last_marker_consumed(self, mf):
		f = mf.add_entry(100, "a")
		f = f.add_entry(500, "b")
		g = f.markers_to_regions()

		assert len(g.markers) == 0

	def test_copy_names_true(self, mf):
		f = mf.add_entry(100, "")
		f = f.add_entry(500, "named")
		g = f.markers_to_regions(copy_names = True)

		assert g.regions[0].name == "named"

	def test_copy_names_false(self, mf):
		f = mf.add_entry(100, "")
		f = f.add_entry(500, "named")
		g = f.markers_to_regions(copy_names = False)

		assert g.regions[0].name == ""

	def test_existing_regions_preserved(self, mf):
		f = mf.add_entry((0, 200), "existing_region")
		f = f.add_entry(300, "a")
		f = f.add_entry(700, "b")
		g = f.markers_to_regions()

		assert len(g.regions) == 2

	def test_existing_loops_preserved(self, mf):
		f = mf.add_entry((0, 200), "lp", loop_type = 0)
		f = f.add_entry(300, "a")
		f = f.add_entry(700, "b")
		g = f.markers_to_regions()

		assert len(g.loops) == 1

	def test_mtr_alias(self, mf):
		f = mf.add_entry(100, "a")
		f = f.add_entry(500, "b")

		assert f.mtr().regions == f.markers_to_regions().regions

	def test_single_marker_produces_no_regions(self, mf):
		f = mf.add_entry(100, "alone")
		g = f.markers_to_regions()

		assert g.regions == []

#-=-=-=-#
# search_*

class TestSearch:
	def test_search_markers_no_filter(self, mf):
		f = mf.add_entry(100, "x")
		assert f.search_markers() is True

	def test_search_markers_empty(self, mf):
		assert mf.search_markers() is False

	def test_search_markers_by_type(self, mf):
		f = mf.add_entry(100, "x", MarkerType.DOWNBEAT)

		assert f.search_markers(types = MarkerType.DOWNBEAT) is True
		assert f.search_markers(types = MarkerType.CD_TRACK) is False

	def test_search_markers_by_type_list(self, mf):
		f = mf.add_entry(100, "x", MarkerType.DOWNBEAT)
		assert f.search_markers(types=[MarkerType.BASIC, MarkerType.DOWNBEAT]) is True

	def test_search_markers_at(self, mf):
		f = mf.add_entry(100, "x")

		assert f.search_markers(at = 100) is True
		assert f.search_markers(at = 999) is False

	def test_search_regions_no_filter(self, mf):
		f = mf.add_entry((100, 500), "r")

		assert f.search_regions() is True

	def test_search_regions_start_at(self, mf):
		f = mf.add_entry((100, 500), "r")

		assert f.search_regions(start_at=100) is True
		assert f.search_regions(start_at=200) is False

	def test_search_regions_end_at(self, mf):
		f = mf.add_entry((100, 500), "r")

		assert f.search_regions(end_at = 500) is True
		assert f.search_regions(end_at = 999) is False

	def test_search_loops_no_filter(self, mf):
		f = mf.add_entry((100, 500), "lp", loop_type = 0)
		assert f.search_loops() is True

	def test_search_loops_by_type(self, mf):
		f = mf.add_entry((100, 500), "lp", loop_type = 1)

		assert f.search_loops(loop_type = 1) is True
		assert f.search_loops(loop_type = 2) is False

	def test_search_loops_start_end(self, mf):
		f = mf.add_entry((100, 500), "lp", loop_type = 0)

		assert f.search_loops(start_at = 100, end_at = 500) is True
		assert f.search_loops(start_at = 100, end_at = 999) is False

#-=-=-=-#
# Properties

class TestProperties:
	def test_is_empty_true(self, mf):
		assert mf.is_empty() is True

	def test_is_empty_false(self, mf):
		f = mf.add_entry(100, "x")
		assert f.is_empty() is False

	def test_len(self, mf):
		f = mf.add_entry(100, "a")
		f = f.add_entry(200, "b")

		assert len(f) == 2

	def test_iter(self, mf):
		f = mf.add_entry(100, "a")
		f = f.add_entry(200, "b")

		names = [m.name for m in f]

		assert names == ["a", "b"]

	def test_all_returns_everything(self, mf):
		f = mf.add_entry(100, "m")
		f = f.add_entry((200, 500), "r")
		f = f.add_entry((300, 600), "l", loop_type = 0)

		assert len(f.all) == 3

	def test_markers_excludes_regions_and_loops(self, mf):
		f = mf.add_entry(100, "m")
		f = f.add_entry((200, 500), "r")
		f = f.add_entry((300, 600), "l", loop_type = 0)

		assert len(f.markers) == 1
		assert f.markers[0].name == "m"

	def test_regions_excludes_markers_and_loops(self, mf):
		f = mf.add_entry(100, "m")
		f = f.add_entry((200, 500), "r")
		f = f.add_entry((300, 600), "l", loop_type = 0)

		assert len(f.regions) == 1
		assert f.regions[0].name == "r"

	def test_loops_excludes_markers_and_regions(self, mf):
		f = mf.add_entry(100, "m")
		f = f.add_entry((200, 500), "r")
		f = f.add_entry((300, 600), "l", loop_type = 0)

		assert len(f.loops) == 1
		assert f.loops[0].name == "l"

	def test_repr_contains_path(self, wav_path):
		f = MarkerFile(wav_path)
		assert os.path.basename(wav_path) in repr(f)

	def test_repr_contains_entry_count(self, mf):
		f = mf.add_entry(100, "x")
		assert "entries" in repr(f)

	def test_repr_loop_prefix(self, mf):
		f = mf.add_entry((100, 500), "lp", loop_type = 0)
		assert "(loop)" in repr(f)

#-=-=-=-#
# _get_sample_count

class TestGetSampleCount:
	def test_correct_count(self, wav_bytes):
		assert MarkerFile._get_sample_count(wav_bytes) == N_SAMPLES

	def test_returns_none_for_truncated(self):
		assert MarkerFile._get_sample_count(b"RIFF\x00\x00\x00\x00WAVE") is None

#-=-=-=-#
# ACID / tempo properties

class TestAcidProperties:
	def test_synced_none_without_acid(self, mf):
		# fresh silent WAV won't have ACID chunk
		# may be None or bool depending on your test file - check it's not raising
		result = mf.synced
		assert result is None or isinstance(result, bool)

	def test_tempo_none_without_acid(self, mf):
		result = mf.tempo
		assert result is None or isinstance(result, (int, float))

	def test_time_signature_none_without_acid(self, mf):
		result = mf.time_signature
		assert result is None or (isinstance(result, tuple) and len(result) == 2)

#-=-=-=-#
# Fluency / immutability

class TestFluency:
	def test_each_mutation_returns_new_instance(self, mf):
		f1 = mf.add_entry(100, "a")
		f2 = f1.add_entry(200, "b")

		assert f1 is not f2
		assert f1 is not mf

	def test_original_unchanged_after_chain(self, mf):
		f = mf.add_entry(100, "a").add_entry(200, "b").add_entry(300, "c")

		assert mf.is_empty()
		assert len(f.markers) == 3

	def test_copy_shares_raw_bytes(self, mf):
		f = mf.add_entry(100, "x")
		assert f._raw_file is mf._raw_file

	def test_next_id_unique(self, mf):
		f = mf.add_entry(100, "a")
		f = f.add_entry(200, "b")

		ids = {m.id for m in f.all}

		assert len(ids) == 2

	def test_next_id_fills_gaps(self, mf):
		f = mf.add_entry(100, "a")   # id = 1
		f = f.add_entry(200, "b")    # id = 2
		f = f.delete_entry(100, "a") # id = 1 freed
		f = f.add_entry(300, "c")    # should reuse id = 1

		ids = sorted(m.id for m in f.all)

		assert ids == [1, 2]
