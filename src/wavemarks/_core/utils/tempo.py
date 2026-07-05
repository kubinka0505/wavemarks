"""Tempo detector

A reliable tempo detector with modes for different kinds of source material.

Detection modes:
	0: Quick / loop mode.
		For short, pre-edited loops (a few seconds), the most reliable signal
		isn't autocorrelation at all - it's the fact that loop content
		is almost always cut to an exact number of beats or bars.

		This mode solves:
		- bpm = beats * 60 / duration

		...for plausible beat counts and scores them, using onset
		periodicity only as a tie-breaker.

	1: Constant tempo.
		Global onset-flux autocorrelation over the whole track (the single-tempo estimator).

	2: Variable tempo.
		Slides a window over the onset envelope and estimates
		local tempo per window (a small tempogram), returning the median as the
		headline BPM (with an option to get the whole curve back).

	3: Universal.
		Looks at the audio and decides which of the above applies:
		- short clip -> mode 0,
		- long clip with locally-agreeing tempo -> mode 1,
		- long clip whose local tempo estimates disagree -> mode 2.

Usage:
	from tempo import detect

	bpm = detect("song.wav")                # mode 3 (universal) by default
	bpm = detect("loop.wav", mode = 0)      # short, pre-cut loop
	bpm = detect("song.wav", mode = 1)      # steady, one-tempo track
	bpm = detect("live_take.wav", mode = 2) # tempo drifts over time
	bpm, times, curve = detect(             # full tempo-over-time curve
		"live_take.wav", mode = 2,
		return_curve = True
	)
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import soundfile as sf
from scipy import signal

#-=-=-=-#
# Utils
#-=-=-=-#

def _load_audio(
	sig: Union[str, np.ndarray],
	channel: Optional[int]
) -> tuple[np.ndarray, int]:
	"""
	Loads the audio source.

	Returns
	-------
		np.ndarray, int:
			Audio data, sample rate
	"""
	if isinstance(sig, str):
		with sf.SoundFile(sig) as f:
			sr = f.samplerate
			sig = f.read(dtype = "float64")

	# normalize int PCM to [-1, 1] floats regardless of bit depth
	if np.issubdtype(sig.dtype, np.integer):
		max_val = float(np.iinfo(sig.dtype).max)
		sig /= max_val

	# downmix to mono if mandatory
	if sig.ndim > 1:
		sig = sig.mean(axis = 1)

	return sig, sr

def _beats_to_samples(
	beat_frames: list[int],
	frame_times: np.ndarray,
	sr: int,
) -> tuple[int, ...]:
	"""
	Map onset-envelope frame indices to sample indices in the original audio.
	"""
	samples = []
	n_times = len(frame_times)

	for f in beat_frames:
		f_clamped = min(max(f, 0), n_times - 1)
		samples.append(int(round(frame_times[f_clamped] * sr)))

	return tuple(samples)

#-=-=-=-#
# Core

def _adaptive_stft_params(
	sig_len: int
) -> tuple[int, int]:
	"""
	Pick n_fft/hop_length that fit the signal.

	The defaults (2048/512) work fine for full songs but are wasteful/broken on a 1-2 second loop,
	so it is scaled down for short clips so they still get a usable onset envelope.
	"""
	default_n_fft = 2048

	# leave room for at least approx 16 STFT frames if possible
	max_n_fft = max(256, 1 << int(math.floor(math.log2(max(sig_len // 8, 256)))))

	n_fft = min(default_n_fft, max_n_fft)
	hop_length = max(64, n_fft // 4)

	return n_fft, hop_length

def _onset_envelope(
	sig: np.ndarray,
	sr: int,

	n_fft: int = 2048,
	hop_length: int = 512
) -> tuple[np.ndarray, float, np.ndarray]:
	"""
	Compute a spectral-flux onset-strength envelope.

	Returns (envelope, frames_per_second, frame_times).

	frame_times gives the time (in seconds, from the start of `sig`)
	each envelope sample corresponds to, taken straight from scipy's STFT
	frame centers (and shifted by one to line up with the frame-differencing used for flux)
	so beat/onset frame indices can be converted back to exact sample
	positions instead of assuming a uniform hop_length grid.
	"""
	_, times, stft = signal.stft(
		sig,
		fs = sr,
		nperseg = n_fft,
		noverlap = n_fft - hop_length,
		boundary = None
	)

	mag = np.abs(stft)

	# Log-magnitude compression makes flux far less sensitive to whichever
	# frequency band happens to be loudest (bass drum vs. hi-hat vs. vocal)
	log_mag = np.log1p(mag * 1000.0)

	# Frame-to-frame difference, half-wave rectified
	# (only increases count as onsets, matching how percussive/attack energy behaves)
	flux = np.diff(log_mag, axis = 1)
	flux = np.maximum(flux, 0.0)

	# sum across frequency bins -> one value per frame
	env = flux.sum(axis = 0)
	frame_times = times[1:] # difference drops the first STFT frame

	# smooth slightly and remove the DC/mean so autocorrelation isn't dominated by an offset
	if len(env) > 5:
		smooth_win = min(5, len(env) - (1 - len(env) % 2))

		if smooth_win >= 3:
			env = signal.savgol_filter(env, smooth_win | 1, 2)

	env = env - env.mean()

	frames_per_second = sr / hop_length
	return env, frames_per_second, frame_times

#-=-=-=-#
## Mode 1
#-=-=-=-#
def _parabolic_peak(
	sig: np.ndarray,
	i: int
) -> float:
	"""
	Sub-sample peak location via parabolic interpolation around index `i`.
	"""
	if i <= 0 or i >= len(sig) - 1:
		return float(i)

	sig0, sig1, sig2 = sig[i - 1], sig[i], sig[i + 1]

	denom = sig0 - 2 * sig1 + sig2
	if not denom:
		return float(i)

	offset = 0.5 * (sig0 - sig2) / denom

	return i + offset

def _tempo_from_envelope(
	env: np.ndarray,
	frames_per_second: float,

	bpm_min: float = 40.0,
	bpm_max: float = 220.0,

	preferred_bpm: float = 120.0,
	preferred_sigma_octaves: float = 1.0,
) -> Optional[float]:
	"""
	Autocorrelate the onset envelope and pick the best tempo, biasing
	away from octave errors with a soft log-normal prior centered on
	`preferred_bpm`.
	
	This is the same method librosa/Ellis-style beat trackers use so
	2x/0.5x tempo guesses don't win just because percussive subdivisions
	autocorrelate strongly as well.

	Returns
	-------
		None:
			Only when the envelope is too short/quiet to say anything.
			Callers that scan many small windows (mode 2) rely on this to just skip bad windows.
	"""
	if len(env) == 0 or np.allclose(env, 0):
		return None

	# full autocorrelation, keep the non-negative lags.
	corr = np.correlate(env, env, mode = "full")
	corr = corr[len(corr) // 2:]
	corr[0] = 0 # ignore the zero-lag self-similarity spike

	lag_min = max(1, int(round(frames_per_second * 60.0 / bpm_max)))
	lag_max = min(len(corr) - 2, int(round(frames_per_second * 60.0 / bpm_min)))

	if lag_max <= lag_min:
		return None

	window = corr[lag_min: lag_max + 1]

	# candidate lags = local maxima within the search window
	peak_idx_local, _ = signal.find_peaks(window)

	if not len(peak_idx_local):
		# fallback to the single global max if no clean local peaks exist
		peak_idx_local = np.array([int(np.argmax(window))])

	best_score = -np.inf
	best_bpm = None

	for local_i in peak_idx_local:
		lag = lag_min + local_i
		refined_lag = _parabolic_peak(corr, lag)

		if refined_lag <= 0:
			continue

		bpm = 60.0 * frames_per_second / refined_lag

		# strength of this periodicity in the signal itself
		strength = corr[lag]

		# soft log-normal prior favoring musically common tempo, so a
		# correlated subdivision (e.g. true tempo x2) doesn't win purely
		# because sixteenth-note hats autocorrelate strongly
		octave_dist = math.log2(bpm / preferred_bpm)
		prior = math.exp(
			-(octave_dist ** 2) / (2 * preferred_sigma_octaves ** 2)
		)

		score = strength * prior
		if score > best_score:
			best_score = score
			best_bpm = bpm

	return best_bpm

#-=-=-=-#
## Beat tracking
# Turn a tempo (or tempo curve) into actual beat
# sample positions, via a dynamic-programming
# beat tracker (Ellis 2007 style).
#-=-=-=- #

def _dp_beat_track(
	env: np.ndarray,
	period_at_frame,
	alpha: float = 100.0,
) -> list[int]:
	"""
	Find the sequence of onset-envelope frame indices that best explains
	the envelope as a sequence of evenly (or locally-evenly) spaced beats.

	`period_at_frame(i)` returns the expected inter-beat period, in frames,
	around frame i - a constant for fixed-tempo audio, or interpolated from
	a tempo curve for variable-tempo audio.

	This is the standard "score + backtrack" dynamic program:
	- each frame's cumulative score is its own onset strength
	plus the best achievable score of a previous beat
	penalized by how far that previous beat's spacing deviates (in log space)
	from the expected period.

	Backtracking the best-scoring path recovers the beat sequence.
	"""
	n = len(env)
	if not n:
		return []

	onset = env.astype(np.float64)
	std = onset.std()
	if std > 1e-12:
		onset = onset / std

	cum_score = np.full(n, -np.inf)
	backlink = np.full(n, -1, dtype = int)
	cum_score[0] = onset[0]

	for i in range(1, n):
		period = period_at_frame(i)

		if period is None or period <= 1:
			cum_score[i] = onset[i]
			continue

		tau_lo = max(0, i - int(round(2 * period)))
		tau_hi = max(0, i - max(1, int(round(period / 2))))

		if tau_hi < tau_lo:
			cum_score[i] = onset[i]
			continue

		taus = np.arange(tau_lo, tau_hi + 1)
		deltas = np.maximum((i - taus).astype(np.float64), 1e-6)
		transition_cost = -alpha * (np.log(deltas / period)) ** 2
		candidate_scores = cum_score[taus] + transition_cost

		best_local = int(np.argmax(candidate_scores))
		best_val = candidate_scores[best_local]

		if not np.isfinite(best_val):
			cum_score[i] = onset[i]
			backlink[i] = -1
		else:
			cum_score[i] = onset[i] + best_val
			backlink[i] = taus[best_local]

	end_idx = int(np.argmax(cum_score))
	beats = [end_idx]

	while backlink[beats[-1]] >= 0:
		beats.append(int(backlink[beats[-1]]))

	beats.reverse()
	return beats

# -=-=-=- #
## Mode 0
# Quick / loop estimation
# -=-=-=- #
def _loop_tempo(
	sig: np.ndarray,
	sr: int,

	bpm_min: float = 40.0,
	bpm_max: float = 220.0,

	preferred_bpm: float = 120.0,
	preferred_sigma_octaves: float = 1.2,

	beats_per_bar: int = 4,
	max_bars: int = 64,
) -> float:
	"""
	Estimate tempo for a short, pre-edited loop.

	Loops (samples, one-shots-turned-loops, DAW export selections, ...) are
	almost always cut to an exact number of beats/bars, so duration alone
	is strong evidence:
	- bpm = beats * 60 / duration.
	
	It is tried to try every plausible bar count, to score each candidate with the
	octave-bias prior (same idea as mode 1) plus a mild preference for "round"
	bar counts, and to use a short-window onset-autocorrelation estimate
	purely as a tie-breaker when candidates are close in score
	"""
	duration = len(sig) / sr
	if duration <= 0:
		raise ValueError("Audio is empty; cannot estimate tempo.")

	# candidate bar counts (typical loop lengths).
	bar_counts = [b for b in (1, 2, 4, 8, 16, 32, 64) if b <= max_bars]

	candidates = []
	for bars in bar_counts:
		beats = bars * beats_per_bar
		bpm = beats * 60.0 / duration

		if bpm_min <= bpm <= bpm_max:
			octave_dist = math.log2(bpm / preferred_bpm)
			prior = math.exp(-(octave_dist ** 2) / (2 * preferred_sigma_octaves ** 2))

			# mild bias toward shorter/rounder loops (1-8 bars are far more common than 64-bar "loops")
			length_bias = 1.0 / math.log2(bars + 2)
			candidates.append((bpm, prior * length_bias, bars))

	if not candidates:
		# duration doesn't line up with any plausible bar count (unusual length)
		# fall back to whatever onset periodicity found
		n_fft, hop_length = _adaptive_stft_params(len(sig))
		env, fps, _ = _onset_envelope(sig, sr, n_fft = n_fft, hop_length = hop_length)

		bpm = _tempo_from_envelope(
			env,
			fps,
			bpm_min = bpm_min,
			bpm_max = bpm_max,
			preferred_bpm = preferred_bpm,
			preferred_sigma_octaves = preferred_sigma_octaves,
		)

		if bpm is None:
			raise ValueError(
				"Could not estimate a tempo for this loop; duration doesn't "
				"match a plausible bar count and no clear onset periodicity "
				"was found."
			)

		return bpm

	# onset-based estimate, used only to break ties between close candidates
	n_fft, hop_length = _adaptive_stft_params(len(sig))
	env, fps, _ = _onset_envelope(sig, sr, n_fft = n_fft, hop_length = hop_length)

	onset_bpm = _tempo_from_envelope(
		env, fps, bpm_min = bpm_min, bpm_max = bpm_max,
		preferred_bpm = preferred_bpm,
		preferred_sigma_octaves = preferred_sigma_octaves,
	)

	if onset_bpm is not None:
		# re-score candidates, rewarding ones close to the onset estimate
		rescored = []

		for bpm, score, bars in candidates:
			closeness = math.exp(-((math.log2(bpm / onset_bpm)) ** 2) / (2 * 0.15 ** 2))
			rescored.append((bpm, score * (0.5 + closeness), bars))

		candidates = rescored

	candidates.sort(key = lambda c: c[1], reverse = True)
	return candidates[0][0]

# -=-=-=- #
## Mode 2
# Variable tempo (windowed tempogram)
# -=-=-=- #
def _variable_tempo(
	sig: np.ndarray,
	sr: int,

	hop_length: int = 512,
	window_seconds: float = 6.0,
	step_seconds: float = 3.0,

	bpm_min: float = 40.0,
	bpm_max: float = 220.0,

	preferred_bpm: float = 120.0,
) -> tuple[float, np.ndarray, np.ndarray]:
	"""
	Estimate a time-varying tempo curve by running the mode-1 estimator on
	a sliding window of the onset envelope.

	Returns (overall_bpm, window_center_times, window_bpms).
	"""
	n_fft = 2048
	env, fps, _ = _onset_envelope(sig, sr, n_fft = n_fft, hop_length = hop_length)

	duration = len(sig) / sr

	# if the clip is shorter than the requested window, shrink the window
	# (and step) to fit - a fixed 6s window is meaningless on a 5s clip.
	# it still needs enough frames for `_tempo_from_envelope`'s lag search to
	# resolve `bpm_min`, so floor at whatever that requires.
	min_window_seconds = 60.0 / bpm_min * 2.5 # a few periods of the slowest tempo it searches for

	effective_window_seconds = min(window_seconds, max(min_window_seconds, duration * 0.9))
	effective_step_seconds = min(step_seconds, effective_window_seconds / 2)

	window_frames = max(8, int(round(effective_window_seconds * fps)))
	step_frames = max(1, int(round(effective_step_seconds * fps)))

	if window_frames > len(env):
		raise ValueError(
			f"Clip is too short ({duration:.3f}s) for variable-tempo analysis "
			f"even after shrinking the window; try mode 0 or 1 instead."
		)

	times = []
	bpms = []

	start = 0
	while start + window_frames <= len(env):
		chunk = env[start: start + window_frames]
		bpm = _tempo_from_envelope(
			chunk,
			fps,
			bpm_min = bpm_min,
			bpm_max = bpm_max,
			preferred_bpm = preferred_bpm
		)

		if bpm is not None:
			center_time = (start + window_frames / 2) / fps
			times.append(center_time)
			bpms.append(bpm)

		start += step_frames

	if not bpms:
		raise ValueError(
			"Could not estimate a variable tempo curve; audio may be too "
			"short or too quiet."
		)

	times = np.asarray(times)
	bpms = np.asarray(bpms)

	# de-octave the curve
	# pull any window that drifted to approx 2x or 0.5x of the
	# running median back in line, since real tempo rarely actually
	# doubles for one window and then jumps back.
	running_median = np.median(bpms)
	fixed = bpms.copy()

	for i, bpm in enumerate(bpms):
		for factor in (2.0, 0.5):
			if (
				abs(math.log2((bpm * factor) / running_median)) \
				< \
				abs(math.log2(bpm / running_median))
			):
				fixed[i] = bpm * factor

	bpms = fixed

	overall_bpm = float(np.median(bpms))
	return overall_bpm, times, bpms

# -=-=-=- #
# Public API
# -=-=-=- #

THRESHOLD_DURATION_SECONDS_LOOP = 8.0
THRESHOLD_TEMPO_VARIABLE_CV = 0.04 # coefficient of variation

def detect(
	sig: Union[str, np.ndarray],
	mode: int = 3,
	sr: Optional[int] = None,
	hop_length: int = 512,
	return_curve: bool = False,
	return_beats: bool = False,
) -> Union[
	float,
	tuple[float, tuple[int, ...]],
	tuple[float, np.ndarray, np.ndarray],
	tuple[float, tuple[int, ...], np.ndarray, np.ndarray],
]:
	"""
	Estimate the tempo (BPM) of a piece of audio, and
	optionally the exact sample positions of each detected beat.

	Parameters
	----------
	sig : str | np.ndarray
		Either a path to an audio file readable by SoundFile
		or an in-memory numpy array of samples (mono or multi-channel).
	mode : int, default 3
		Detection mode:
		0 - Quick estimation for short, pre-edited loops.
		1 - Constant-tempo detection (whole-track autocorrelation).
		2 - Variable-tempo detection (windowed tempogram, returns the
		    median tempo unless return_curve = True).
		3 - Universal: auto-picks 0, 1, or 2 based on duration and how
		    much local tempo estimates agree with each other.
	sr : int, optional
		Sample rate of `sig`. Required when `sig` is a numpy array.
		Ignored (read from the file) when `sig` is a file path.
	hop_length : int, default 512
		STFT hop size used for the onset-strength envelope in modes 1/2/3.
		Mode 0 picks its own hop size adapted to the loop's length.
	return_curve : bool, default False
		Only affects mode 2 (directly, or via mode 3 resolving to it)
		If True, includes (window_times, window_bpms) in the return value.
	return_beats : bool, default False
		If True, also runs a dynamic-programming beat tracker against the
		estimated tempo (or tempo curve, for mode 2) and returns the exact
		sample index of every detected beat.

		Beat 0 is the first detected beat, not necessarily sample 0.

	Returns
	-------
	- float                                       # (default)
	- (float, (int, int, ...))                    # return_beats = True
	- (float, np.ndarray, np.ndarray)             # return_curve = True (mode 2 only)
	- (float, (int, ...), np.ndarray, np.ndarray) # both flags, mode 2 only
	"""
	sig, sample_rate = _load_audio(sig, sr)
	duration = len(sig) / sample_rate

	if mode == 3:
		if duration < THRESHOLD_DURATION_SECONDS_LOOP:
			mode = 0
		else:
			# peek at local tempo agreement to decide steady vs. variable
			try:
				_, _, probe_bpms = _variable_tempo(sig, sample_rate, hop_length = hop_length)
				cv = float(np.std(probe_bpms) / np.mean(probe_bpms))
			except ValueError:
				cv = 0.0 # not enough data to tell -> assume steady

			mode = 2 if cv > THRESHOLD_TEMPO_VARIABLE_CV else 1

	if mode == 0:
		bpm = _loop_tempo(sig, sample_rate)

		if not return_beats:
			return bpm

		n_fft, adaptive_hop = _adaptive_stft_params(len(sig))
		env, fps, frame_times = _onset_envelope(
			sig,
			sample_rate,
			n_fft = n_fft,
			hop_length = adaptive_hop
		)

		period_frames = fps * 60.0 / bpm
		beat_frames = _dp_beat_track(env, lambda i: period_frames)
		beat_samples = _beats_to_samples(beat_frames, frame_times, sample_rate)

		return bpm, beat_samples

	elif mode == 1:
		if len(sig) < hop_length * 8:
			raise ValueError("Audio is too short to reliably estimate a tempo.")

		env, frames_per_second, frame_times = _onset_envelope(sig, sample_rate, hop_length = hop_length)
		bpm = _tempo_from_envelope(env, frames_per_second)

		if bpm is None:
			raise ValueError("Could not estimate a tempo for this audio.")

		if not return_beats:
			return bpm

		period_frames = frames_per_second * 60.0 / bpm
		beat_frames = _dp_beat_track(env, lambda i: period_frames)
		beat_samples = _beats_to_samples(beat_frames, frame_times, sample_rate)

		return bpm, beat_samples

	elif mode == 2:
		overall_bpm, times, curve = _variable_tempo(sig, sample_rate, hop_length = hop_length)

		if not return_beats:
			if return_curve:
				return overall_bpm, times, curve

			return overall_bpm

		env, fps, frame_times = _onset_envelope(sig, sample_rate, hop_length = hop_length)

		def _period_at_frame(
			i: int,
			_fps = fps,
			_times = times,
			_curve = curve
		) -> float:
			t = i / _fps
			bpm_local = float(np.interp(t, _times, _curve))

			return _fps * 60.0 / bpm_local

		beat_frames = _dp_beat_track(env, _period_at_frame)
		beat_samples = _beats_to_samples(beat_frames, frame_times, sample_rate)

		if return_curve:
			return overall_bpm, beat_samples, times, curve

		return overall_bpm, beat_samples
	else:
		raise ValueError(f"Unknown mode: {mode!r} (expected 0, 1, 2, or 3)")

#-=-=-=-#

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(
		description = "Estimate the BPM of a file."
	)

	parser.add_argument(
		"-i", "--filename",
		required = True,
		help = "Sound file to analyze"
	)

	parser.add_argument(
		"-m", "--mode",
		type = int,
		default = 3,
		help = "0 = loop, 1 = constant tempo, 2 = variable tempo, 3 = universal"
	)

	parser.add_argument(
		"-c", "--curve",
		action = "store_true",
		help = "Output the full tempo-over-time curve (mode 2 only)"
	)

	parser.add_argument(
		"-b", "--beats",
		action = "store_true",
		help = "Detect and print beat sample positions"
	)

	args = parser.parse_args()

	result = detect(
		args.filename,
		mode = args.mode,
		return_curve = args.curve,
		return_beats = args.beats
	)

	if isinstance(result, tuple):
		bpm = result[0]
		rest = result[1:]
		print(f"Overall estimated tempo: {bpm:.3f} BPM")

		if args.beats:
			beat_samples = rest[0]
			rest = rest[1:]

			print(f"Detected {len(beat_samples)} beats (sample indices):")
			print(" ", beat_samples)

		if args.curve and rest:
			times, curve = rest

			for t, b in zip(times, curve):
				print(f"  t={t:6.3f}s  bpm={b:7.3f}")
	else:
		print(f"Estimated tempo: {result:.3f} BPM")