<img src="https://raw.githubusercontent.com/kubinka0505/wavemarks/refs/heads/main/docs/img/logo.svg" width=275>

<a href="https://github.com/kubinka0505/wavemarks/commit"><img src="https://custom-icon-badges.demolab.com/github/last-commit/kubinka0505/wavemarks?logo=commit&style=for-the-badge&cacheSeconds=60" alt="Last commit date"></a>　<a href="https://github.com/kubinka0505/wavemarks/blob/main/License.txt"><img src="https://custom-icon-badges.demolab.com/github/license/kubinka0505/wavemarks?logo=law&color=red&style=for-the-badge&cacheSeconds=60" alt="View license"></a>　<a href="https://app.codacy.com/gh/kubinka0505/wavemarks"><img src="https://img.shields.io/codacy/grade/979ef5b104c94c2ca42cc15210f4d8cf?logo=codacy&style=for-the-badge&cacheSeconds=60" alt="View grade"></a>　<a href="https://app.codacy.com/gh/kubinka0505/wavemarks/coverage"><img src="https://img.shields.io/codacy/coverage/979ef5b104c94c2ca42cc15210f4d8cf?logo=codacy&style=for-the-badge&cacheSeconds=60"></a><!--a href="https://pypi.org/project/wavemarks"><img src="https://img.shields.io/pypi/v/wavemarks?logo=pypi&logoColor=white&style=for-the-badge" alt="PyPI version"></a-->

## Description 📝
Library for managing marker/region-related objects in audio files. 🚩

### Features ✨
- Read, write, delete and rename `CUE` markers, regions and `SMPL` loops
- Modify `ACID` fields such as tempo, root note, time signature and more
- Cross-file marker copying
- Region extraction to separate files
- Marker to region conversion
- Immutable API — every mutation returns a new main file class object

### Marker types 🏷️

| Constant | [FourCC](https://wikipedia.org/wiki/FourCC) |
|:-|:-|
| 🟦 `MarkerType.BASIC` | `rgn ` |
| 🟨 `MarkerType.BEAT` | `beat` |
| 🟩 `MarkerType.DOWNBEAT` | `dwnb` |
| 🟪 `MarkerType.CD_TRACK` | `trak` |
| 🟦 `MarkerType.CD_INDEX` | `indx` |
| 🟩 `MarkerType.SINGLE_CYCLE` | `wtsc` |

### Roadmap 🏁
- [ ] Bug fixing 🐛
- [ ] [WavPack](https://wikipedia.org/wiki/WavPack) support ⚙️

## Installation 🖥️
1. [`git`](https://git-scm.com) (recommended)
```bash
git clone https://github.com/kubinka0505/wavemarks
cd wavemarks
python -m pip install -e .
```

2. [`pip`](https://pypi.org/project/pip)
```bash
pip install git+https://github.com/kubinka0505/wavemarks -U
```

## Usage 📝

### Add or remove marker/region ✏️

```python
>>> from wavemarks import MarkerFile, MarkerType, Entry
>>> 
>>> # Open file
>>> file_obj = MarkerFile("audio.wav")
>>> 
>>> # Marker named "important" at sample 2801 with type SINGLE_CYCLE
>>> # By adding "end" argument a region is created
>>> my_marker = Entry(name = "important", start = 2801, type = MarkerType.SINGLE_CYCLE)
>>> 
>>> # Add it
>>> file_obj += my_marker
>>> 
>>> # Check if exists
>>> my_marker in file_obj
True
>>> 
>>> # Then remove it (must be equivalent!)
>>> file_obj -= my_marker
>>> 
>>> my_marker in file_obj
False
>>> 
>>> # Save to output
>>> file_obj.save("audio_modified.wav")
```

### Rename marker 🔄

```python
>>> from wavemarks import MarkerFile, Entry, Note
>>> 
>>> # Open file
>>> file_obj = MarkerFile("Loop.wav")
>>> 
>>> # Initialize variables
>>> name_old = "old"
>>> name_new = "new"
>>> 
>>> # Find target region
>>> target = [m for m in file_obj.regions if m.name == name_old][0]
>>> 
>>> # Rename
>>> file_obj = file_obj.rename_cue(target, name_new)
>>> 
>>> # Save to input file (no `path` argument), and optionally
>>> # detect tempo and write initial key to A-flat 4th octave (read help(save))
>>> file_obj.save(tempo = -2, note_root = Note.Ab4)
```

### `CUE` inspection 🧐

```python
>>> from wavemarks import MarkerFile, MarkerType
>>> 
>>> # Open file
>>> file_obj = MarkerFile("input.wav")
>>> 
>>> # File has 1 marker named "intro" at sample 0 with type RegionType.BASIC
>>> file_obj.markers
[Entry(name='intro', type='rgn ', start=0, end=None)]
>>> 
>>> # File has no regions (as end = None dictates marker)
>>> file_obj.regions
[]
>>> # ...but the file is NOT empty, since it has that 1 marker
>>> file_obj.is_empty()
False
```

### `ACID` inspection 🔬

```python
>>> from wavemarks import MarkerFile, MarkerType
>>> 
>>> # Open file and pray that someone qualified managed its metadata
>>> file_obj = MarkerFile("LPS_LOOP_01.wav")
>>> 
>>> print(f"File tempo is: {file_obj.tempo:.3f} BPM")
File tempo is: 128.000 BPM
>>> 
>>> print(f"File time signature is:", "/".join(file_obj.time_signature))
File time signature is: 4/4
```

### Search for objects 🔍

```python
>>> from wavemarks import MarkerFile, Entry
>>> 
>>> # Open file
>>> file_obj = MarkerFile("../snd.wav")
>>> 
>>> # Search for all markers named "abracadabra" with CD_INDEX type
>>> mks = []
>>> 
>>> for m in file_obj.markers:
>>>     if m.name == "abracadabra" and m.type == RegionType.CD_INDEX:
>>>         mks.append(m)
>>> 
>>> # Search for all regions between sample 900 and 1601
>>> rgs = []
>>> 
>>> for r in file_obj.regions:
>>>     if r.start >= 900 and r.end <= 1601:
>>>         rgs.append(r)
```

### Advanced utilities 🛠️

```python
>>> from wavemarks import MarkerFile, MarkerType
>>> 
>>> # Open file
>>> file_obj = MarkerFile("input.wav")
>>> 
>>> # Get all entries
>>> list(file_obj)
[Entry(name='', type='rgn ', start=22935, end=None),
 Entry(name='drop', type='rgn ', start=89408, end=None)]
>>> 
>>> # Convert consecutive point markers into regions
>>> file_obj = file_obj.markers_to_regions()
>>> file_obj.regions
[Entry(name='drop', type='rgn ', start=22935, end=89408)]
>>> 
>>> # Extract audio regions to separate wave files
>>> #
>>> # Basename is:
>>> # For named regions: `{region.name}`
>>> # Otherwise: `region_{marker.start}_{marker.end}`
>>> file_obj = MarkerFile("input.wav")
>>> 
>>> for region in file_obj.regions:
>>>     file_obj.export(region, "existing_directory/") 
>>> 
>>> # Copy all entries from one file to another
>>> MarkerFile("input.wav").copy("output.wav")
```

### Copying and context manager 📋

```python
>>> from wavemarks import MarkerFile, MarkerType, Entry
>>> 
>>> # Initialize variables
>>> file_path = "input.wav"
>>> 
>>> # Context manager
>>> with MarkerFile(file_path) as f:
...     # marker
...     f += Entry(start = 101, name = "example point", type = MarkerType.BASIC)
...     # region
...     f += Entry(start = 2000, end = 4000, "example region", type = MarkerType.CD_TRACK)
...     f.save("output.wav")
>>>
>>> # Reapply markers after external processing
>>> snapshot = MarkerFile(file_path).markers
>>>
>>> # normalize_input_audio(file_path)
>>> 
>>> # Reload and save
>>> MarkerFile(file_path).apply(snapshot).save()
```

## Disclaimer ⚠️
This software:
- Operates on WAV (`RIFF/WAVE`) files only.
- Does not support BWF (`bext`), iXML, or other metadata chunks — they are preserved byte-for-byte but not parsed.
- Round-trips all non-marker chunks without modification; however, malformed or non-standard RIFF files may not parse correctly.