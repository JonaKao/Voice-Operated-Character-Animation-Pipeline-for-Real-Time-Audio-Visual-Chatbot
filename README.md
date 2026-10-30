# Voice-Operated Character Animation Pipeline for a Real-Time Audio-Visual Chatbot

This project turns a chatbot's spoken reply into a talking 3D face. When an MP3 file of speech lands in a watched folder, the pipeline animates a FLAME head mesh in sync with the audio. It renders the result to an MP4 video with sound and serves that video through a small web page.

The animation itself comes from **[VOCA](https://github.com/TimoBolkart/voca)** (Voice Operated Character Animation, Cudeiro et al., CVPR 2019). VOCA maps speech features from a DeepSpeech model to per-frame vertex displacements of a FLAME template mesh. This repository adds the automation around it: folder watching, audio conversion, frame rendering, video assembly and a web viewer. It also contains a few modifications to VOCA's own files.

> **This repository is not standalone.** It contains modified VOCA files plus new scripts, and expects to be dropped into a working VOCA checkout. See [Installation](#installation).

## How it works

```
 chatbot reply (text)
        │  generate_mp3.py  (gTTS, stand-in for the chatbot's TTS)
        ▼
 audio_in/<name>.mp3 ──► auto_pipeline.py (watchdog)
                              │
                              ├─ 1. ffmpeg: MP3 → mono 22.05 kHz 16-bit WAV   (audio/<name>.wav)
                              ├─ 2. run_voca.py: DeepSpeech features → VOCA → per-frame OBJ meshes
                              │                                                (auto_output/<name>/meshes/)
                              ├─ 3. compute_FLAME_params.py: fit FLAME shape/jaw/expression params
                              │                                                (data/params_<name>.npy)
                              ├─ 4. render_frames_open3d.py: offscreen render, 800×800 PNGs (frames/)
                              ├─ 5. ffmpeg: frames @ 60 fps + WAV → H.264/AAC MP4 (<name>.mp4)
                              └─ 6. copy into videos/ and open the video
                                                       │
                                     app.py (Flask) ◄──┘  lists and streams videos/*.mp4 on port 5000
```

## Repository contents

**New files**

| File | Purpose |
|---|---|
| `auto_pipeline.py` | Main entry point. Watches `audio_in/` for new `.mp3` files and runs the full chain for each one |
| `app.py` | Flask web app that lists the MP4 files in `videos/` and streams them (`0.0.0.0:5000`) |
| `generate_mp3.py` | Creates a test MP3 from text with gTTS and writes it into the watch folder |
| `render_frames_open3d.py` | Renders a mesh sequence to PNG frames with Open3D's offscreen renderer (used by the pipeline) |
| `render_frames.py` | Alternative frame renderer using trimesh's `scene.save_image` |
| `test_Tensor.py` | Checks whether TensorFlow can see a GPU |

**Modified VOCA files**

| File | VOCA location | Change |
|---|---|---|
| `inference.py` | `utils/` | Loads the template mesh and exports meshes with `trimesh` instead of `psbody.mesh`, and restores the model from its `.meta` graph |
| `audio_handler.py` | `utils/` | Runs DeepSpeech feature extraction on the CPU only, which avoids GPU memory clashes with VOCA |
| `compute_FLAME_params.py` | root | Reads OBJ and PLY meshes with `trimesh` instead of `psbody.mesh` |

`audio_handler copy.py` is the unmodified original of `audio_handler.py`, kept as a backup. The remaining `.py` files (`run_voca.py`, `voca_model.py`, `rendering.py`, `batcher.py`, …) are unchanged copies of VOCA's code.

## Installation

### 1. Set up VOCA

Follow the [VOCA setup instructions](https://github.com/TimoBolkart/voca#set-up).

VOCA is TensorFlow 1.x code: it uses `tf.Session` and `tf.placeholder`, and its requirements pin `tensorflow-gpu==1.15.2`. Use a Python 3.6 or 3.7 virtual environment.

Download the following assets, which need registration and come under their own licences:
- the trained VOCA model and template meshes from [voca.is.tue.mpg.de](https://voca.is.tue.mpg.de) → `model/gstep_52280.model*`, `template/FLAME_sample.ply`
- the FLAME model from [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de) → `flame/generic_model.pkl`
- the DeepSpeech **v0.1.0** model from [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech/releases/tag/v0.1.0) → `ds_graph/output_graph.pb`

### 2. Add this repository's files

Copy the files into your VOCA checkout:

```
voca/
├── auto_pipeline.py          ← new
├── app.py                    ← new
├── generate_mp3.py           ← new
├── render_frames_open3d.py   ← new
├── render_frames.py          ← new
├── compute_FLAME_params.py   ← replace
└── utils/
    ├── inference.py          ← replace
    └── audio_handler.py      ← replace
```

### 3. Install the extra dependencies

```bash
pip install watchdog flask gTTS open3d
```

Choose an Open3D version that includes `OffscreenRenderer` (0.11 or later) and still supports your Python version. **ffmpeg** must be on your `PATH`.

### 4. Create the working folders

```bash
mkdir audio_in audio auto_output data videos frames templates
```

## Usage

**Start the watcher** from the VOCA root:

```bash
python auto_pipeline.py
# Watching .../audio_in for new MP3 files…
```

**Feed it audio.** Either drop any speech MP3 into `audio_in/`, or edit the text and `output_dir` in `generate_mp3.py` and run it:

```bash
python generate_mp3.py
```

The pipeline settings are constants at the top of `auto_pipeline.py`:
- `MODEL_FILE`, `DS_GRAPH`, `TEMPLATE`: paths to the downloaded assets
- `CONDITION`: VOCA's speaking-style subject index, from 1 to 8

**View the results** in a browser:

```bash
python app.py
# open http://localhost:5000 (or http://<your-ip>:5000 from another device on the LAN)
```

`app.py` renders `templates/index.html`, which is not included in the repository. A minimal version:

```html
<!doctype html>
<title>VOCA videos</title>
{% for v in videos %}
  <h3>{{ v }}</h3>
  <video src="{{ url_for('serve_video', filename=v) }}" controls width="480"></video>
{% endfor %}
```

**Run VOCA on its own** (without the watcher):

```bash
python run_voca.py --audio_fname ./audio/test_sentence.wav --out_path ./animation_output --visualize False
```

## Known issues

- **Old frames are not cleared.** `frames/` is never emptied between runs, so a shorter clip can pick up leftover frames from a longer one. (`-shortest` cuts the video to the audio length, but this only partly hides the problem.)
- **Parts of the code are Windows-only.** `os.startfile`, the `shell=True` list call to ffmpeg, and the hard-coded `C:\voca_project\...` path in `generate_mp3.py` only work on Windows. On Linux or macOS, replace `os.startfile` (for example with `xdg-open` or `open`) and drop `shell=True`.
- **Keep `--visualize False`.** The modified `utils/inference.py` calls `render_mesh_helper` without importing it, so the built-in pyrender visualisation fails. The pipeline already passes `--visualize False`.
- **The FLAME parameters are not used yet.** `compute_FLAME_params.py` (step 3) writes FLAME parameters that no later step reads. It adds runtime, and can be skipped if you don't need the parameters.
- **The copy into `videos/` assumes the folder exists.** It is created only when `app.py` runs, or when you create it yourself as in step 4.
- **The pipeline is not yet real-time.** Each clip runs as a sequence of separate processes (DeepSpeech, VOCA, FLAME fitting, rendering, encoding). Latency is seconds to minutes per sentence, depending on clip length and hardware.

## Licence and credits

Most of the code here comes from **VOCA** and is © Max-Planck-Gesellschaft. VOCA's licence allows **non-commercial scientific research use only**; see [voca.is.tue.mpg.de](https://voca.is.tue.mpg.de) for the terms. The FLAME model and DeepSpeech have their own licences. Make sure your use complies with all three.

If you use VOCA, please cite:

```bibtex
@inproceedings{VOCA2019,
  title     = {Capture, Learning, and Synthesis of {3D} Speaking Styles},
  author    = {Cudeiro, Daniel and Bolkart, Timo and Laidlaw, Cassidy and Ranjan, Anurag and Black, Michael},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {10101--10111},
  year      = {2019}
}
```
