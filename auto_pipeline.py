import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration
VOCA_DIR    = os.path.abspath(os.path.dirname(__file__))
WATCH_DIR   = os.path.join(VOCA_DIR, 'audio_in')
OUTPUT_ROOT = os.path.join(VOCA_DIR, 'auto_output')
MODEL_FILE  = os.path.join(VOCA_DIR, 'model', 'gstep_52280.model')
DS_GRAPH    = os.path.join(VOCA_DIR, 'ds_graph', 'output_graph.pb')
TEMPLATE    = os.path.join(VOCA_DIR, 'template', 'FLAME_sample.ply')
CONDITION   = '3'

class MP3Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.lower().endswith('.mp3'):
            base = os.path.splitext(os.path.basename(event.src_path))[0]
            print(f"\nDetected new MP3: {base}.mp3")
            # give the file time to fully arrive
            time.sleep(2)

            # 1) Convert to WAV
            wav_path = os.path.join(VOCA_DIR, 'audio', f'{base}.wav')
            subprocess.run([
                'ffmpeg', '-y', '-i', event.src_path,
                '-ac', '1', '-ar', '22050', '-sample_fmt', 's16',
                wav_path
            ], check=True)

            # Prepare output folder
            seq_out = os.path.join(OUTPUT_ROOT, base)
            os.makedirs(os.path.join(seq_out, 'meshes'), exist_ok=True)

            # 2) Run VOCA inference
            subprocess.run([
                'python', 'run_voca.py',
                '--tf_model_fname', MODEL_FILE,
                '--ds_fname', DS_GRAPH,
                '--audio_fname', wav_path,
                '--template_fname', TEMPLATE,
                '--condition_idx', CONDITION,
                '--out_path', seq_out,
                '--visualize', 'False'
            ], cwd=VOCA_DIR, check=True)

            # 3) Compute FLAME parameters
            params_file = os.path.join(VOCA_DIR, 'data', f'params_{base}.npy')
            results_dir = os.path.join(VOCA_DIR, f'results_{base}')
            subprocess.run([
                'python', 'compute_FLAME_params.py',
                '--source_path', os.path.join(seq_out, 'meshes'),
                '--params_fname', params_file,
                '--out_path', results_dir,
                '--flame_model_path', os.path.join(VOCA_DIR, 'flame', 'generic_model.pkl'),
                '--template_fname', TEMPLATE
            ], cwd=VOCA_DIR, check=True)

            # 4) Render frames
            subprocess.run(['python', 'render_frames_open3d.py'], cwd=VOCA_DIR, check=True)

            # 5) Stitch and play
            video_path = os.path.join(VOCA_DIR, f'{base}.mp4')
            subprocess.run([
                'ffmpeg', '-y', '-framerate', '60',
                '-i', os.path.join('frames', '%05d.png'),
                '-i', wav_path,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-shortest', video_path
            ], cwd=VOCA_DIR, check=True, shell=True)

            # Copy to the Flask 'videos' folder so the web UI can serve it
            import shutil
            shutil.copy(
                video_path,
                os.path.join(VOCA_DIR, 'videos', os.path.basename(video_path))
            )


            os.startfile(video_path)
            print(f"Done → {video_path}\n")

if __name__ == "__main__":
    os.makedirs(WATCH_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    observer = Observer()
    observer.schedule(MP3Handler(), WATCH_DIR, recursive=False)
    observer.start()
    print(f"Watching {WATCH_DIR} for new MP3 files…")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
