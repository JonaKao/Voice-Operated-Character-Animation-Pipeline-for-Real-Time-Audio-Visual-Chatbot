import os
from flask import Flask, render_template, send_from_directory

app = Flask(__name__)
VIDEO_DIR = os.path.join(app.root_path, 'videos')

@app.route('/')
def index():
    # list all .mp4 files in /videos
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    return render_template('index.html', videos=videos)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    # serve the raw MP4 file
    return send_from_directory(VIDEO_DIR, filename)

if __name__ == '__main__':
    # make sure the videos folder exists
    os.makedirs(VIDEO_DIR, exist_ok=True)
    # launch on port 5000, visible on your LAN
    app.run(host='0.0.0.0', port=5000, debug=True)
