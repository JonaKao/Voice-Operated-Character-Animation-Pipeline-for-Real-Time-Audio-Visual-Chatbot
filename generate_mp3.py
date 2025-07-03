from gtts import gTTS
import os

# Target directory
output_dir = r"C:\voca_project\voca\audio_in"
os.makedirs(output_dir, exist_ok=True)

text = "that sounds difficult, I hope I can help you with it"
tts = gTTS(text, lang='en', slow=False)

# Full path into your specified folder
file_path = os.path.join(output_dir, "output.mp3")
tts.save(file_path)

print(f"Generated {file_path}")