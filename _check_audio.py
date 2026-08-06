"""Check if video files have audio streams and test ffmpeg merge"""
import os, sys, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import imageio_ffmpeg
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

# Check the latest video
video = 'photo/videos/video_txt2video_20260731_222319_seed1515986539.mp4'
print(f'Video exists: {os.path.exists(video)}')

# FFprobe
r = subprocess.run([ffmpeg, '-i', video], capture_output=True, text=True)
print('\n--- FFprobe ---')
for line in r.stderr.splitlines():
    if any(w in line for w in ['Stream', 'Audio', 'Video', 'Duration']):
        print(line.strip())

# Check for wav files (TTS output)
import tempfile
print(f'\nTemp dir: {tempfile.gettempdir()}')
# Check for recent wav files
wav_dir = tempfile.gettempdir()
for f in os.listdir(wav_dir):
    if f.endswith('.wav') and f.startswith('tmp'):
        path = os.path.join(wav_dir, f)
        age = os.path.getmtime(path)
        size = os.path.getsize(path)
        print(f'  {f}: {size} bytes')
        if size > 0:
            print(f'    Non-empty wav found! Testing merge...')
            out = video.replace('.mp4', '_testvoiced.mp4')
            cmd = [ffmpeg, '-y', '-i', video, '-i', path,
                   '-map', '0:v:0', '-map', '1:a:0',
                   '-c:v', 'copy', '-shortest', out]
            r2 = subprocess.run(cmd, capture_output=True, text=True)
            if r2.returncode == 0:
                print(f'    MERGE OK -> {out}')
            else:
                print(f'    MERGE FAILED (code={r2.returncode}): {r2.stderr[:500]}')
            break  # Only test one