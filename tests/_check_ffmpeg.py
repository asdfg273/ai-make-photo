"""Check ffmpeg availability and test merge"""
import os, sys, subprocess, shutil
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Check ffmpeg
ffmpeg1 = shutil.which('ffmpeg')
print(f'1. shutil.which("ffmpeg"): {ffmpeg1}')

try:
    import imageio_ffmpeg
    ffmpeg2 = imageio_ffmpeg.get_ffmpeg_exe()
    print(f'2. imageio_ffmpeg.get_ffmpeg_exe(): {ffmpeg2}')
    print(f'   exists: {os.path.exists(ffmpeg2)}')
except Exception as e:
    print(f'2. imageio_ffmpeg failed: {e}')

# Check if any video files exist in output
video_dir = "photo/videos"
if os.path.exists(video_dir):
    files = os.listdir(video_dir)
    mp4s = [f for f in files if f.endswith('.mp4')]
    print(f'\n3. Videos in {video_dir}: {len(mp4s)}')
    for mp4 in mp4s[-3:]:
        path = os.path.join(video_dir, mp4)
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f'   {mp4} ({size_mb:.1f}MB)')