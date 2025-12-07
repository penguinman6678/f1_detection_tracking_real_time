import cv2 
import os


"""
Extracting frames from videos found in ./raw_videos
"""

def extract_frames(video_path, output_dir="./frames/", fps=15):

    # Check if the directories exist --> otherwise create one
    if os.path.exists(output_dir) == True:
        print(f"Output directory exists: {output_dir}")
    else:
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    captured_video = cv2.VideoCapture(video_path)
    orig_video_fps = captured_video.get(cv2.CAP_PROP_FPS)

    frame_interval = int(round(orig_video_fps / fps))
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = captured_video.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            path = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_{saved_idx:08d}.jpg")
            cv2.imwrite(path, frame)
            saved_idx += 1
        frame_idx += 1
    
    captured_video.release()


if __name__ == "__main__":

    video_dir = "./raw_videos/raw_videos"
    frames_dir = "./raw_videos/frames"

    for video in os.listdir(video_dir):
        if video.endswith(".mp4"):
            extract_frames(video_path=os.path.join(video_dir, video), 
                            output_dir=os.path.join(frames_dir, os.path.splitext(video)[0]),
                            fps=15) 
    
