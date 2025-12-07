import os 
import cv2
from ultralytics import YOLO


def extract_cropped_images(model_path=None, video_name=None, video_path=None, output_path=None, image_size=640, conf_threshold=0.4):

    captured_video = cv2.VideoCapture(video_path)
    frame_idx = 0
    yolo_model = YOLO(model_path)

    while True:

        ret, frame = captured_video.read()
        if ret == False:
            break
        
        results = yolo_model.predict(
            source=frame,
            conf=conf_threshold,
            imgsz=image_size,
            device=0,
            verbose=False
        )

        # Retrieve result + bounding boxes
        result = results[0]
        bboxes = result.boxes

        if bboxes is None or len(bboxes) == 0:
            frame_idx +=1 
            continue

        for idx in range(len(bboxes)):

            bbox = bboxes[idx]
            x1, y1, x2, y2 = map(int, bbox.xyxy[0].cpu().numpy())
            cropped_image = frame[y1:y2, x1:x2]
            if cropped_image.size == 0:
                continue
            
            cropped_image_path = os.path.join(output_path, f"{video_name}_frame{frame_idx:08d}_b{idx:02d}.jpg")
            cv2.imwrite(cropped_image_path, cropped_image)
        
    captured_video.release()

    print(f"Check output dir: {output_path}")

    return


if __name__ == "__main__":

    vid_name = "race5"
    model_path = "../models/runs/detect/train5/weights/best.pt" # 100 epochs on yolov8s
    video_path = f"./raw_videos/raw_videos/{vid_name}.mp4"
    output_path = "./f1_teams/classification/" # ending should match with video_path w.r.t race video for consistency's sake
    
    extract_cropped_images(model_path=model_path, video_name=vid_name, video_path=video_path, output_path=output_path, image_size=640, conf_threshold=0.5)

