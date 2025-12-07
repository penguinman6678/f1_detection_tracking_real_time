import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from run_vit import load_vit_model
from PIL import Image


def track_with_deepsort(model_path=None, vit_model_path=None, video_path=None, output_path=None, image_size=640, conf_threshold=0.4, team_conf_threshold=0.0):

    if model_path == None or video_path == None or output_path == None or vit_model_path==None:
        print(f"Something is missing...")
        return
    
    # yolov8 model --> current one is yolov8s for quicker implementation
    yolo_model = YOLO(model_path)

    captured_video = cv2.VideoCapture(video_path)
    fps = captured_video.get(cv2.CAP_PROP_FPS)
    cvw = int(captured_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    cvh = int(captured_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (cvw, cvh))

    # Initialize DeepSort Tracker
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.3,
        nms_max_overlap=1.0,
        embedder="mobilenet",
        half=True,
        bgr=True
    )

    # Initialize VIT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model = load_vit_model(vit_model_path, device)

    frame_idx = 0
    tid_team_name = {}
    tid_team_conf = {}

    while True:
        
        ret, frame = captured_video.read()
        if ret == False:
            break
        
        results = yolo_model.predict(
            source=frame,
            conf=conf_threshold,
            imgsz=image_size,
            device=0, # remember to change if not running cuda
            verbose=False
        )

        # Retrieve result + bounding boxes to input for the Byte Track algo
        result = results[0]
        bboxes = result.boxes

        detections = []
        if len(bboxes) == 0:
            writer.write(frame)
            frame_idx += 1
            continue

        if bboxes is not None and len(bboxes) > 0:
        
            bbox_coords = bboxes.xyxy.cpu().numpy()
            conf_values = bboxes.conf.cpu().numpy()
            cls_idxes = bboxes.cls.cpu().numpy().astype(int)

            for idx in range(len(bbox_coords)):
                
                x1, y1, x2, y2 = bbox_coords[idx]
                conf_val = conf_values[idx]
                cls_idx = cls_idxes[idx]

                if conf_val < conf_threshold: # Sanity check
                    continue
                
                detections.append([[x1, y1, x2, y2], float(conf_val), yolo_model.names[int(cls_idx)]])
        
        # update tracking status of DeepSort
        currently_tracked = tracker.update_tracks(detections, frame=frame)

        # Update frame with bbox and labels
        for track in currently_tracked:
            if track.is_confirmed() == False:
                continue
            
            tid = track.track_id

            # want to get last YOLO box, not the kahman modified box
            bbox_coords_tracked = track.to_tlwh(orig=True, orig_strict=True)
            if bbox_coords_tracked is not None:
                x1, y1 = bbox_coords_tracked[0], bbox_coords_tracked[1]
                x2, y2 = bbox_coords_tracked[2], bbox_coords_tracked[3]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            else:
                continue
                # bbox_coords_tracked = track.to_ltrb()
                # x1, y1, x2, y2 = map(int, bbox_coords_tracked) # Reverse engineer position of bbox w/ kahman

            x1_cropped = max(0, x1)
            y1_cropped = max(0, y1)
            x2_cropped = min(cvw - 1, x2)
            y2_cropped = min(cvh - 1, y2)

            cropped_img = frame[y1_cropped: y2_cropped, x1_cropped: x2_cropped]

            team_name, pred_conf = predict_vit_model_team(cropped_img, vit_model, device, conf_threshold=0.0)
            if team_name is not None and pred_conf >= team_conf_threshold:
                tid_team_name[tid] = team_name
                tid_team_conf[tid] = pred_conf

            if track.get_det_class() is not None:
                cls_name = track.get_det_class()

            team_predicted_label = f"ID {tid} {cls_name} {tid_team_name.get(tid, 'Unknown')} with " \
                                    f"{pred_conf:.3f}" if tid in tid_team_conf else f"ID {tid} {cls_name} {tid_team_name.get(tid, 'Unknown')} with Conf Unk"

            
            
            # label = f"ID {tid}"
            # if len(cls_name) != 0:
            #     label = f"ID {tid} {cls_name}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, team_predicted_label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        writer.write(frame)
        frame_idx += 1
    
    captured_video.release()
    writer.release()

    print(f"DeepSort Algo -- Saved at {output_path}")

def predict_vit_model_team(cropped_img, vit_model, device, conf_threshold=0.6):

    team_names = ["Alpine", "AshtonMartin", "Ferrari", "Haas", "Kick", "McLaren", "Mercedes", "RacingBull", "RedBull", "Williams"]

    vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
    vit_transforms = vit_weights.transforms()

    if cropped_img is None or cropped_img.size == 0:
        return None, confidence
    
    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    cropped_img_rgb = Image.fromarray(cropped_img_rgb)
    cropped_img_rgb = vit_transforms(cropped_img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = vit_model(cropped_img_rgb)
        probs = torch.softmax(predictions, dim=1)[0]
        confidence, cls_idx = torch.max(probs, dim=0)
        confidence = float(confidence.cpu().item())
        cls_idx = int(cls_idx.cpu().item())
    
    if confidence < conf_threshold:
        return None, confidence
    
    team_name = team_names[cls_idx]
    return team_name, confidence


if __name__ == "__main__":

    vid_name = "race13"
    model_path = "./runs/detect/train5/weights/best.pt" # 100 epochs on yolov8s
    video_path = f"../data/raw_videos/raw_videos/{vid_name}.mp4"
    output_path = f"./deepsort_tracked_videos/deepsort_{vid_name}_time_test.mp4" # ending should match with video_path w.r.t race video for consistency's sake
    vit_model_path = "./vit_runs/vit_200.pth"

    track_with_deepsort(model_path=model_path, vit_model_path=vit_model_path, video_path=video_path, output_path=output_path, image_size=640, conf_threshold=0.4)
                
