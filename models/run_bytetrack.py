from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from run_vit import load_vit_model
from PIL import Image

"""
Utilize ByteTrack algorithm to perform tracking over multiple frames in a video
At a high level, ByteTrack uses both high and low conf scores to improve tracking + consistency and reducing FNs.
-- high confidence to detect initially and low confidence to find + recover tracks
"""


def track_with_bytetrack(yolo_model_path=None, vit_model_path=None, video_path=None, output_path=None, image_size=640, conf_threshold=0.4, team_conf_threshold=0.0):

    if yolo_model_path == None or video_path == None or output_path == None or vit_model_path == None:
        print(f"Something is missing...")
        return
    
    # yolov8 model --> current one is yolov8s for quicker implementation
    yolo_model = YOLO(yolo_model_path)

    captured_video = cv2.VideoCapture(video_path)
    fps = captured_video.get(cv2.CAP_PROP_FPS)
    cvw = int(captured_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    cvh = int(captured_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (cvw, cvh))

    # Initialize ByteTrack model
    tracker = sv.ByteTrack()

    # Initialize VIT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model = load_vit_model(vit_model_path, device)

    bbox_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()


    tid_team_name = {}
    tid_team_conf = {}

    frame_idx = 0
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

        # Retrieve result + bounding boxes to input for the Byte Track algo
        result = results[0]
        bboxes = result.boxes
        
        # Need to consider when no cars are detected based on threshold (i.e. F1 transitions between laps)
        if len(bboxes) == 0:
            writer.write(frame)
            frame_idx += 1
            continue
        
        bbox_coords = bboxes.xyxy.cpu().numpy()
        conf_values = bboxes.conf.cpu().numpy()
        cls_idxes = bboxes.cls.cpu().numpy().astype(int)

        detections = sv.Detections(
            xyxy=bbox_coords,
            confidence=conf_values,
            class_id=cls_idxes
        )

        # update the tracker with the frame bboxes
        currently_tracked = tracker.update_with_detections(detections)

        # create labels to keep track between frames / scenes
        tracked_labels = []

        ### Keep this if I only want to predict car vs no car
        # for tid, cid in zip(currently_tracked.tracker_id, currently_tracked.class_id):
        #     cls_name = yolo_model.names[int(cid)] # technically should be only one class "car"
        #     tracked_labels.append(f"ID {tid} {cls_name}")

        ### Keep this if I want the actual team name prediction
        for tracked_idx in range(len(currently_tracked)):

            x1,y1,x2,y2 = currently_tracked.xyxy[tracked_idx].astype(int)
            tid = currently_tracked.tracker_id[tracked_idx]
            cid = currently_tracked.class_id[tracked_idx]

            x1_cropped = max(0, x1)
            y1_cropped = max(0, y1)
            x2_cropped = min(cvw - 1, x2)
            y2_cropped = min(cvh - 1, y2)

            cropped_img = frame[y1_cropped: y2_cropped, x1_cropped: x2_cropped]

            team_name, pred_conf = predict_vit_model_team(cropped_img, vit_model, device, conf_threshold=0.0)
            if team_name is not None and pred_conf >= team_conf_threshold:
                tid_team_name[tid] = team_name
                tid_team_conf[tid] = pred_conf
            

            cls_name = yolo_model.names[int(cid)]
            team_predicted_label = f"ID {tid} {cls_name} {tid_team_name.get(tid, 'Unknown')} with " \
                                    f"{pred_conf:.3f}" if tid in tid_team_conf else f"ID {tid} {cls_name} {tid_team_name.get(tid, 'Unknown')} with Conf Unk"
            tracked_labels.append(team_predicted_label)


        
        # update frame with bbox and labels
        annotated_tracked_frame = bbox_annotator.annotate(scene=frame.copy(), 
                                                          detections=currently_tracked
                                                        )
        annotated_tracked_frame = label_annotator.annotate(scene=annotated_tracked_frame.copy(),
                                                           detections=currently_tracked,
                                                           labels=tracked_labels)
        
        # insert frame into the video
        writer.write(annotated_tracked_frame)
        frame_idx += 1
    
    captured_video.release()
    writer.release()
    print(f"Bytetrack Algo -- Saved at {output_path}")

    return


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

def live_stream_track_with_bytetrack(stream_url=None, yolo_model_path=None, vit_model_path=None, output_path=None, image_size=640, conf_threshold=0.4, team_conf_threshold=0.0):

    if yolo_model_path == None or stream_url == None or vit_model_path == None:
        print(f"Something is missing...")
        return
    
    live_stream = cv2.VideoCapture(stream_url)
    if live_stream.isOpened() == False:
        print(f"Cannot open stream at {stream_url}. Double check the url?")
        return
    
    # fps = live_stream.get(cv2.CAP_PROP_FPS)
    # if fps <= 0 or np.isnan(fps):
    #     fps = 30
    fps = 30
    cvw = int(live_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    cvh = int(live_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (cvw, cvh))
    
    # yolov8 model --> current one is yolov8s for quicker implementation
    yolo_model = YOLO(yolo_model_path)

    # Initialize ByteTrack model
    tracker = sv.ByteTrack()

    # Initialize VIT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model = load_vit_model(vit_model_path, device)

    bbox_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    tid_team_name = {}
    tid_team_conf = {}

    frame_idx = 0
    while True:
        ret, frame = live_stream.read()
        if ret == False:
            print("No live stream")
            return
        
        results = yolo_model.predict(
            source=frame,
            conf=conf_threshold,
            imgsz=image_size,
            device=0, 
            verbose=False
        )

        # Retrieve result + bounding boxes to input for the Byte Track algo
        result = results[0]
        bboxes = result.boxes
        
        # Need to consider when no cars are detected based on threshold (i.e. F1 transitions between laps)
        if len(bboxes) == 0:
            cv2.imshow("Live Tracking and Detection", frame)
            writer.write(frame)
            frame_idx += 1
            continue
        bbox_coords = bboxes.xyxy.cpu().numpy()
        conf_values = bboxes.conf.cpu().numpy()
        cls_idxes = bboxes.cls.cpu().numpy().astype(int)

        detections = sv.Detections(
            xyxy=bbox_coords,
            confidence=conf_values,
            class_id=cls_idxes
        )

        # update the tracker with the frame bboxes
        currently_tracked = tracker.update_with_detections(detections)

        # create labels to keep track between frames / scenes
        tracked_labels = []

        ### Keep this if I want the actual team name prediction
        for tracked_idx in range(len(currently_tracked)):

            x1,y1,x2,y2 = currently_tracked.xyxy[tracked_idx].astype(int)
            tid = currently_tracked.tracker_id[tracked_idx]
            cid = currently_tracked.class_id[tracked_idx]

            x1_cropped = max(0, x1)
            y1_cropped = max(0, y1)
            x2_cropped = min(cvw - 1, x2)
            y2_cropped = min(cvh - 1, y2)

            cropped_img = frame[y1_cropped: y2_cropped, x1_cropped: x2_cropped]

            team_name, pred_conf = predict_vit_model_team(cropped_img, vit_model, device, conf_threshold=0.0)
            if team_name is not None and pred_conf >= team_conf_threshold:
                tid_team_name[tid] = team_name
                tid_team_conf[tid] = pred_conf
            

            cls_name = yolo_model.names[int(cid)]
            team_predicted_label = f"ID {tid} {cls_name} {tid_team_name.get(tid, 'Unknown')} with " \
                                    f"{pred_conf:.3f}" if tid in tid_team_conf else f"ID {tid} {cls_name} {tid_team_name.get(tid, 'Unknown')} with Conf Unk"
            tracked_labels.append(team_predicted_label)

        # update frame with bbox and labels
        annotated_tracked_frame = bbox_annotator.annotate(scene=frame.copy(), 
                                                        detections=currently_tracked
                                                        )
        annotated_tracked_frame = label_annotator.annotate(scene=annotated_tracked_frame.copy(),
                                                        detections=currently_tracked,
                                                        labels=tracked_labels)

        # insert frame into the video
        cv2.imshow("Live Tracking and Detection", annotated_tracked_frame)
        writer.write(annotated_tracked_frame)
        frame_idx += 1

        # exit program --> easier than cntrl c when holding the phone i think
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
    
    live_stream.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Bytetrack Algo Live Stream -- Saved at {output_path}")
    return
    
        


if __name__ == "__main__":

    vid_name = "race13"
    model_path = "./runs/detect/train5/weights/best.pt"
    video_path = f"../data/raw_videos/raw_videos/{vid_name}.mp4"
    output_path = f"./bytetrack_tracked_videos/bytetrack_{vid_name}_vit_time_test.mp4" # ending should match with video_path w.r.t race video for consistency's sake
    vit_model_path = "./vit_runs/vit_200.pth"
    stream_url = "http://192.168.1.189:4747/video"
    # counter = 1
    # live_stream_output_path = f"./bytetrack_tracked_videos/bytetrack_livestream_{counter}_vit.mp4"
    track_with_bytetrack(yolo_model_path=model_path, vit_model_path=vit_model_path, video_path=video_path, output_path=output_path, image_size=640, conf_threshold=0.4)
    # live_stream_track_with_bytetrack(stream_url=stream_url, yolo_model_path=model_path, vit_model_path=vit_model_path, output_path=live_stream_output_path, image_size=640, conf_threshold=0.4)