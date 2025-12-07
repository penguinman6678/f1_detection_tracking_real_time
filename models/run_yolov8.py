from ultralytics import YOLO
import os
import random
import cv2


def train_yolo_model(model_name=None, data_yaml=None, epochs=50, img_size=640, batch=32):

    # Utilizing yolov8-yolov8s.pt --> using small model
    yolov8s_model = YOLO("yolov8s.pt")
    
    # should als save the plots over time
    results = yolov8s_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        device=0
    )

    return results



def inference_yolo_model_test_set(model_path, test_img_path, output_dir=None, image_size=640, conf_threshold=0.40):

    yolo_model = YOLO(model_path)
    results = yolo_model.predict(
        source=test_img_path,
        conf=conf_threshold,
        imgsz=image_size,
        device=0,
        verbose=False
    )

    # Retrieve result + bounding boxes
    result = results[0]
    boxes = result.boxes 

    orig_img = cv2.imread(test_img_path)
    if orig_img is None:
        raise FileNotFoundError(f"File doesn't exist or cannot be read: {test_img_path}")
    
    output = []

    if boxes is not None and len(boxes) > 0:

        bbox_coords = boxes.xyxy.cpu().numpy()
        conf_values = boxes.conf.cpu().numpy()
        cls_idxes = boxes.cls.cpu().numpy().astype(int)

        for idx in range(len(bbox_coords)):
            
            x1, y1, x2, y2 = map(int, bbox_coords[idx])
            conf_val = conf_values[idx]
            cls_idx = cls_idxes[idx]

            output.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf_val,
                    "cls": int(cls_idx)
                }
            )
        
            label = f"{yolo_model.names[cls_idx]} {conf_val:.3f}"
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(orig_img, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    if output_dir is not None:
        cv2.imwrite(output_dir, orig_img)
    
    return output, orig_img

def inference_yolo_model_test_set_metrics(model_path, data_yaml, split="test", img_size=640):

    yolo_model = YOLO(model_path)
    metrics = yolo_model.val(data=data_yaml, split=split, imgsz=img_size)

    return metrics


### HELPER FUNCTIONS
def k_randomized_files(images_dir="../data/test/images", k=10):

    images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    selected_images = random.sample(images, k)

    return selected_images


if __name__ == "__main__":
    
    # Training the model
    # yolov8s_model_results = train_yolo_model(data_yaml="../data/data.yaml", epochs=100, img_size=640, batch=32)
    
    # Inference on K images in test set
    model_path = "./runs/detect/train5/weights/best.pt"
    images_dir = "../data/test/images"

    # k_image_files = k_randomized_files(images_dir=images_dir)
    # for idx in range(len(k_image_files)):
        
    #     test_img_path = k_image_files[idx]
    #     prediction_stats, modified_test_img = inference_yolo_model_test_set(model_path=model_path, test_img_path=f"{images_dir}/{test_img_path}", output_dir=f"./saved_examples_on_test/{test_img_path}")
    #     print(f"For Image {idx}: {prediction_stats}")

    # Test Set Metrics
    metrics_dict = inference_yolo_model_test_set_metrics(model_path=model_path, data_yaml="../data/data.yaml")
    print(metrics_dict.results_dict)
    




