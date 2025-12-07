# Real-Time F1 Detection, Tracking, and Classification 

This repo holds the codebase for a real time computer vision pipeline for F1 racing videos. It performs:
* F1 Car detection via rrMYOLO (fine-tuned YOLOv8s model)
* F1 Cars tracking (ByteTrack and DeepSORT)
* F1 Team classification via rrmViT (fine-tined ViT-B/16 model)
* Real Time Stream Inferenece (e.g. using DroidCam)

This project was developed as a DLCV final project, with the goal of building a robust system that can handle occlusions and high-motion blurs in real time settings (e.g. enhancing viewing experience, improving broadcast analysis, etc...)

Check out the slides presentation for the TLDR and videos: [DLCV Final Presentation](https://docs.google.com/presentation/d/1j2LuDxHH660Uu1knTbscoAdO0lDr-ei5-5KPmwgqnq0/edit?usp=sharing)

## Brief Abstract 
This project presents a multi-stage computer vision system for real-time detecting, tracking, and classifying Formula 1 cars from video sources (e.g. race highlights and clips). This non-trivial task consists of handling (partial) occlusions, blurs caused by fast movements, and subtle visual differences between teams. 
The approach from this paper consists of a fine-tuned YOLOv8 model, rrMYOLO, for detection, ByteTrack and DeepSORT algorithms for consistent ID tracking of cars throughout continuous, temporal frames, and a fine-tuned Vision Transformer model, rrMViT, for team classification. 
To handle class imbalance within the dataset, we utilize a weighted sampling technique and weighted cross entropy loss. Based on a modified dataset and custom, curated dataset, the system achieves strong detection and classification performances and stable ID tracking under fast-paced scenarios, (partial) occlusions, and high-motion blurs. 
Finally, we demonstrate the system's live-streaming inference capabilities using a mobile camera, enabling real-time annotations of frames. The results demonstrate that modern object detection and tracking with transformer based visual classification can be effectively integrated and deployed in both offline and real-time scenarios.

## Getting Started
The following experiments were performed on a desktop equipped with an NVIDIA RTX 5070TI GPU and 16GB of RAM. The mobile device was an iPhone 13 Pro and the streaming software was DroidCam.

### Dataset
The detection dataset is from Roboflow by [Yoav Fogel](https://universe.roboflow.com/yoav-fogel-yia3f/f1-car-recognition), and it should contain 5899 total images (with their respective training/validation/test splits). Place this within the ```/data/directory/```. Modify the ```/data/data.yaml``` file as needed

The team classification dataset is already in the repository (within ```/data/f1_teams/classification/```). There should be ten subfolders to represent each of the current teams. This is only up-to-date with the 2024 and 2025 seasons. Be aware that starting from the 2026 season, Kick will be replaced by Audi and Cadillac will be entering as a new team. These images were collected via Youtube videos such as [2025 British GP](https://www.youtube.com/watch?v=daWr9xnkKS4) and [2025 Canadian GP](https://www.youtube.com/watch?v=93ZnZF_zWds). *Note* Because there is some class imbalance, a weighted sampler and weighted cross entropy loss function are used.

### Key Dependencies
* PyTorch + torchvision
* Ultralytics
* Opencv
* ByteTrack / DeepSORT dependencies
* Numpy
* DroidCam (for setting up a iPhone as a "live camera" and send frames to a desktop)


### Executing programs
To train / predict with the Yolov8s model, uncomment the training / inference section and run
```
python ./models/run_yolov8.py
```
To execute the ByteTrack tracking algorithm (with a trained Yolov8s model) for saved video / live stream settings, uncomment the necessary sections and run
```
python ./models/run_bytetrack.py
```
To execute the DeepSORT tracking algorithm (with a trained Yolov8s model) for saved video / live stream settings, uncomment the necessary sections and run
```
python ./models/run_deepsort.py
```
To train / predict with the ViT model, uncomment the training / inference section and run either
```
python ./models/run_vit.py
```
or 
```
python ./models/run_vit_scratch.py
```
To curate your own F1 dataset based on a fine-tuned YOLOv8 model (e.g. rrMYOLO):
```
python ./data/extract_cropped_images.py
```

rrMYOLO (model and weights) should be in the repository within ```/models/```.
rrMViT can be accessed [here](https://drive.google.com/file/d/1hMPNhpjnRcBZz2-V1-xsbiwIfphSlAJ7/view?usp=sharing)
The ViT trained from scratch can be accessed [here](https://drive.google.com/file/d/1FKQ8MXxqF1uJK8qSS2wW6GPRw7ZQ0_Dr/view?usp=sharing)

## Brief Quantitative Metrics
Plots and confusion matrics can be found throughout ```/models/``` (depending on which metric you are looking for). For example, if you want to see the training and validation plots for rrViT, check out ```/models/vit_runs/```

**Key Metrics for rrMYOLO:**
| Metric        | Performance   |
| ------------- | ------------- |
| mAP50         | 0.97          |
| mAP50-95      | 0.83          |

**Key Metrics for rrViT and ViT from scratch (based on early stopping):**
| Model         | Train Acc     | Train Loss    | Val Acc       | Val Loss      | Test Acc      |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| rrViT         | 1.00          | 0.003         | 0.92          | 0.0124        | 0.849         |
| ViT Scratch   | 0.83          | 0.0246        | 0.67          | 0.0388        | 0.634         |

**Confusion Matrix for rrViT:**

![CM for rrViT](/models/vit_cm.png)

## Fun Qualitative Visualizations
For "live feed" demos, check [this](https://drive.google.com/file/d/1ekWVNShC8kMhRTC-b8jkQLrdtCA5yJcw/view?usp=sharing) or [this](https://drive.google.com/file/d/1Lh2__MQMmPjdjFzf0Az_e81CcBJFhw6i/view?usp=sharing). The source video was the [2025 Qatar GP](https://www.youtube.com/watch?v=BeaVJggQ2dc&t=9s).

Example images can be found ```/models/saved_examples_test/``` for rrMYOLO.
Below is an example from the validation set:
![Example predictions for rrMYOLO](/models/runs/detect/val2/val_batch2_pred.jpg)

Example images can be found ```/models/vit_tracked_examples_on_test/``` for rrMViT and ```/models/vit_scratch_tracked_examples_on_test/``` for ViT trained from scratch.

Below is an example from ```/models/vit_tracked_examples_on_test/```:

![Example predictions for rrMViT](/models/vit_tracked_examples_on_test/test_0.png)



## Authors

1. Samuel Lee (sl5806)


## Key Sources
* [ByteTrack](https://github.com/FoundationVision/ByteTrack)
* [DeepSORT](https://github.com/nwojke/deep_sort)
* [Ultralytics](https://github.com/ultralytics)
* [ViT](https://docs.pytorch.org/vision/main/models/vision_transformer.html)
* [Original Dataset from Yoav](https://universe.roboflow.com/yoav-fogel-yia3f/f1-car-recognition)
