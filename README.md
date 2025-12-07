# Real-Time F1 Detection, Tracking, and Classification 

This repo holds the codebase for a real time computer vision pipeline for F1 racing videos. It performs:
* F1 Car detection via rrMYOLO (fine-tuned YOLOv8s model)
* F1 Cars tracking (ByteTrack and DeepSORT)
* F1 Team classification via rrmViT (fine-tined ViT-B/16 model)
* Real Time Stream Inferenece (e.g. using DroidCam)

This project was developed as a DLCV final project, with the goal of building a robust system that can handle occlusions and high-motion blurs in real time settings (e.g. enhancing viewing experience, improving broadcast analysis, etc...)

## Brief Abstract 
This project presents a multi-stage computer vision system for real-time detecting, tracking, and classifying Formula 1 cars from video sources (e.g. race highlights and clips). This non-trivial task consists of handling (partial) occlusions, blurs caused by fast movements, and subtle visual differences between teams. 
The approach from this paper consists of a fine-tuned YOLOv8 model, rrMYOLO, for detection, ByteTrack and DeepSORT algorithms for consistent ID tracking of cars throughout continuous, temporal frames, and a fine-tuned Vision Transformer model, rrMViT, for team classification. 
To handle class imbalance within the dataset, we utilize a weighted sampling technique and weighted cross entropy loss. Based on a modified dataset and custom, curated dataset, the system achieves strong detection and classification performances and stable ID tracking under fast-paced scenarios, (partial) occlusions, and high-motion blurs. 
Finally, we demonstrate the system's live-streaming inference capabilities using a mobile camera, enabling real-time annotations of frames. The results demonstrate that modern object detection and tracking with transformer based visual classification can be effectively integrated and deployed in both offline and real-time scenarios.

## Getting Started

### Dependencies



### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
