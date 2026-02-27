# sport_tactical_ai

Pipeline for tactical analysis in sports using detection, tracking and
self-supervised learning.

Structure:
- src/perception : detection & tracking
- src/preprocessing : team assignment and sequence building
- src/models : tactical models
- src/training : self-supervised training
- src/inference : inference pipelines
- data : local datasets (not tracked by git)



## Player & Ball Detection Module

File: `src/perception/detector.py`
This module provides a lightweight wrapper around a YOLOv8 detector to perform
object detection on individual video frames. It is designed to serve as the
first stage of the perception pipeline and produces detections that can be
directly consumed by the tracking, team assignment and sequence building
modules.

### Class: PlayerBallDetector

The `PlayerBallDetector` class loads a YOLOv8 model and exposes a simple
interface for running inference on frames.

**Initialization**

```python
PlayerBallDetector(model_path="yolov8x.pt", device="cuda")
````

* `model_path` specifies the YOLOv8 weights file.
* `device` determines the execution device (`"cuda"` for GPU or `"cpu"`).
* The YOLO model is loaded once during object creation.

### Detection interface

```python
detections = detector.detect(frame)
```

The `detect` method accepts a single image frame and performs object detection
using the loaded YOLO model.

Internally, the model is executed as:
results = self.model(frame, device=self.device, verbose=False)[0]
Only the first result is used since a single frame is passed at a time.

### Output format

Each detected object is returned as a dictionary with the following structure:

```python
{
    "bbox": [x1, y1, x2, y2],
    "class": class_id,
    "confidence": score
}
```

* `bbox` contains the bounding box in pixel coordinates in XYXY format.
* `class` is the predicted class index produced by the YOLO model.
* `confidence` is the detection confidence score.

The function returns a list of such dictionaries.

### Implementation details

The detector extracts raw prediction tensors from the Ultralytics `Results`
object:

* `results.boxes.xyxy` for bounding box coordinates,
* `results.boxes.cls` for class indices,
* `results.boxes.conf` for confidence scores.

These tensors are moved to the CPU and converted to NumPy arrays before being
converted into standard Python types. This ensures that the output can be
easily serialized and passed to downstream modules such as tracking and team
assignment.

### Notes

* The default model (`yolov8x.pt`) uses COCO class labels.
* This module does not perform any class filtering (for example, restricting
  detections to players or the ball). All detected objects are returned.
  Class-specific filtering is intended to be handled in later stages of the
  pipeline.
* The output format is intentionally minimal so it can be easily stored under
  `data/detections/` and reused by other components of the project.


