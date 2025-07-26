Object Detector

This is a simple Object Detector project built using YOLOv8, Python, and OpenCV. It performs real-time object detection on videos or live webcam feeds and counts total detected objects.


Features

1. Real-time object detection from webcam or video input
2. Uses YOLOv8x (most accurate version) for detection
3. Draws bounding boxes with class labels and confidence scores
4. Displays and counts total objects per frame
5. Annotated output video is saved automatically

 How It Works

1. The script loads a pre-trained YOLOv8 model from the Ultralytics library.
2. You can choose between using a video file or live webcam as input.
3. Each frame is passed through the model for object detection.
4. Detected objects are drawn with bounding boxes, and object count is displayed.
5. The result is either shown live (webcam) or saved as an output video.

Setup Instructions

1. Clone the Repository

   ```bash
   git clone https://github.com/your_username/object_detector.git
   cd object_detector
   ```

2. (Optional) Create a Virtual Environment

    Windows:

     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   macOS/Linux:

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install Dependencies

   ```bash
   pip install -r requirements.txt
   ```

> You can also manually install:
>
> ```bash
> pip install ultralytics opencv-python torch
> ```


 Run with Input Video

1. Place your input video as `input_video.mp4` in the `input/` folder.
2. In `test.py`, set:

   ```python
   source = 'input/input_video.mp4'
   ```
3. Run the script:

   ```cmd
   python test.py
   ```



 Run with Live Webcam

1. In `test.py`, set:

   ```python
   source = 0
   ```
2. Run the script:

   ```cmd
   python test.py
   ```

---

 Configuration Options

* Change model version (`'yolov8n'`, `'yolov8s'`, `'yolov8m'`, `'yolov8l'`, `'yolov8x'`) inside `test.py`
* Set confidence threshold:

  ```python
  results = model.predict(source=source, conf=0.5)
  ```
* Save annotated video output using OpenCV's `VideoWriter`

---

 Notes

* Make sure `input/` and `output/` folders exist before running
* Webcam input requires a working camera and OpenCV installed
* Best results achieved with `'yolov8x'` on machines with GPU (e.g., RTX 2050)
