import cv2
from ultralytics import YOLO

# Load YOLOv8 model (choose 'yolov8s.pt', 'yolov8m.pt', or 'yolov8l.pt' for higher accuracy)
model = YOLO("yolov8x.pt")  # You can change this to yolov8m.pt or yolov8l.pt

# Set confidence threshold
model.conf = 0.5

# Open webcam (0 = default camera)
cap = cv2.VideoCapture('input/input_video.mp4')

if not cap.isOpened():
    print("Error: Could not open webcam or input video isn't present")
    exit()

cv2.namedWindow("YOLOv8 - Webcam Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 - Webcam Object Detection", 1280, 720)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
out = cv2.VideoWriter("output/output_video.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # Extract detections
    result = results[0]
    boxes = result.boxes
    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()

    # Count objects
    count = len(class_ids)

    # Annotate detections
    annotated_frame = result.plot()

    cv2.putText(annotated_frame, f"Objects Detected: {count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

    cv2.imshow("YOLOv8 - Webcam Object Detection", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
