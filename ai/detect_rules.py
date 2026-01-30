"""
AI core (MVP):
- Detect people using YOLOv8
- Crowd gathering alert: people_count_in_roi >= CROWD_THRESHOLD
- Restricted area intrusion alert: any person intersects restricted ROI

Run (local):
  pip install ultralytics opencv-python shapely
  python ai/detect_rules.py --source 0
  python ai/detect_rules.py --source path/to/video.mp4

Note:
- ROI is hard-coded for now. We'll upgrade to load from JSON later.
"""

import argparse
import time
from typing import List, Tuple
import numpy as np

import cv2
from shapely.geometry import Polygon, box

from ultralytics import YOLO

# -----------------------------
# Config (edit these)
# -----------------------------
CROWD_THRESHOLD = 8          # >= this number => crowd alert
CROWD_HOLD_SECONDS = 3.0     # must persist for N seconds
INTRUSION_HOLD_SECONDS = 1.0 # must persist for N seconds

# ROI polygons (x, y) in pixels (example values!)
# You MUST adjust these based on your camera/video frame.
CROWD_ROI_POINTS = [(100, 100), (540, 100), (540, 420), (100, 420)]
RESTRICTED_ROI_POINTS = [(560, 120), (840, 120), (840, 420), (560, 420)]

# -----------------------------
# Helpers
# -----------------------------
def draw_polygon(img, pts, color=(0, 255, 255), thickness=2):
    pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts_np], True, color, thickness)

def poly_from_points(points: List[Tuple[int, int]]) -> Polygon:
    return Polygon(points)

def bbox_to_shapely(x1, y1, x2, y2):
    return box(x1, y1, x2, y2)

def now_s() -> float:
    return time.time()

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0 for webcam or path to video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model (n/s/m/l/x)")
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    args = parser.parse_args()

    source = 0 if str(args.source) == "0" else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    model = YOLO(args.model)

    crowd_roi = poly_from_points(CROWD_ROI_POINTS)
    restricted_roi = poly_from_points(RESTRICTED_ROI_POINTS)

    crowd_start = None
    intrusion_start = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        # YOLO inference
        results = model.predict(frame, conf=args.conf, verbose=False)
        r0 = results[0]

        people_boxes = []
        if r0.boxes is not None:
            for b in r0.boxes:
                cls = int(b.cls[0].item())
                # COCO class 0 = person
                if cls != 0:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                people_boxes.append((x1, y1, x2, y2))

        # Rules
        people_in_crowd_roi = 0
        intrusion_hit = False

        for (x1, y1, x2, y2) in people_boxes:
            bb = bbox_to_shapely(x1, y1, x2, y2)
            if bb.intersects(crowd_roi):
                people_in_crowd_roi += 1
            if bb.intersects(restricted_roi):
                intrusion_hit = True

        # Crowd hold logic
        crowd_alert = False
        if people_in_crowd_roi >= CROWD_THRESHOLD:
            if crowd_start is None:
                crowd_start = now_s()
            if now_s() - crowd_start >= CROWD_HOLD_SECONDS:
                crowd_alert = True
        else:
            crowd_start = None

        # Intrusion hold logic
        intrusion_alert = False
        if intrusion_hit:
            if intrusion_start is None:
                intrusion_start = now_s()
            if now_s() - intrusion_start >= INTRUSION_HOLD_SECONDS:
                intrusion_alert = True
        else:
            intrusion_start = None

        # Draw
        # Draw ROIs
        crowd_pts = np.array(CROWD_ROI_POINTS, dtype=np.int32).reshape((-1, 1, 2))
        rest_pts  = np.array(RESTRICTED_ROI_POINTS, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [crowd_pts], True, (0, 255, 255), 2)
        cv2.polylines(frame, [rest_pts], True, (0, 0, 255), 2)

        # Draw boxes
        for (x1, y1, x2, y2) in people_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

        # Overlay text
        status_lines = [
            f"People in crowd ROI: {people_in_crowd_roi} (thr={CROWD_THRESHOLD})",
            f"CROWD ALERT: {'YES' if crowd_alert else 'no'}",
            f"INTRUSION ALERT: {'YES' if intrusion_alert else 'no'}",
        ]
        y = 30
        for line in status_lines:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 28

        cv2.imshow("Campus Abnormal Behavior Detection - MVP", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
