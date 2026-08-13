"""
main.py
-------
Real-time fight detection from a local webcam / CCTV feed.

Pipeline:
    Webcam --> YOLO11n-Pose (+ByteTrack) --> normalize/filter keypoints
            --> rolling 30-frame buffer --> GRU --> Fight probability
            --> temporal smoothing --> on-screen alert

Run with:
    python main.py
"""

import time
import collections

import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

import config
from pose_utils import filter_and_select_people, build_frame_features, FrameBuffer


def load_models():
    print("Loading YOLO pose model...")
    pose_model = YOLO(config.YOLO_POSE_MODEL_PATH)

    print("Loading GRU fight-classifier model...")
    gru_model = load_model(config.GRU_MODEL_PATH)

    return pose_model, gru_model


def process_frame(frame, pose_model, track_persist=True):
    """
    Runs YOLO pose+tracking on a single frame and returns filtered,
    normalized, flattened features ready to push into the FrameBuffer.
    """
    results = pose_model.track(
        source=frame,
        conf=config.POSE_CONF_THRESHOLD,
        persist=track_persist,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    if result.keypoints is None or len(result.keypoints) == 0:
        people_kpts = np.zeros((0, config.NUM_KEYPOINTS, config.FEATURES_PER_PERSON))
    else:
        xy = result.keypoints.xy.cpu().numpy()
        conf = result.keypoints.conf.cpu().numpy()
        box_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(xy))
        people_kpts = filter_and_select_people(xy, conf, box_conf)

    frame_features = build_frame_features(people_kpts)
    annotated_frame = result.plot()   # <-- NEW: draws skeleton, keypoints, boxes, track IDs

    return frame_features, result, annotated_frame
    frame_features = build_frame_features(people_kpts)
    return frame_features, result


def draw_overlay(frame, fight_prob, is_alert, fps=None):
    h, w = frame.shape[:2]

    color = (0, 0, 255) if is_alert else (0, 200, 0)
    label = f"FIGHT DETECTED  ({fight_prob:.2f})" if is_alert else f"Normal  ({fight_prob:.2f})"

    cv2.rectangle(frame, (0, 0), (w, 40), color, -1)
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if fps is not None and config.DISPLAY_FPS:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


def main():
    pose_model, gru_model = load_models()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"ERROR: could not open camera index {config.CAMERA_INDEX}")
        return

    buffer = FrameBuffer()
    recent_flags = collections.deque(maxlen=config.TEMPORAL_SMOOTHING_WINDOW)

    print("Starting real-time detection. Press 'q' to quit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        frame_features, _, annotated_frame = process_frame(frame, pose_model)
        buffer.push(frame_features)

        fight_prob = 0.0
        is_alert = False

        if buffer.is_ready():
            sequence = buffer.get_sequence()  # (1, SEQ_LEN, feature_dim)
            fight_prob = float(gru_model.predict(sequence, verbose=0)[0][0])

            recent_flags.append(fight_prob > config.FIGHT_PROB_THRESHOLD)
            is_alert = (
                len(recent_flags) == config.TEMPORAL_SMOOTHING_WINDOW
                and all(recent_flags)
            )

        # FPS calc
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
        prev_time = curr_time

        annotated_frame = draw_overlay(annotated_frame, fight_prob, is_alert, fps=fps)
        cv2.imshow("Fight Detection", annotated_frame)

        if is_alert:
            print(f"[ALERT] Fight detected — probability {fight_prob:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
