"""
config.py
---------
All tunable settings for the local fight-detection system live here.
Change values here rather than digging through main.py / pose_utils.py.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_POSE_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n-pose.pt")
GRU_MODEL_PATH = os.path.join(BASE_DIR, "models", "fight_gru_final.keras")

# ---------------------------------------------------------------------------
# Pose extraction — MUST match the values used during training (Colab notebook)
# ---------------------------------------------------------------------------
SEQ_LEN = 30              # frames per sequence fed to the GRU (matches training)
MAX_PEOPLE = 2            # max people tracked per frame (matches training)
NUM_KEYPOINTS = 17        # COCO pose keypoints
FEATURES_PER_PERSON = 3   # x, y, confidence

POSE_CONF_THRESHOLD = 0.4     # YOLO detection confidence (matches training predict/track calls)
MIN_AVG_KP_CONF = 0.12        # per-person avg keypoint confidence filter (matches training)
MIN_BOX_CONF = 0.35           # per-person box confidence filter (matches training)

# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------
FIGHT_PROB_THRESHOLD = 0.40
# NOTE: during evaluation (Step 7) we swept thresholds 0.3-0.6 and looked at
# False Positive Rate vs False Negative Rate. 0.5 is the naive default; we
# lean lower (~0.4) because missing a real fight (false negative) is worse
# than a false alarm for a security use case. Adjust based on your own
# threshold-sweep results from the notebook.

TEMPORAL_SMOOTHING_WINDOW = 5
# Require this many consecutive frames above FIGHT_PROB_THRESHOLD before
# raising an alert. Prevents single noisy frames from triggering false alarms.
# (Step 13 from the original plan.)

# ---------------------------------------------------------------------------
# Camera / runtime
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0           # 0 = default webcam. Change if using an external/USB camera.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DISPLAY_FPS = True
