# Local Real-Time Fight Detection

Pipeline: `Webcam → YOLO11n-Pose (+ByteTrack) → GRU classifier → Fight/NonFight`

## Setup

1. **Create the folder structure** (if not already present):
   ```
   fight-detection-local/
   ├── models/
   │   ├── yolo11n-pose.pt
   │   └── fight_gru_final.keras
   ├── config.py
   ├── pose_utils.py
   ├── main.py
   └── requirements.txt
   ```

2. **Copy your two trained/pretrained model files into `models/`:**
   - `yolo11n-pose.pt` — pretrained pose model (downloaded from Colab)
   - `fight_gru_final.keras` — your trained GRU classifier (exported from Colab, Step 8)

3. **Install dependencies** (ideally in a virtual environment):
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux

   pip install -r requirements.txt
   ```

4. **Run it:**
   ```
   python main.py
   ```
   Press `q` in the video window to quit.

## Files

| File | Purpose |
|---|---|
| `config.py` | All tunable constants (paths, thresholds, camera settings). Edit here first when tuning. |
| `pose_utils.py` | Pose normalization + filtering logic. **Must match the Colab training preprocessing exactly** — if you retrain with different logic, update this file too. |
| `main.py` | The real-time loop: capture → pose extraction → buffer → GRU inference → temporal smoothing → on-screen alert. |

## Tuning

- **`FIGHT_PROB_THRESHOLD`** in `config.py` — controls sensitivity. Lower = catches more real fights but more false alarms. Higher = fewer false alarms but risks missing real fights. Pick based on your notebook's threshold-sweep results (Step 7 evaluation).
- **`TEMPORAL_SMOOTHING_WINDOW`** — number of consecutive above-threshold frames required before alerting. Higher = fewer false alarms from momentary noise, but slower to react.
- **`CAMERA_INDEX`** — change if you're using an external/USB camera instead of a laptop's built-in webcam (try 1, 2, etc.).

## Known limitations (carry over from training)

- The GRU was trained on `RWF-2000`, real-world/surveillance-style footage — performance on very different camera angles, lighting, or scene types not represented in that dataset may be weaker.
- Validation results: ~82% accuracy, ROC-AUC ~0.92, with a false-negative rate around 18% at the default 0.5 threshold before tuning (see notebook Step 7 for full confusion matrix and threshold sweep).

