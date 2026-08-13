"""
pose_utils.py
-------------
Pose normalization + per-frame filtering logic.

IMPORTANT: this must stay in sync with the preprocessing used in the Colab
training notebook (Section: "Sequence Padding & Normalization" and the
extract_video_keypoints function). If you retrain the GRU with different
normalization, update this file to match, or the model will see
out-of-distribution input at inference time and predictions will degrade.
"""

import numpy as np
import config


def normalize_pose(person_kpts):
    """
    person_kpts: (17, 3) array -> x, y, confidence

    Normalizes x,y relative to the hip center and body scale (distance
    between shoulder-center and hip-center), so the model sees relative
    body movement rather than absolute camera position.

    Confidence values are passed through unchanged.
    """
    xy = person_kpts[:, :2]
    conf = person_kpts[:, 2:3]

    # COCO keypoint indices: 11 = left hip, 12 = right hip
    left_hip, right_hip = xy[11], xy[12]
    hip_center = (left_hip + right_hip) / 2.0

    # COCO: 5 = left shoulder, 6 = right shoulder
    left_shoulder, right_shoulder = xy[5], xy[6]
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    body_scale = np.linalg.norm(shoulder_center - hip_center)
    if body_scale < 1e-6:
        body_scale = 1.0  # avoid divide-by-zero on bad/degenerate detections

    norm_xy = (xy - hip_center) / body_scale
    return np.concatenate([norm_xy, conf], axis=-1)  # (17, 3)


def filter_and_select_people(xy, conf, box_conf,
                              max_people=config.MAX_PEOPLE,
                              min_avg_conf=config.MIN_AVG_KP_CONF,
                              min_box_conf=config.MIN_BOX_CONF):
    """
    Applies the same confidence-based filtering used during training:
    - drop detections with low average keypoint confidence AND low box confidence
    - keep at most `max_people`, ranked by average keypoint confidence

    xy:       (N, 17, 2)
    conf:     (N, 17)
    box_conf: (N,)

    Returns: (M, 17, 3) where M <= max_people
    """
    if xy.shape[0] == 0:
        return np.zeros((0, config.NUM_KEYPOINTS, config.FEATURES_PER_PERSON))

    combined = np.concatenate([xy, conf[..., None]], axis=-1)  # (N, 17, 3)
    avg_kp_conf = conf.mean(axis=1)  # (N,)

    valid_mask = (avg_kp_conf >= min_avg_conf) & (box_conf >= min_box_conf)
    combined = combined[valid_mask]
    avg_valid = avg_kp_conf[valid_mask]

    if len(combined) > max_people:
        top_idx = np.argsort(avg_valid)[::-1][:max_people]
        combined = combined[top_idx]

    return combined


def build_frame_features(people_kpts, max_people=config.MAX_PEOPLE):
    """
    Converts a single frame's filtered people (M, 17, 3) into a fixed-size,
    normalized, flattened feature vector matching training-time shape.

    Returns: flat vector of length max_people * 17 * 3 (e.g. 102 for max_people=2)
    """
    frame_out = np.zeros((max_people, config.NUM_KEYPOINTS, config.FEATURES_PER_PERSON),
                          dtype=np.float32)

    num_people = min(people_kpts.shape[0], max_people)
    for p in range(num_people):
        frame_out[p] = normalize_pose(people_kpts[p])
    # remaining slots (if fewer than max_people detected) stay zero-padded,
    # matching the padding strategy used when building training sequences.

    return frame_out.flatten()  # (max_people * 17 * 3,)


class FrameBuffer:
    """
    Rolling buffer that holds the last SEQ_LEN frames of pose features.
    Once full, .get_sequence() returns a (SEQ_LEN, features) array ready
    for the GRU model.

    Uses "pad by repeating the last frame" for the first SEQ_LEN-1 frames,
    matching the training-time padding strategy for short sequences.
    """

    def __init__(self, seq_len=config.SEQ_LEN,
                 feature_dim=config.MAX_PEOPLE * config.NUM_KEYPOINTS * config.FEATURES_PER_PERSON):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.buffer = []

    def push(self, frame_features):
        self.buffer.append(frame_features)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

    def is_ready(self):
        return len(self.buffer) >= 1  # we can produce a padded sequence from frame 1 onward

    def get_sequence(self):
        """
        Returns a (1, seq_len, feature_dim) array, batch-ready for model.predict().
        If fewer than seq_len frames have been seen yet, pads by repeating
        the earliest available frame's features backward (best-effort warmup).
        """
        frames = list(self.buffer)
        if len(frames) < self.seq_len:
            pad_count = self.seq_len - len(frames)
            frames = [frames[0]] * pad_count + frames

        seq = np.stack(frames, axis=0)  # (seq_len, feature_dim)
        return np.expand_dims(seq, axis=0).astype(np.float32)  # (1, seq_len, feature_dim)
