from .kalman_utils import construct_position_filter, project
import numpy as np
from .detection import Detection


class Tracklet():
    def __init__(self, frame_num, trk_id, det, min_hits=3):
        self.trk_id = trk_id
        
        self.min_hits = min_hits
        self.hit_streak = 1  # Number of consecutive detections for the tracker
        self.time_since_update = 0
        self.last_update_frame = frame_num
        self.age = 1
        self.in_probation = True
        self.history = []

        self.kf = construct_position_filter(det)
        self.body_R = self.kf.R

    def predict(self):
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        
        # Keep aspect ratio positive
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        # Keep height positive
        if((self.kf.x[7]+self.kf.x[3])<=0):
            self.kf.x[7] *= 0.0

        self.kf.predict()

    def update(self, frame_num: int, detection: Detection):
        self.kf.update(detection.to_xyah().reshape((4,1)))
        self.hit(frame_num)

    def hit(self, frame_num: int):
        # Update hit streak if it has not already this frame
        if frame_num != self.last_update_frame:
            self.last_update_frame = frame_num
            self.hit_streak += 1
            self.time_since_update = 0
            if self.hit_streak >= self.min_hits:
                self.in_probation = False

    def get_bbox(self):
        ret = self.kf.x[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        
        return Detection(ret, 1.0)

    def get_state(self):
        return project(self.kf)

    def record_state(self):
        state = self.get_state()
        self.history.append(state)