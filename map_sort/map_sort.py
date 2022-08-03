from typing import List, Tuple
import numpy as np
import shapely.geometry as geo

from .mapper import PixelMapper
from .tracklet import Tracklet
from .distance import iou_cost_function, iou
from . import matching
from .detection import Detection


class Map_SORT(object):
    def __init__(self, pixel_mapper: PixelMapper, entry_polys: List[geo.Polygon], max_age=900, min_hits=3, iou_threshold=0.7, limit_entry=True):
        self.pixel_mapper = pixel_mapper
        self.entry_polys = entry_polys
        self.limit_entry = limit_entry
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracklet_counter = 1
        self.iou_threshold = iou_threshold
        self.frames = 0

        self.trackers: List[Tracklet] = []
        self.lost_trackers: List[Tracklet] = []

    def get_trackers(self):
        return self.trackers, self.lost_trackers

    def _init_tracklets(self, dets: List[Detection]):
        new_trackers = []
        for det in dets:
            create = False

            # Only create if the map point is within an entry region
            if self.frames > self.min_hits and self.limit_entry:
                map_point = geo.Point(*self.pixel_mapper.detection_to_map(det, [0,1]))
                for poly in self.entry_polys:
                    if poly.contains(map_point):
                        create = True
            else:
                create = True

            # Gate detection by IOU with current trackers
            if create:
                for trk in self.trackers:
                    if iou(det, trk.get_bbox()) > self.iou_threshold:
                        create = False
                        break

            if create:
                trk = Tracklet(self.frames, self.tracklet_counter, det, self.pixel_mapper, self.min_hits)
                new_trackers.append(trk)
                self.tracklet_counter += 1

        self.trackers += new_trackers

    def predict(self):
        self.frames += 1
        for trk in self.trackers:
            trk.predict()
    
    def _match(self, dets: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = []
        unconfirmed_tracks = []
        for i, t in enumerate(self.trackers):
            if t.in_probation:
                unconfirmed_tracks.append(i)
            else:
                confirmed_tracks.append(i)

        # Associate confirmed tracks using matching cascade
        matches_a, unmatched_tracks_a, unmatched_detections = \
            matching.matching_cascade(
                self.max_age, self.trackers, dets, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + \
            [t for t in unmatched_tracks_a if self.trackers[t].time_since_update == 1]
        unmatched_tracks_a = [
            t for t in unmatched_tracks_a if self.trackers[t].time_since_update != 1]

        matches_b, unmatched_tracks_b, unmatched_detections = matching.min_cost_matching(
            iou_cost_function, 1-self.iou_threshold, self.trackers, dets, iou_track_candidates, unmatched_detections)

        # Construct final matching results
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def update(self, dets: List[Detection]):
        # Associate predictions to observations
        matched, unmatched_trks, unmatched_dets = self._match(dets)
        for trk_id, det_id in matched:
            trk = self.trackers[trk_id]
            trk.update(self.frames, dets[det_id])

        # Create new trackers for unmatched detections
        self._init_tracklets([dets[d] for d in unmatched_dets])

    def remove_trackers(self):
        new_trackers = []
        # Delete trackers with too high age
        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                self.lost_trackers.append(trk)
            else:
                new_trackers.append(trk)

        # Update the active trackers
        self.trackers = new_trackers

    def get_output(self):
        ret = []
        for trk in self.trackers:
            # If alive and not in probation, add it to the return
            if (self.frames < self.min_hits) or (trk.time_since_update == 0 and not trk.in_probation):
                d = trk.get_bbox()
                map_point, map_vel = trk.get_map_point()
                ret.append((d, map_point, map_vel, trk.trk_id))

        return ret
    
    def step(self, detections: List[Detection]):
        self.predict()
        self.update(detections)
        self.remove_trackers()
        return self.get_output()