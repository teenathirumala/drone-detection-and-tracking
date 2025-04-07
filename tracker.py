import numpy as np
import cv2
import datetime
from collections import defaultdict
import json
from typing import Dict, List, Tuple, Optional

class DroneTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_id = 0
        self.objects = {}  # {id: (x, y)}
        self.disappeared = {}  # {id: frames_missing}
        self.trajectory = defaultdict(list)
        self.visibility = defaultdict(list)
        self.reappearances = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.frame_size = (0, 0)

    def register(self, centroid):
        # Check for reappearances first
        for obj_id in list(self.trajectory.keys()):
            if obj_id not in self.objects and self.trajectory[obj_id]:
                last_pos = self.trajectory[obj_id][-1]
                distance = np.linalg.norm(np.array(centroid) - np.array(last_pos))
                if distance < self.max_distance * 3:  # More lenient threshold
                    self._reappear(obj_id, centroid)
                    return obj_id
        
        # Register new object
        obj_id = self.next_id
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.trajectory[obj_id].append(centroid)
        self.visibility[obj_id].append((datetime.datetime.now(), None))
        self.next_id += 1
        return obj_id

    def _reappear(self, obj_id, centroid):
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.trajectory[obj_id].append(centroid)
        self.reappearances[obj_id] += 1
        if self.visibility[obj_id] and self.visibility[obj_id][-1][1] is None:
            self.visibility[obj_id][-1] = (self.visibility[obj_id][-1][0], datetime.datetime.now())
        self.visibility[obj_id].append((datetime.datetime.now(), None))

    def deregister(self, obj_id):
        if obj_id in self.visibility and self.visibility[obj_id][-1][1] is None:
            self.visibility[obj_id][-1] = (self.visibility[obj_id][-1][0], datetime.datetime.now())
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, detections, frame_size=None):
        if frame_size:
            self.frame_size = frame_size
            
        if not detections:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects
            
        if not self.objects:
            for centroid in detections:
                self.register(centroid)
            return self.objects
            
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        
        D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array(detections), axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        used_rows, used_cols = set(), set()
        
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row][col] > self.max_distance:
                continue
                
            obj_id = object_ids[row]
            self.objects[obj_id] = detections[col]
            self.disappeared[obj_id] = 0
            self.trajectory[obj_id].append(detections[col])
            used_rows.add(row)
            used_cols.add(col)
            
        unused_cols = set(range(len(detections))) - used_cols
        for col in unused_cols:
            self.register(detections[col])
            
        unused_rows = set(range(len(object_centroids))) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)
                
        return self.objects

    def draw_tracking(self, frame):
        for obj_id, path in self.trajectory.items():
            if obj_id not in self.objects:
                continue
                
            for i in range(1, len(path)):
                alpha = i / len(path)
                color = (int(255 * alpha), 0, int(255 * (1 - alpha)))
                cv2.line(frame, path[i-1], path[i], color, 2)
                
            cv2.circle(frame, self.objects[obj_id], 5, (0, 255, 0), -1)
            text = f"ID {obj_id}"
            if self.reappearances.get(obj_id, 0) > 0:
                text += f" (R{self.reappearances[obj_id]})"
            cv2.putText(frame, text, 
                       (self.objects[obj_id][0] + 10, self.objects[obj_id][1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def get_visibility_stats(self):
        stats = {}
        for obj_id, periods in self.visibility.items():
            total_time = sum(
                ((end or datetime.datetime.now()) - start).total_seconds()
                for start, end in periods
            )
            stats[obj_id] = {
                'reappearances': self.reappearances.get(obj_id, 0),
                'total_visible_time': total_time,
                'visibility_periods': [
                    (start.isoformat(), end.isoformat() if end else None)
                    for start, end in periods
                ]
            }
        return stats

    def save_event_log(self, filename):
        events = []
        for obj_id, periods in self.visibility.items():
            for i, (start, end) in enumerate(periods):
                events.append((start, obj_id, "APPEARED" if i == 0 else "REAPPEARED"))
                if end:
                    events.append((end, obj_id, "DISAPPEARED"))
        
        events.sort()
        
        with open(filename, 'w') as f:
            for time, obj_id, event in events:
                f.write(f"{time.isoformat()}, ID {obj_id}, {event}\n")

    def save_trajectory_data(self):
        return {
            'frame_size': self.frame_size,
            'trajectories': {
                str(obj_id): [tuple(map(int, pos)) for pos in path]
                for obj_id, path in self.trajectory.items()
            },
            'visibility_stats': self.get_visibility_stats()
        }