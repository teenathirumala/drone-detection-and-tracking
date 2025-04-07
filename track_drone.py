import cv2
import torch
import numpy as np
import os
import time
from tqdm import tqdm
from tracker import DroneTracker
import json
from datetime import datetime

class DroneDetectionAndTracking:
    def __init__(self, model_path="drone_model.pth", confidence_threshold=0.5, device=None):
        self.confidence_threshold = confidence_threshold
        self.tracker = DroneTracker()
        self.device = device or ('cuda' if torch.cuda.is_available() else 
                                'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.performance_metrics = {
            'detection_time': [],
            'tracking_time': []
        }

    def process_frame(self, frame):
        start_time = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        det_start = time.time()
        detections, _ = self._get_detections(frame_rgb)
        self.performance_metrics['detection_time'].append(time.time() - det_start)
        
        track_start = time.time()
        if detections:
            centroids = [(int(x + w/2), int(y + h/2)) for x, y, w, h in detections]
            self.tracker.update(centroids, frame_size=(frame.shape[1], frame.shape[0]))
        else:
            self.tracker.update([], frame_size=(frame.shape[1], frame.shape[0]))
        self.performance_metrics['tracking_time'].append(time.time() - track_start)
        
        output_frame = self.tracker.draw_tracking(frame.copy())
        fps = 1 / (time.time() - start_time)
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return output_frame

    def _get_detections(self, frame_rgb):
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model([frame_tensor])[0]
        
        detections = []
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        for i, score in enumerate(scores):
            if score > self.confidence_threshold:
                x1, y1, x2, y2 = boxes[i]
                detections.append([x1, y1, x2-x1, y2-y1])
        
        return detections, float(scores[0]) if len(scores) > 0 else 0.0

    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        start_time = time.time()
        with tqdm(desc="Processing") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed = self.process_frame(frame)
                
                if output_path:
                    out.write(processed)
                
                cv2.imshow('Tracking', processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                pbar.update(1)
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        self._save_results(video_path, start_time)

    def _save_results(self, video_path, start_time):
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"results/{os.path.splitext(os.path.basename(video_path))[0]}_{timestamp}"
        
        metrics = {
            'video': video_path,
            'processing_time': time.time() - start_time,
            'avg_fps': len(self.performance_metrics['detection_time']) / max(1, (time.time() - start_time)),
            'avg_detection_time': np.mean([t for t in self.performance_metrics['detection_time'] if t > 0]) or 0,
            'avg_tracking_time': np.mean([t for t in self.performance_metrics['tracking_time'] if t > 0]) or 0,
            **self.tracker.save_trajectory_data()
        }
        
        with open(f"{base_name}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        self.tracker.save_event_log(f"{base_name}_events.log")
        print(f"Results saved to {base_name}_*")

def process_all_videos():
    import glob
    video_files = sorted(glob.glob("dataset/cam*.mp4"))
    
    if not video_files:
        print("No videos found in dataset/ folder")
        return
    
    tracker = DroneDetectionAndTracking()
    
    for video_file in video_files:
        output_file = f"results/{os.path.basename(video_file).replace('.mp4', '_tracked.mp4')}"
        print(f"\nProcessing {video_file}...")
        tracker.process_video(video_file, output_file)

if __name__ == '__main__':
    process_all_videos()