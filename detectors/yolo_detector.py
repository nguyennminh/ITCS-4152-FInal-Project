"""Adapter for a custom 13-rank Ultralytics detection model."""
from pathlib import Path
from blackjack import RANKS
from regions import Detection


class YoloDetector:
    def __init__(self, weights):
        if not weights or not Path(weights).is_file():
            raise ValueError('Supply --weights pointing to your trained best.pt file.')
        from ultralytics import YOLO
        self.model = YOLO(weights)
        if set(self.model.names.values()) != set(RANKS):
            raise ValueError('Model must use the 13 rank labels from prepare_data.py.')

    def detect(self, frame):
        # Retain weak candidates so the app can withhold uncertain advice.
        result = self.model.predict(frame, conf=0.10, iou=0.45, verbose=False)[0]
        return [Detection(self.model.names[int(b.cls.item())], float(b.conf.item()),
                          tuple(b.xyxy[0].cpu().tolist())) for b in result.boxes]
