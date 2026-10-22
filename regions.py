"""Normalized tabletop regions and conservative frame stabilization."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    rank: str | None
    score: float
    box: tuple  # x1, y1, x2, y2 in image pixels


def region_for(box, shape, config):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    x, y = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
    for name, (left, top, right, bottom) in config['regions'].items():
        if left <= x <= right and top <= y <= bottom:
            return name
    return None


def read_table(detections, shape, config):
    hands = {'player': [], 'dealer': []}
    uncertain = False
    for det in detections:
        region = region_for(det.box, shape, config)
        if region in hands:
            if det.rank is None or det.score < config['confidence']:
                uncertain = True
            else:
                hands[region].append(det.rank)
    if uncertain or len(hands['player']) < 2 or len(hands['dealer']) != 1:
        return None
    # Sort without deduplicating: two identical ranks are two distinct cards.
    return tuple(sorted(hands['player'])), tuple(hands['dealer'])


class Stabilizer:
    def __init__(self, frames=8):
        if frames < 1:
            raise ValueError('Stable frames must be positive')
        self.frames = frames
        self.previous = None
        self.count = 0

    def update(self, state):
        if state is None:
            self.previous, self.count = None, 0
            return None
        self.count = self.count + 1 if state == self.previous else 1
        self.previous = state
        # Withhold the old action immediately when a new/uncertain frame appears.
        return state if self.count >= self.frames else None
