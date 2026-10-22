"""Classical baseline for separated white cards on a dark, plain table.

Scores are template similarities, not calibrated probabilities.
Tune crop/thresholds using validation images, never test images.
"""
from pathlib import Path
import cv2
import numpy as np
from blackjack import RANKS
from regions import Detection


def warp_card(frame, points):
    p = np.asarray(points, np.float32).reshape(4, 2)
    # Angular order avoids duplicate corners from sum/difference ties at 45 deg.
    center = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0]))]
    p = np.roll(p, -np.argmin(p.sum(axis=1)), axis=0)
    if np.linalg.norm(p[1] - p[0]) > np.linalg.norm(p[2] - p[1]):
        p = np.roll(p, -1, axis=0)
    target = np.float32([[0, 0], [299, 0], [299, 419], [0, 419]])
    matrix = cv2.getPerspectiveTransform(p, target)
    return cv2.warpPerspective(frame, matrix, (300, 420))


def locate_cards(frame, settings):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = frame.shape[0] * frame.shape[1]
    found = []
    for contour in contours:
        a = cv2.contourArea(contour) / area
        if not settings['min_area_fraction'] <= a <= settings['max_area_fraction']:
            continue
        quad = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        _, (rw, rh), _ = cv2.minAreaRect(quad)
        ratio = min(rw, rh) / max(rw, rh)
        if not 0.40 <= ratio <= 0.90:
            continue
        x, y, w, h = cv2.boundingRect(quad)
        found.append(((x, y, x + w, y + h), warp_card(frame, quad)))
    return found


def rank_patch(card, crop):
    h, w = card.shape[:2]
    x1, y1, x2, y2 = crop
    roi = card[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Keep meaningful connected components; preserve both digits in '10'.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    clean = np.zeros_like(ink)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 8:
            clean[labels == i] = 255
    points = cv2.findNonZero(clean)
    if points is None:
        return np.zeros((80, 60), np.uint8)
    x, y, rw, rh = cv2.boundingRect(points)
    return cv2.resize(clean[y:y+rh, x:x+rw], (60, 80), interpolation=cv2.INTER_NEAREST)


class BaselineDetector:
    def __init__(self, templates, settings):
        self.settings = settings
        self.templates = {}
        for rank in RANKS:
            paths = list(Path(templates).glob(f'{rank}_*.png'))
            if (Path(templates) / f'{rank}.png').exists():
                paths.append(Path(templates) / f'{rank}.png')
            images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
            images = [im for im in images if im is not None and im.shape == (80, 60)]
            if images:
                self.templates[rank] = images
        missing = set(RANKS) - self.templates.keys()
        if missing:
            raise ValueError('Missing rank templates: ' + ', '.join(sorted(missing)) +
                             '. Run python -m scripts.make_templates first.')

    def detect(self, frame):
        output = []
        for box, card in locate_cards(frame, self.settings):
            # Both printed corners become top-left after a 180-degree rotation.
            patches = [rank_patch(c, self.settings['corner_crop'])
                       for c in (card, cv2.rotate(card, cv2.ROTATE_180))]
            scores = []
            for rank, templates in self.templates.items():
                best = -1.0
                for patch in patches:
                    if not np.any(patch):
                        continue
                    for template in templates:
                        score = float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0])
                        if np.isfinite(score):
                            best = max(best, score)
                scores.append((best, rank))
            scores.sort(reverse=True)
            score, rank = scores[0]
            if score - scores[1][0] < self.settings['min_match_margin']:
                rank = None
            output.append(Detection(rank, max(0.0, score), box))
        return output
