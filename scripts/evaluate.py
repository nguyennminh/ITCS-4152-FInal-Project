"""Evaluate either detector on prepared real val/test images at IoU >= 0.5.

Optional --actions JSON maps image filenames to INDEPENDENTLY verified actions.
Without it, recommendation accuracy is null (not self-graded with our strategy).
"""
import argparse
import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path
import cv2
from blackjack import RANKS, evaluate_hand
from regions import Detection, read_table
from strategy import recommend


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def match_counts(predicted, truth):
    used, true_positive = set(), 0
    for pred in sorted(predicted, key=lambda d: d.score, reverse=True):
        candidates = [(iou(pred.box, gt.box), i) for i, gt in enumerate(truth)
                      if i not in used and pred.rank == gt.rank]
        overlap, index = max(candidates, default=(0, -1))
        if overlap >= 0.5:
            used.add(index)
            true_positive += 1
    return true_positive, len(predicted)-true_positive, len(truth)-true_positive


def load_truth(path, shape):
    h, w = shape[:2]
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = map(float, line.split())
        result.append(Detection(RANKS[int(cls)], 1.0,
                                ((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detector', choices=['baseline', 'yolo'], required=True)
    p.add_argument('--weights')
    p.add_argument('--data', default='data/prepared')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--config', default='config.json')
    p.add_argument('--templates', default='templates')
    p.add_argument('--actions', help='JSON object: image filename -> verified action')
    p.add_argument('--output', default='reports')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.detector == 'baseline':
        from detectors.baseline import BaselineDetector
        detector = BaselineDetector(args.templates, config['baseline'])
    else:
        from detectors.yolo_detector import YoloDetector
        detector = YoloDetector(args.weights)
    data = Path(args.data)
    manifest = json.loads((data / 'manifest.json').read_text())
    entries = [r for r in manifest if r['split'] == args.split and r['session'] != 'synthetic']
    if not entries:
        raise ValueError('No real images in selected split')
    actions = json.loads(Path(args.actions).read_text()) if args.actions else {}
    valid_actions = {'HIT', 'STAND', 'DOUBLE', 'SPLIT', 'BUST', 'BLACKJACK'}
    if not set(actions).issubset({r['image'] for r in entries}):
        raise ValueError('Action JSON contains filenames outside the selected split')
    if any(v not in valid_actions for v in actions.values()):
        raise ValueError('Action JSON contains an invalid action')
    first = cv2.imread(str(data / 'images' / args.split / entries[0]['image']))
    if first is None:
        raise ValueError('Cannot load first image')
    detector.detect(first)  # Warm-up excluded from timing.
    rows = []
    for entry in entries:
        name = entry['image']
        frame = cv2.imread(str(data / 'images' / args.split / name))
        if frame is None:
            raise ValueError(f'Cannot read {name}')
        truth = load_truth(data / 'labels' / args.split / Path(name).with_suffix('.txt'), frame.shape)
        start = time.perf_counter()
        predictions = detector.detect(frame)
        ms = (time.perf_counter() - start) * 1000
        accepted = [d for d in predictions if d.rank and d.score >= config['confidence']]
        tp, fp, fn = match_counts(accepted, truth)
        actual = read_table(truth, frame.shape, config)
        predicted = read_table(predictions, frame.shape, config)
        eligible = actual is not None
        exact = bool(eligible and predicted == actual)
        total_correct = bool(eligible and predicted and
                             evaluate_hand(actual[0]).total == evaluate_hand(predicted[0]).total)
        action = recommend(predicted[0], predicted[1][0]) if predicted else 'WAIT'
        rows.append({'image': name, 'session':entry['session'], 'tp':tp, 'fp':fp, 'fn':fn,
                     'latency_ms':ms, 'eligible_hand':int(eligible), 'exact_hand':int(exact),
                     'correct_total':int(total_correct), 'action':action,
                     'expected_action':actions.get(name, ''),
                     'action_correct':int(action == actions[name]) if name in actions else ''})
    counts = Counter()
    for row in rows:
        for field in ('tp', 'fp', 'fn', 'eligible_hand', 'exact_hand', 'correct_total'):
            counts[field] += row[field]
    def ratio(a, b):
        return a / b if b else None
    latencies = sorted(row['latency_ms'] for row in rows)
    summary = {
        'detector':args.detector, 'split':args.split, 'images':len(rows),
        'confidence_threshold':config['confidence'], 'iou_threshold':0.5,
        'rank_precision':ratio(counts['tp'], counts['tp']+counts['fp']),
        'rank_recall':ratio(counts['tp'], counts['tp']+counts['fn']),
        'eligible_hands':counts['eligible_hand'],
        'complete_hand_accuracy':ratio(counts['exact_hand'], counts['eligible_hand']),
        'correct_total_accuracy':ratio(counts['correct_total'], counts['eligible_hand']),
        'verified_action_images':len(actions),
        'recommendation_accuracy':ratio(sum(r['action_correct'] for r in rows if r['action_correct'] != ''), len(actions)),
        'mean_detection_ms':statistics.mean(latencies),
        'p95_detection_ms':latencies[min(len(latencies)-1, int(0.95*len(latencies)))],
        'detection_only_fps':1000/statistics.mean(latencies),
        'note':'Still-image evaluation; no temporal stabilization, capture, or display time.'
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{args.detector}_{args.split}'
    (out / f'{stem}.json').write_text(json.dumps(summary, indent=2))
    with (out / f'{stem}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
