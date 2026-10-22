"""Run manual mode without dependencies; webcam mode uses OpenCV."""
import argparse
import json
import time
from pathlib import Path
from blackjack import evaluate_hand
from strategy import recommend

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manual', action='store_true')
    parser.add_argument('--player', nargs='+', default=['A', '7'])
    parser.add_argument('--dealer', default='6')
    parser.add_argument('--detector', choices=['baseline', 'yolo'], default='baseline')
    parser.add_argument('--weights')
    parser.add_argument('--camera', type=int)
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--templates', default=str(ROOT / 'templates'))
    args = parser.parse_args()
    if args.manual:
        hand = evaluate_hand(args.player)
        action = recommend(args.player, args.dealer)
        print(f'Player: {args.player} | Dealer: {args.dealer}')
        print(f'Total: {hand.total} ({"soft" if hand.soft else "hard"}) | {action}')
        return

    import cv2
    import numpy as np
    from regions import Stabilizer, read_table
    config = json.loads(Path(args.config).read_text())
    if args.detector == 'baseline':
        from detectors.baseline import BaselineDetector
        detector = BaselineDetector(args.templates, config['baseline'])
    else:
        from detectors.yolo_detector import YoloDetector
        detector = YoloDetector(args.weights)
    cap = cv2.VideoCapture(config['camera'] if args.camera is None else args.camera)
    if not cap.isOpened():
        raise RuntimeError('Camera could not open. Grant camera access or try --camera 1.')
    stable = Stabilizer(config['stable_frames'])
    try:
        while True:
            start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector.detect(frame)
            state = stable.update(read_table(detections, frame.shape, config))
            h, w = frame.shape[:2]
            for name, (x1, y1, x2, y2) in config['regions'].items():
                a, b = (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h))
                cv2.rectangle(frame, a, b, (150, 200, 60), 2)
                cv2.putText(frame, name.upper(), (a[0], max(18, a[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 60), 2)
            for det in detections:
                x1, y1, x2, y2 = map(int, det.box)
                color = (0, 200, 0) if det.rank and det.score >= config['confidence'] else (0, 150, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{det.rank or "?"} {det.score:.2f}', (x1, max(18, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            panel = np.zeros((110, w, 3), np.uint8)
            message = 'WAIT: show 2+ player cards and exactly 1 dealer upcard; hold still.'
            if state:
                player, dealer = state
                hand = evaluate_hand(player)
                message = f'{list(player)} = {hand.total} ({"soft" if hand.soft else "hard"}) | Dealer {dealer[0]} | {recommend(player, dealer[0])}'
            ms = 1000 * (time.perf_counter() - start)
            for i, line in enumerate([message, f'Processing {ms:.0f} ms | q: quit | r: reset stability',
                                      'Check every physical card has a box. Split-hand play is not supported.']):
                cv2.putText(panel, line, (10, 26 + 30*i), cv2.FONT_HERSHEY_SIMPLEX,
                            min(0.52, w / 1600), (240, 240, 240), 1)
            cv2.imshow('Blackjack Vision', np.vstack([frame, panel]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                stable = Stabilizer(config['stable_frames'])
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
