"""Capture rank patches from one training card at a time."""
import argparse
import json
import time
from pathlib import Path
import cv2
from blackjack import RANKS
from detectors.baseline import locate_cards, rank_patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--rank', choices=RANKS, required=True)
    p.add_argument('--config', default='config.json')
    args = p.parse_args()
    settings = json.loads(Path(args.config).read_text())['baseline']
    Path('templates').mkdir(exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera')
    print('Show exactly ONE card. f: flip orientation | SPACE: save | q: quit.')
    print('The patch must contain the rank only, not the suit. Adjust corner_crop if needed.')
    flip = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cards = locate_cards(frame, settings)
            patch = None
            if len(cards) == 1:
                box, card = cards[0]
                if flip:
                    card = cv2.rotate(card, cv2.ROTATE_180)
                patch = rank_patch(card, settings['corner_crop'])
                cv2.imshow('Warped card', card)
                cv2.imshow('Rank only - verify before saving', cv2.resize(patch, (180, 240)))
            cv2.imshow('Template capture', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('f'):
                flip = not flip
            if key == 32 and patch is not None and patch.any():
                name = Path('templates') / f'{args.rank}_{time.time_ns()}.png'
                if not cv2.imwrite(str(name), patch):
                    raise OSError('Could not write template')
                print(name)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
