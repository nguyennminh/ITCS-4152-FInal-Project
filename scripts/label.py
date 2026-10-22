"""Small local annotator: draw one full-card box, then enter its rank."""
import argparse
import json
from pathlib import Path
import cv2
from blackjack import RANKS, normalize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', required=True)
    args = p.parse_args()
    folder = Path('data/raw') / args.session
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not images:
        raise ValueError('No images in session')
    print('Draw box, ENTER accepts. Then type rank in terminal. ESC finishes image.')
    print('Existing labels are skipped. Delete a label .txt to re-label its image.')
    for path in images:
        if path.with_suffix('.txt').exists():
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Cannot read {path}')
        h, w = frame.shape[:2]
        # Display large images at a usable size; labels remain normalized.
        scale = min(1.0, 1100 / w, 750 / h)
        shown = cv2.resize(frame, (round(w*scale), round(h*scale)))
        sh, sw = shown.shape[:2]
        rows = []
        while True:
            x, y, bw, bh = cv2.selectROI(str(path), shown, showCrosshair=True)
            if not bw or not bh:
                break
            while True:
                try:
                    rank = normalize(input('Rank (A, 2-10, J, Q, K): '))
                    break
                except ValueError as error:
                    print(error)
            rows.append(f'{RANKS.index(rank)} {(x+bw/2)/sw:.6f} {(y+bh/2)/sh:.6f} {bw/sw:.6f} {bh/sh:.6f}')
            cv2.rectangle(shown, (x, y), (x+bw, y+bh), (0, 200, 0), 2)
        cv2.destroyAllWindows()
        if input(f'Save {len(rows)} boxes for {path.name}? [y/N] ').strip().lower() == 'y':
            path.with_suffix('.txt').write_text('\n'.join(rows) + ('\n' if rows else ''))
    Path('data/classes.json').write_text(json.dumps(list(RANKS), indent=2))


if __name__ == '__main__':
    main()
