"""Save deliberately varied frames under a named capture session."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', required=True, help='Example: day1_dim_rotated')
    p.add_argument('--camera', type=int, default=0)
    args = p.parse_args()
    if Path(args.session).name != args.session or args.session in ('.', '..'):
        p.error('Use a simple folder name for --session.')
    folder = Path('data/raw') / args.session
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera. Check permission and --camera.')
    print('SPACE: save clean frame | q: quit. Change the scene between captures.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow('Capture: SPACE saves, q quits', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == 32:
                name = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f.jpg')
                if not cv2.imwrite(str(folder / name), frame):
                    raise OSError('Could not save image')
                print(folder / name)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
