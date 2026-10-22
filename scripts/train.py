"""Fine-tune a small pretrained detector. Test split is never used here."""
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/prepared/cards.yaml')
    p.add_argument('--model', default='yolo11n.pt')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--device', default='cpu', help='cpu, mps (Apple Silicon), or 0 (CUDA)')
    p.add_argument('--batch', type=int, default=8)
    args = p.parse_args()
    from ultralytics import YOLO
    YOLO(args.model).train(data=args.data, epochs=args.epochs, imgsz=640,
                          batch=args.batch, device=args.device, seed=42,
                          degrees=30, scale=0.3, hsv_v=0.3, fliplr=0.0, flipud=0.0,
                          project='runs', name='cards', exist_ok=False)


if __name__ == '__main__':
    main()
