# Blackjack Vision

A local Python starter for the UNC Charlotte Blackjack Vision proposal.
One player, one dealer upcard, separated face-up cards, fixed webcam.

**Start with [START_HERE.md](START_HERE.md).** Manual mode runs immediately.
Real card recognition requires your deck's rank templates or your own trained
13-rank YOLO weights. Neither a dataset nor trained weights are bundled.

## Files and responsibilities

| File | Purpose |
| --- | --- |
| `app.py` | Manual demo, webcam loop, boxes, scores, stable hand display |
| `blackjack.py` | Rank validation, hard/soft totals, busts, natural blackjack |
| `strategy.py` | Fixed-rule basic strategy |
| `regions.py` | Player/dealer regions and consecutive-frame stability |
| `config.json` | Camera, thresholds, regions, corner crop |
| `detectors/baseline.py` | Thresholding, quadrilateral contours, warping, template matching |
| `detectors/yolo_detector.py` | Custom Ultralytics model adapter |
| `scripts/capture.py` | Collect real images grouped by capture session |
| `scripts/label.py` | Draw full-card boxes and enter rank labels |
| `scripts/make_templates.py` | Capture baseline rank templates |
| `scripts/prepare_data.py` | Session split, label conversion, duplicate guard, dataset manifest |
| `scripts/train.py` | Train YOLO using training and validation data |
| `scripts/evaluate.py` | Same-image detector and hand evaluation |
| `tests/` | Hand, strategy, stability, and input-validation checks |

Run all commands from this folder. Python 3.11 or 3.12 recommended.
No API keys or paid services are required. Initial dependency installation and
pretrained YOLO weight download require internet access.

## Rules and boundaries

- Six-deck, dealer stands on soft 17, double on any initial two cards,
  doubling after splitting allowed, no surrender.
- Additional assumption: American hole-card/peek rules; dealer blackjack is
  ruled out before strategy decisions. The app cannot see the hole card.
- Total-dependent basic strategy; no card counting or composition-dependent exceptions.
- `SPLIT` is advice only. Stop that hand's demo at SPLIT; split-hand play,
  resplits, forced single-card split aces, bets, and settlement are not modeled.
- `BLACKJACK` identifies a natural player hand; it does not mean a confirmed win.
- Hold cards still. The app reconstructs the visible hand each frame and requires
  a configurable number of consecutive consistent frames.
- Unrecognized/weak detected cards block advice. A completely missed card can
  still lead to wrong advice: verify every physical card has a box.
- The baseline is sensitive to shadows, glare, background, crop position, deck
  font, and perspective. Use separated white cards on a dark matte surface.
- Baseline scores are template correlations, not probabilities. YOLO and
  baseline thresholds should be selected separately using validation data.

## Data workflow

1. Capture roughly 300-500 images across at least 10 varied sessions. Change
   lighting, orientation, background, and hand size. Do not capture near-identical
   video bursts and call them separate sessions. Include all 13 ranks.
2. Keep each continuous capture/setup session in one folder:

   ```text
   data/raw/day1_bright/frame1.jpg
   data/raw/day1_bright/frame1.txt
   data/raw/day2_dim/frame2.jpg
   data/raw/day2_dim/frame2.txt
   ```

3. Label the **entire card**, not just the corner. YOLO rows are
   `class_id center_x center_y width height`, with normalized coordinates.
   Each image needs a same-stem `.txt`; an empty `.txt` represents no cards.
4. Run `python -m scripts.prepare_data`. Real sessions are divided approximately
   70/15/15 by session count before augmentation. Image proportions can differ.
   The script refuses an existing output directory and cross-split exact duplicates.
   It does not detect near duplicates; session discipline is still essential.
5. `data/prepared/manifest.json` records every image, source, session, split, and hash.
   Inspect class coverage in each split before training. If rare ranks are absent,
   collect more sessions rather than moving neighboring frames across splits.
6. Use only training-session cards/images for rank templates. Do not use validation
   or test images as templates. If live template capture uses a separate session,
   mark it as training-only and exclude it from all evaluation sessions.

External synthetic data is optional. The proposal references the
[Playing Cards Object Detection Dataset](https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset).
Review its license and actual annotation layout before use. This project does
not download it or assume its class order. Stage image/label pairs in a folder;
provide a JSON list matching the dataset's original class IDs, for example
`["AS", "2S", ...]`. Supported name formats include AS, 10H, and ace_of_spades.
Unknown names fail instead of silently mapping to the wrong rank.

```bash
python -m scripts.prepare_data --synthetic data/synthetic --synthetic-classes data/synthetic_classes.json
```

Synthetic images go into training only. If a dataset supplies separate `images`
and `labels` directories, first stage each label beside its matching image.

## YOLO training

```bash
python -m pip install -r requirements-yolo.txt
python -m scripts.train --device cpu
```

On an Apple Silicon Mac, try `--device mps`; if unsupported, use `cpu`.
Training can be slow on a laptop. The default is `yolo11n.pt`, 60 epochs,
640-pixel images, batch 8. Use `--batch 4` if memory is tight.
Photometric, rotation, and scale augmentation happen only during training.
Check the training log for the actual output directory; subsequent runs may
receive numbered directory names.

```bash
python app.py --detector yolo --weights runs/cards/weights/best.pt
```

Generic pretrained weights cannot recognize your 13 ranks without fine-tuning.
The adapter rejects models with a different class-name set.

## Evaluation for your report

Tune only on validation data. After selecting thresholds and model, freeze them
and run the held-out test split once for final reporting.

```bash
python -m scripts.evaluate --detector baseline --split val
python -m scripts.evaluate --detector yolo --weights runs/cards/weights/best.pt --split val
python -m scripts.evaluate --detector baseline --split test
python -m scripts.evaluate --detector yolo --weights runs/cards/weights/best.pt --split test
```

JSON summaries and per-image CSVs go in `reports/`. Outputs with the same
detector/split overwrite earlier reports; use `--output reports/experiment_2`
to preserve experiments. Use different `--config` files for independently tuned
detector thresholds, and record those thresholds in your report.

- Rank-aware precision/recall: greedy confidence-ordered matching, correct rank,
  box IoU >= 0.5. Unknown ranks are abstentions and unmatched truth counts as missed.
- Complete-hand accuracy: exact rank multiset in the player region and correct
  single dealer upcard. Duplicate ranks are preserved.
- Correct-total accuracy: correct player total among eligible labeled hands;
  withheld predictions count as failures.
- Eligible hand: labels contain >=2 player cards and exactly one dealer card
  under the same normalized region configuration. Other images still contribute
  to detection metrics. Confirm the regions match evaluation images.
- Recommendation accuracy: requires `--actions data/test_actions.json`, a JSON
  object mapping test image filenames to independently verified uppercase actions.
  Example: `{"real_000002.jpg": "HIT", "real_000003.jpg": "SPLIT"}`.
  Replace example filenames with actual manifest filenames. Verify against a
  separate trusted strategy chart; do not generate answers using `strategy.py`.
  Use all eligible images to avoid selectively reporting favorable cases.
- Latency/FPS: detector-only still-image timings after one warm-up. They exclude
  camera capture, rendering, and temporal stabilization, so do not present them
  as end-to-end webcam FPS. Record full application FPS separately on demo hardware.

Use the CSV `session` column to compare lighting/rotation conditions. Save example
success/failure screenshots yourselves and explain glare, occlusion, missed cards,
or rank confusion. The project cannot supply real accuracy results before your data
is collected and models are evaluated.

## Suggested five-person split

| Person | Deliverable |
| --- | --- |
| 1 | Capture sessions, card annotations, split and label review |
| 2 | Baseline templates, crop tuning, OpenCV error analysis |
| 3 | YOLO training, validation, model selection |
| 4 | Webcam UI, region placement, stability and demo integration |
| 5 | Independent strategy checks, evaluation, figures and report |

Everyone should understand the detector interface and avoid changing the test set
after results are observed.

## References

- Project scope: uploaded *Blackjack Vision: A Real-Time Practice Assistant* proposal.
- Strategy: https://wizardofodds.com/games/blackjack/strategy/4-decks/
  (S17 rules; no-surrender fallback, DAS enabled).
- Ultralytics training: https://docs.ultralytics.com/modes/train/
- Ultralytics labels: https://docs.ultralytics.com/datasets/detect/

Third-party packages and datasets retain their own licenses. Review the
Ultralytics license before distributing a product built on it.
