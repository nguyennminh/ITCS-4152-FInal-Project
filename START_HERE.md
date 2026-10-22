# Start here

## 1. Open the project in VS Code

Unzip `blackjack-vision.zip`, then open its `blackjack-vision` folder.
Open a terminal in that folder.

## 2. Create your environment (Mac)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activation is `.venv\Scripts\Activate.ps1` in PowerShell.

## 3. Run the immediate demo

```bash
python app.py --manual --player A 7 --dealer 6
```

Expected: soft 18, `DOUBLE`.

```bash
python app.py --manual --player A 7 K --dealer 9
python app.py --manual --player 8 8 --dealer 10
python -m unittest discover -s tests -v
```

Expected: hard 18, `STAND`; pair of eights, `SPLIT`; passing tests.

## 4. Create your rank templates

Place a single card on a dark, plain surface under the webcam. Keep the camera
mostly overhead. Start with an ace:

```bash
python -m scripts.make_templates --rank A
```

Check the enlarged patch: it must show only the rank, including both digits for
10. It should not include the suit. SPACE saves; `f` flips the warped card;
`q` quits. Adjust `baseline.corner_crop` in `config.json` if needed, then recreate
all templates with that same crop.

Repeat for **A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K**. For example:

```bash
python -m scripts.make_templates --rank 2
python -m scripts.make_templates --rank 10
python -m scripts.make_templates --rank K
```

Save several clean examples per rank, including red and black suits. This is
training data: do not later evaluate on these captures.

## 5. Start the webcam demo

```bash
python app.py
```

Put exactly one dealer upcard in the upper marked region and two or more player
cards in the lower marked region. Separate cards, keep them entirely visible,
and hold still. The app displays boxes, rank scores, total, and action.
`q` quits; `r` restarts the stability counter.

If the wrong camera opens:

```bash
python app.py --camera 1
```

On macOS, allow camera access for Terminal/VS Code in System Settings > Privacy
& Security > Camera. A connected iPhone may appear as another camera.

## 6. Collect and label real data

```bash
python -m scripts.capture --session day1_bright
python -m scripts.label --session day1_bright
```

Capture: SPACE saves; q quits. Label: draw a box around the whole card, ENTER
accepts, then enter its rank in the terminal. Repeat for every card. ESC finishes
the image; confirm saving in the terminal. Labeling can require switching focus
between the image window and terminal.

Repeat with independently varied sessions (10+ recommended), then:

```bash
python -m scripts.prepare_data
```

See README for YOLO training, optional synthetic data, evaluation, and team tasks.

## What is still your group's work?

- Collect/label the real dataset and review annotation quality.
- Create deck-specific templates and tune on validation data.
- Train the YOLO model and compare approaches.
- Measure actual recognition accuracy and application speed on your hardware.
- Produce the final demo, failure analysis, figures, and project report.

This ZIP provides implementation code and workflows, not a trained or validated
card-recognition model.
