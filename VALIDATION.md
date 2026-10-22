# Validation performed on this starter

- Eight unittest test methods passed (some cover many subcases).
- Includes 8,788 three-card/dealer combinations checking that doubling and
  splitting are never recommended after a hit.
- Strategy checks cover soft hands, pair decisions, no-surrender fallback,
  hard totals, natural blackjack, and busts.
- A temporary generated-image integration check exercised three card contours,
  rank patches, baseline classification, duplicate ranks, region assignment,
  session-based dataset preparation, and the evaluation CLI together.
- The generated-image check is software integration evidence only. It is not
  an estimate of real camera accuracy, recall, latency, or FPS.

Not verified here: physical webcam access, interactive capture/labeling windows
on macOS, your deck's printed ranks, YOLO weight training/inference, and any
real-image test-set performance. No trained models or fabricated evaluation
results are included. Run the real workflow and record your own measurements.
