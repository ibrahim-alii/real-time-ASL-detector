# ASL to English

## Final solution

- Collect verified ASL video and English text pairs from YouTube-ASL and realigned How2Sign.
- Extract hand, face and body landmarks with MMPose: 133 points and 266 xy values per frame.
- Use a learned projection and pretrained T5.1.1-base to translate landmark sequences into English.
- Train on filtered YouTube-ASL, fine-tune on How2Sign, and evaluate recorded-video translations.
- Measure translation quality with BLEU and human meaning review, including questions, negation and failures.

## Where we are — October 19, 2026

- [x] Implemented data import, frozen snapshots, source-overlap checks and verified recording staging.
- [x] Implemented MMPose extraction, missing-point masks and identity-checked feature caches.
- [x] Implemented configurable T5 training, dev validation, checkpoints, resume and recorded-clip inference.
- [x] Implemented person filtering, review queues and resource estimates.
- [x] Saved pilot receipts confirm 32 How2Sign train and 8 dev examples selected and 40 full recordings staged.
- [x] Pilot extraction retry submitted after a source-video decode failure; completion and visual review remain unverified.
- [ ] Complete real paired-data training; the saved receipts still mark training and semantic review pending.
- [ ] Establish useful translation quality; implementation tests do not demonstrate ASL accuracy.

## What we need to accomplish

- [ ] Inspect pose overlays, signer selection and caption alignment on representative recordings.
- [ ] Verify extraction completion, compatible caches and all requested example counts.
- [ ] Run the ten-update How2Sign smoke and verify uninterrupted versus 5+5 resumed training.
- [ ] Run a short YouTube-ASL → How2Sign training pilot and save initial/latest/best checkpoints.
- [ ] Review dev predictions, BLEU, meaning preservation, failures and shuffled-input controls.
- [ ] Measure usable signing hours, source/signer diversity, training time, GPU memory and storage.
- [ ] Expand data and training only after review and resource checks pass.
- [ ] Define measurable quality targets from the dev baseline before final test evaluation.
- [ ] Finalize the chosen model, evaluate held-out test clips and report results and limitations.
