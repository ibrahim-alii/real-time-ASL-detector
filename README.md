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
- [x] Saved acquisition audit covers all 2,212 train, 132 dev and 184 test How2Sign recordings; usable clip quality remains separate.
- [x] Extracted pilot features for 54 train / 9 dev YouTube clips and 32 train / 8 dev How2Sign clips.
- [x] Passed real ten-update versus 5+5 resumed training equivalence checks.
- [x] Completed 32 YouTube updates followed by 16 How2Sign updates, checkpoints, dev controls and recorded inference.
- [x] Scored the pilot with BLEU and BLEURT; selected BLEURT fell below initial scores on both corpora, and YouTube shuffled inputs scored higher than real inputs.
- [ ] Establish useful translation quality; the small engineering pilot does not demonstrate useful ASL translation.
- [ ] Reconcile fresh-cookie YouTube recovery job 66156129; submission is recorded, but its outcome is unverified locally and current Sol SSH checks timed out.

## What we need to accomplish

- [ ] Inspect pose overlays, signer selection and caption alignment on representative recordings.
- [ ] Verify extraction completion, compatible caches and all requested example counts.
- [ ] Diagnose learning on a small reviewed training set before increasing training scale.
- [ ] Measure each download batch's newly acquired pairs, failure categories and useful pairs per hour before increasing concurrency.
- [ ] Review dev predictions, BLEU, meaning preservation, failures and shuffled-input controls.
- [ ] Measure usable signing hours, source/signer diversity, training time, GPU memory and storage.
- [ ] Expand data and training only after review and resource checks pass.
- [ ] Define measurable quality targets from the dev baseline before final test evaluation.
- [ ] Finalize the chosen model, evaluate held-out test clips and report results and limitations.
