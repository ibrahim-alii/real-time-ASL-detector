# ASL to English

## Final solution

- Collect verified ASL video and English text pairs from YouTube-ASL and realigned How2Sign.
- Extract hand, face and body landmarks with MMPose: 133 points and 266 xy values per frame.
- Use a learned projection and pretrained T5.1.1-base to translate landmark sequences into English.
- Train on filtered YouTube-ASL, fine-tune on How2Sign, and evaluate recorded-video translations.
- Measure translation quality with BLEU and human meaning review, including questions, negation and failures.

## Where we are — October 21, 2026

- [x] Implemented data import, frozen snapshots, source-overlap checks and verified recording staging.
- [x] Implemented MMPose extraction, missing-point masks and identity-checked feature caches.
- [x] Implemented configurable T5 training, dev validation, checkpoints, resume and recorded-clip inference.
- [x] Implemented person filtering, review queues and resource estimates.
- [x] Saved acquisition audit covers all 2,212 train, 132 dev and 184 test How2Sign recordings; usable clip quality remains separate.
- [x] Extracted pilot features for 54 train / 9 dev YouTube clips and 32 train / 8 dev How2Sign clips.
- [x] Passed real ten-update versus 5+5 resumed training equivalence checks.
- [x] Completed 32 YouTube updates followed by 16 How2Sign updates, checkpoints, dev controls and recorded inference.
- [x] Fixed token-normalized gradient accumulation, recoverable checkpoint selection and caption-bounded frame sampling; added regression tests.
- [x] Preserve raw pose coordinates, scores and timestamps with regional quality audits and versioned preprocessing identities.
- [x] Stream extraction and overlays, reuse separate warm model workers, and bound verified feature caching to 64 MiB.
- [x] Added source exposure accounting and dev-only conditioning, gradient and input-cost diagnostics.
- [x] Compared the released Uni-Sign pose model with the pilot on eight dev clips: BLEU 6.90 versus 0.78; chrF 23.52 versus 6.79.
- [x] Scored the pilot with BLEU and BLEURT; selected BLEURT fell below initial scores on both corpora, and YouTube shuffled inputs scored higher than real inputs.
- [ ] Establish useful translation quality; the small engineering pilot does not demonstrate useful ASL translation.
- [x] Earlier acquisition checkpoint: a three-worker check added 6 video/caption pairs, bringing saved coverage to 654 pairs at that checkpoint.
- [x] Require fresh acquisition probes bound to the current cookie and official-source identities; cached pairs cannot prove authentication.

The Uni-Sign comparison is an engineering check with unequal training exposure, not an architecture benchmark.
Both models use the same dev references; no held-out test clips were scored. Human meaning review remains pending.
The baseline checkpoint is at step 100. Its real, shuffled and static pose controls have almost identical dev
likelihood, so temporal sensitivity needs investigation before scaling training.

Measured baseline generation repeats were 246–248 ms with identical output after initialization.
Native pose extraction took a median 6.98 seconds per clip (26.39 frames/second); this is separate from generation.
Verified feature reads measured 2.26 ms from disk versus 0.14 ms warm; tokenization measured 0.10 ms,
so an additional token cache is not justified by this sample.

## What we need to accomplish

- [ ] Inspect pose overlays, signer selection and caption alignment on representative recordings.
- [ ] Verify extraction completion, compatible caches and all requested example counts.
- [ ] Diagnose learning on a small reviewed training set before increasing training scale.
- [ ] Measure each download batch's newly acquired pairs, failure categories and useful pairs per hour before increasing concurrency.
- [ ] Continue bounded unique-source acquisition; 5,000 complete pairs is the first training milestone, not a lifetime acquisition cap.
- [ ] Review dev predictions, BLEU, meaning preservation, failures and shuffled-input controls.
- [ ] Measure usable signing hours, source/signer diversity, training time, GPU memory and storage.
- [ ] Expand data and training only after review and resource checks pass.
- [ ] Define measurable quality targets from the dev baseline before final test evaluation.
- [ ] Finalize the chosen model, evaluate held-out test clips and report results and limitations.

## Camera workspace

- [x] Browser camera preview, silent clips up to 20 seconds, playback and English output.
- [x] Actual 133-point MMPose landmarks synchronized with recorded playback after inference.
- [x] Compact desktop/mobile layout, permission errors, single inference writer and temporary-file cleanup.
- [x] Persistent extractor/translator workers with identity checks, failure recovery and total request timing.
- [ ] Verify the browser-to-model workflow against a live Sol GPU allocation; model quality remains unverified.

Start camera-only preview from the checkout with the project's Python environment:

```powershell
.venv/Scripts/python.exe scripts/workspace_server.py
```

Open [the local workspace](http://127.0.0.1:8765). No inference or model downloads run on the laptop.
Recording stays in browser memory; translation sends the clip to the workspace server.

For translation, run the server inside a Sol GPU allocation with `--runtime-config /path/to/private.json`.
Keep this configuration outside version control. It must name existing absolute Sol paths:

```json
{
  "python": "/path/to/translation-environment/bin/python",
  "extraction_python": "/path/to/mmpose-environment/bin/python",
  "checkpoint": "/path/to/selected-checkpoint.pt",
  "pose_config": "/path/to/pose-config.py",
  "pose_weights": "/path/to/pose-weights.pth",
  "detector_config": "/path/to/detector-config.py",
  "detector_weights": "/path/to/detector-weights.pth",
  "work_root": "/scratch/your-user/existing-session-directory"
}
```

The server environment needs OpenCV and `ffmpeg` with H.264 support. Use this updated source on Sol;
the extractor and checkpoint identities must match. Forward local port 8765 to the allocated node's
loopback port 8765 through your configured SSH jump host, then open the local URL above.
The server is loopback-only. Match local and remote port numbers; uploads are capped at 16 MiB,
processing at ten minutes. Session media and caches are removed after processing.

Run synthetic workspace checks with `python -m pytest tests/test_workspace.py -q`.
The JavaScript contract check uses Node when available. These tests need no camera, dataset or weights.

Inspect the bounded dev diagnostic and Uni-Sign comparison commands with
`python -m asl_recognizer.translation.diagnostics --help` and
`python -m asl_recognizer.translation.comparison --help`.
Run real diagnostics and comparisons only inside a Sol GPU allocation; keep media, model assets and outputs on Sol.
