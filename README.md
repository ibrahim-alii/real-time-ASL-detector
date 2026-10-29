# ASL to English

## Direction

- Compare pretrained ASL translation models using common development clips and input controls.
- Adapt promising models on verified YouTube-ASL and realigned How2Sign clips on Sol.
- Evaluate translation and meaning preservation before scaling training or enabling continuous captions.
- The camera workspace routes to experimental Uni-Sign and SHuBERT models; one-clip Sol parity passed for each.

## Where we are

- [x] Data import, frozen snapshots, source-overlap checks, recording staging and pose caches.
- [x] Experimental SHuBERT and Uni-Sign workspace routes; each matched its reference on one Sol clip.
- [x] Removed the project’s older LandmarkT5 translation pipeline; model-internal pretrained components remain.
- [x] Uni-Sign scored 0.28 BLEU / 13.58 chrF on 200 tuning clips using cached MMPose inputs.
- [x] SHuBERT scored 10.30 BLEU / 34.52 chrF on 32 tuning clips and exceeded static and mismatched controls.
- [ ] Demonstrate useful meaning preservation and browser-to-model behavior on representative inputs.

These tuning results use different samples and are not a head-to-head ranking. Checkpoint training exposure
is uncertain; held-out test scoring and human meaning review remain pending. One-clip parity verifies
routing consistency, not accuracy. The observed warm requests took about 42 seconds for Uni-Sign and
74 seconds for SHuBERT, excluding initialization, so real-time translation is not established.

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
  "model": "unisign",
  "unisign_source": "/path/to/pinned-unisign-source",
  "unisign_tokenizer": "/path/to/verified-tokenizer",
  "unisign_pythonpath": ["/path/to/offline-dependencies"],
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

For Uni-Sign, use the released checkpoint and the pinned source and tokenizer directories shown above.
Include any required existing offline dependency directories in `"unisign_pythonpath"`.
To select SHuBERT, add `"model": "shubert"` and `"shubert_model_root"` (the verified Sol
scratch asset directory). The pinned checkpoint is loaded from that directory. Its worker uses
the separate `"python"` environment; MMPose extraction supplies the display overlay.
SHuBERT is the default model; select Uni-Sign explicitly as shown above. Both require warm workers.
Older LandmarkT5 workspace configurations must be replaced with a supported model configuration.

The server environment needs OpenCV and `ffmpeg` with H.264 support. Use this updated source on Sol;
the extractor and checkpoint identities must match. Forward local port 8765 to the allocated node's
loopback port 8765 through your configured SSH jump host, then open the local URL above.
The server is loopback-only. Match local and remote port numbers; uploads are capped at 16 MiB,
processing at ten minutes. Session media and caches are removed after processing.

Run synthetic workspace checks with `python -m pytest tests/test_workspace.py -q`.
The JavaScript contract check uses Node when available. These tests need no camera, dataset or weights.

Run real comparisons only inside a Sol GPU allocation; keep media, model assets and outputs on Sol.
