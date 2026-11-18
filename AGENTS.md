# Working rules

- Use Ponytail: the smallest complete change, existing patterns, no speculative abstractions or dependencies. Preserve validation, public behavior and model/data contracts.
- Keep credentials, media, models and experiment artifacts out of Git. Run real inference/training on Sol, not this laptop.
- Orchestrator: define scope and acceptance checks, assign non-overlapping edits, share findings, review worker diffs and validation, and report blockers honestly. Use Orca orchestration when requested; follow its receipts and wake messages.
- Workers: read these rules, stay within the assigned scope, preserve others' edits, and report changed files, evidence, checks and risks. Do not commit, push or spawn more workers unless assigned.

# Commit and push

- Keep `core.hooksPath` set to `.githooks` (`git config core.hooksPath .githooks`); never bypass the safety hooks.
- Make one commit per coherent outcome, including its tests and docs. Batch related cleanup; avoid commits for every file or tiny step. Split only unrelated changes.
- Before committing, review `git diff`, run relevant checks and `git diff --check`, then stage explicit paths and inspect `git diff --cached`. Report pre-existing failures separately.
- Only the orchestrator commits when requested. Use a short `type: outcome` message (for example, `refactor: remove unused translation helpers`).
- Push only when requested, to the intended branch with `git push` (or `git push -u origin HEAD` for its first push). No force pushes or history rewrites unless explicitly requested.
