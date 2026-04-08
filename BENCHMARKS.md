# Performance Benchmarks (Verified)

These are measured results from running the current `inference.py` implementation locally against all configured tasks.

## Run Configuration

- Command: `python inference.py`
- Tasks: `easy_4_phase,medium_asymmetric,hard_corridor_emergency`
- Model setting: `deepseek-ai/DeepSeek-R1`
- Extracted from `[END]` logs emitted by the script

## Verified Results

| Task | Steps | Score | Success |
|------|------:|------:|:-------:|
| easy_4_phase | 150 | 0.95 | true |
| medium_asymmetric | 150 | 0.98 | true |
| hard_corridor_emergency | 150 | 0.21 | false |

## Notes

- Previous synthetic baseline numbers and estimated impact claims were removed because they were not produced by an automated benchmark script in this repository.
- If you want reproducible baseline comparisons (random, round-robin, queue-only), add explicit baseline policies and run them through the same grader pipeline.
- Hard-task scoring now penalizes severe per-step collapses, unresolved end-state queues, and prolonged emergency presence.
