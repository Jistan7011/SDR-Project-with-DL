# Experiment 2: Session Generalization

This folder is reserved for the next SDR experiment. Keep Experiment 2 raw IQ,
processed datasets, checkpoints, logs, and reports under this folder so the
Experiment 1 baseline remains unchanged.

Use `config/config.exp02.yaml` as the starting configuration and write outputs
under `experiments/exp02_session_generalization/`.

The paper-based review and revised Experiment 2 protocol are documented in:

```text
experiments/exp02_session_generalization/PAPER_REVIEW_REVISED_PLAN.md
```

Core changes from Experiment 1:

- Use the same payload pool for every modulation.
- Split train/val/test by capture session, not random windows from one capture.
- Add channelization/downsampling before 1024-sample window extraction.
- Evaluate by session, payload, gain, offset, and estimated SNR.
