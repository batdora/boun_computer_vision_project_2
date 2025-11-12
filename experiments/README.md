# Experiments

This directory contains optional exploration scripts that build on the core
panorama pipeline implemented in `main.py`.

Run each script as a module so Python can locate the project packages:

```bash
python -m experiments.pairwise_demo <args>
python -m experiments.pipeline_example <mode> <args>
python -m experiments.stitcher_experiment
python -m experiments.correspondance_experiment
```

## Scripts

- `pairwise_demo.py` – legacy two-image workflow kept for quick regression tests.
- `pipeline_example.py` – helper utilities for anchor-based, pairwise, and demo
  stitching pipelines.
- `stitcher_experiment.py` – reproduces the three five-image panorama strategies
  used for the course experiments.
- `correspondance_experiment.py` – evaluates homography quality under different
  correspondence perturbations.

All of these scripts rely on the manual correspondence `.npy` files created
with the interactive picker. They are **not** part of the automated submission.

