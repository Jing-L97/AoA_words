# Age of Acquisition (AoA) Evaluation Benchmark

This repository provides a benchmark for evaluating Age of Acquisition (AoA) in the context of child lexical development based on the paper: [Chang et al. (2022). Emergent Communication: Generalization and Overfitting in Discrete-Sequence Models](https://aclanthology.org/2022.tacl-1.1/).

## Overview

The benchmark computes word surprisal across different training steps to analyze how language models acquire words during training, mirroring patterns observed in child language development. The evaluation extracts surprisal values for target words at various model checkpoints to track acquisition trajectories.

## Usage
To run the evaluation, you need to save intermediate checkpoints during model training and then use the command below

```
python src/scripts/eval/computde.py
```

Optional Arguments

--eval_lst: List of evaluation files for analysis

--interval: Checkpoint interval sampling (default: 10)

--min_context: Minimum number of contexts required (default: 20)

--use_bos_only: Use only beginning-of-sequence token for context

--start: Start index of step range (default: 14)

--end: End index of step range (default: 142)

--debug: Process only the first 5 lines for debugging

--resume: Resume from existing checkpoint
