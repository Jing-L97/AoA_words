#!/usr/bin/env python
import logging
import random
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)


class StepConfig:
    """Configuration for step-wise analysis of model checkpoints."""

    def __init__(
        self,
        resume: bool = False,
        debug: bool = False,
        file_path: Path | None = None,
        interval: int = 1,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> None:
        """Initialize step configuration."""
        # Generate the complete list of steps
        self.steps = self.generate_pythia_checkpoints()

        # Apply interval sampling while preserving start and end steps
        if interval > 1 or start_idx > 0 or end_idx is not None:
            self.steps = self._apply_interval_sampling(self.steps, interval, start_idx, end_idx)
            range_info = f"from index {start_idx}"
            if end_idx is not None:
                range_info += f" to {end_idx}"
            logger.info(
                f"Applied interval sampling with n={interval} {range_info}, resulting in {len(self.steps)} steps"
            )
        if debug:
            self.steps = self.steps[:5]
            logger.info("Entering debugging mode, select first 5 steps.")
        # If resuming, filter out already processed steps
        if resume and file_path is not None:
            self.steps = self.recover_steps(file_path)
        logger.info(f"Generated {len(self.steps)} checkpoint steps")

    def _apply_interval_sampling(
        self, steps: list[int], interval: int = 1, start_idx: int = 0, end_idx: int | None = None
    ) -> list[int]:
        """Sample steps at every nth interval within specified range while preserving start and end steps."""
        if len(steps) <= 2:
            return steps

        # Validate indices
        start_idx = max(0, min(start_idx, len(steps) - 2))  # Ensure it's between 0 and len(steps)-2

        if end_idx is None:
            end_idx = len(steps) - 1
        else:
            end_idx = max(
                start_idx + 1, min(end_idx, len(steps) - 1)
            )  # Ensure it's between start_idx+1 and len(steps)-1

        # Always keep first and last steps of the original list
        first_step = steps[0]
        last_step = steps[-1]

        # Determine the range for sampling
        if start_idx == 0 and end_idx == len(steps) - 1:
            # Full range case - sample all middle elements
            middle_steps = steps[1:-1][::interval]
        else:
            # Create a slice for the range to sample
            range_to_sample = steps[start_idx : end_idx + 1]

            # If start_idx is 0, we need to exclude the first element to avoid duplication
            if start_idx == 0:
                range_to_sample = range_to_sample[1:]

            # If end_idx is the last index, we need to exclude the last element to avoid duplication
            if end_idx == len(steps) - 1:
                range_to_sample = range_to_sample[:-1]

            # Apply interval sampling to the specified range
            middle_steps = range_to_sample[::interval]

        # Combine all parts
        result = []
        # Always include the first step
        result.append(first_step)
        # Include sampled middle steps
        result.extend(middle_steps)
        # Always include the last step
        result.append(last_step)
        # Ensure no duplicates
        return sorted(list(set(result)))

    def generate_pythia_checkpoints(self) -> list[int]:
        """Generate complete list of Pythia checkpoint steps."""
        # Initial checkpoint
        checkpoints = [0]

        # Log-spaced checkpoints (2^0 to 2^9)
        log_spaced = [2**i for i in range(10)]  # 1, 2, 4, ..., 512

        # Evenly-spaced checkpoints from 1000 to 143000
        step_size = (143000 - 1000) // 142  # Calculate step size for even spacing
        linear_spaced = list(range(1000, 143001, step_size))

        # Combine all checkpoints
        checkpoints.extend(log_spaced)
        checkpoints.extend(linear_spaced)

        # Remove duplicates and sort
        return sorted(list(set(checkpoints)))

    def recover_steps(self, file_path: Path) -> list[int]:
        """Filter out steps that have already been processed based on column names."""
        if not file_path.is_file():
            return self.steps

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Extract completed steps from column headers (only consider fully numeric columns)
        completed_steps = set()
        for col in df.columns:
            if col.isdigit():
                completed_steps.add(int(col))

        # Return only steps that haven't been completed yet
        return [step for step in self.steps if step not in completed_steps]
