#!/usr/bin/env python
import logging
import random
import typing as t
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from neuron_analyzer.model_util import StepConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)
T = t.TypeVar("T")

######################################################
# Extract surprisal with different interventions


def load_df(file_path: Path, col_header: str) -> pd.DataFrame:
    """Load df from the given column."""
    df = pd.read_csv(file_path)
    start_idx = df.columns.tolist().index(col_header)
    return df[start_idx:]


######################################################
# Step configuration


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


######################################################
# Surprisal extraction


class StepSurprisalExtractor:
    """Extracts word surprisal across different training steps."""

    def __init__(
        self,
        config: StepConfig,
        model_name: str,
        model_cache_dir: Path,
        device: str,
    ) -> None:
        """Initialize the surprisal extractor."""
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.config = config
        self.device = device
        self.current_step = None
        logger.info(f"Using device: {self.device}")
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration."""
        if isinstance(self.model_cache_dir, str):
            self.model_cache_dir = Path(self.model_cache_dir)

    def load_model_for_step(self, step: int) -> tuple[GPTNeoXForCausalLM, AutoTokenizer]:
        """Load model and tokenizer for a specific step."""
        cache_dir = self.model_cache_dir / f"step{step}"

        try:
            model = GPTNeoXForCausalLM.from_pretrained(self.model_name, revision=f"step{step}", cache_dir=cache_dir)
            model = model.to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=f"step{step}", cache_dir=cache_dir)
            model.eval()
            self.current_step = step
            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model for step {step}: {e!s}")
            raise

    def compute_surprisal(
        self,
        model: GPTNeoXForCausalLM,
        tokenizer: AutoTokenizer,
        context: str,
        target_word: str,
        use_bos_only: bool = True,
    ) -> float:
        """Compute surprisal for a target word given a context."""
        try:
            if use_bos_only:
                bos_token = tokenizer.bos_token
                input_text = bos_token + target_word
                bos_tokens = tokenizer(bos_token, return_tensors="pt").to(self.device)
                context_length = bos_tokens.input_ids.shape[1]
            else:
                input_text = context + target_word
                context_tokens = tokenizer(context, return_tensors="pt").to(self.device)
                context_length = context_tokens.input_ids.shape[1]

            inputs = tokenizer(input_text, return_tensors="pt").to(self.device)

            if context_length >= inputs.input_ids.shape[1]:
                context_length = inputs.input_ids.shape[1] - 1

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                target_logits = logits[0, context_length - 1 : context_length]

                if context_length < inputs.input_ids.shape[1]:
                    target_token_id = inputs.input_ids[0, context_length].item()
                    log_prob = torch.log_softmax(target_logits, dim=-1)[0, target_token_id]
                    surprisal = -log_prob.item()
                else:
                    logger.error("Cannot compute surprisal: unable to identify target token")
                    surprisal = float("nan")

            return surprisal

        except Exception as e:
            logger.error(f"Error in compute_surprisal: {e!s}")
            return float("nan")

    def analyze_steps(
        self, contexts: list[list[str]], target_words: list[str], use_bos_only: bool = True, resume_path: Path = None
    ) -> pd.DataFrame:
        """Analyze surprisal across steps."""
        surprisal_frame = (
            load_df(resume_path, "target_word") if resume_path and resume_path.is_file() else pd.DataFrame()
        )
        results = []

        for step in self.config.steps:
            try:
                model, tokenizer = self.load_model_for_step(step)
                surprisal_lst = []

                for word_contexts, target_word in zip(contexts, target_words, strict=False):
                    for context_idx, context in enumerate(word_contexts):
                        surprisal = self.compute_surprisal(
                            model, tokenizer, context, target_word, use_bos_only=use_bos_only
                        )

                        if surprisal_frame.empty:
                            results.append(
                                {
                                    "target_word": target_word,
                                    "context_id": context_idx,
                                    "context": "BOS_ONLY" if use_bos_only else context,
                                    step: surprisal,
                                }
                            )
                        else:
                            surprisal_lst.append(surprisal)

                # Save intermediate results
                if surprisal_frame.empty and results:
                    surprisal_frame = pd.DataFrame(results)
                elif surprisal_lst:
                    surprisal_frame[step] = surprisal_lst

                if resume_path:
                    surprisal_frame.to_csv(resume_path, index=False)

                # Cleanup resources
                del model, tokenizer
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing step {step}: {e!s}")
                continue

        return surprisal_frame
