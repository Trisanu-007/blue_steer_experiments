# RSAI Experiments

A local implementation of the StrongREJECT evaluation framework for testing fine-tuned language models on harmful prompt refusal capabilities.

## Overview

This project evaluates HuggingFace models from the `ModelOrganismsForEM` organization using the [StrongREJECT benchmark](https://huggingface.co/datasets/walledai/StrongREJECT). It generates responses to potentially harmful prompts and scores them using both automated judges and NVIDIA NIM-based evaluation.

The experiment focuses on three model variants:
- `bad-medical-advice`
- `risky-financial-advice`
- `extreme-sports`

## Project Structure

```
.
├── om_em_strongreject_local.py    # Main evaluation script
├── inference_strongreject_judge.py # NVIDIA NIM judge evaluation
├── run_experiment.slurm           # SLURM script for main experiment
├── run_judge_experiments.slurm    # SLURM script for judge evaluation
├── pyproject.toml                 # Project dependencies
├── README.md                      # This file

## Files Description

### Core Scripts

#### `om_em_strongreject_local.py`
Main evaluation script that:
- Discovers and filters models from HuggingFace org
- Downloads models and datasets to local cache
- Generates responses to StrongREJECT prompts
- Scores responses using the StrongREJECT fine-tuned judge
- Produces CSV outputs with scores and analysis

**Key Features:**
- Two-stage pipeline: `prefetch` (download) and `sweep` (evaluate)
- Model size filtering (default: 14B-24B parameters)
- Automatic caching and resume capability
- Batch processing with OOM handling
- Supports both baseline and adapter models

#### `inference_strongreject_judge.py`
Secondary evaluation using NVIDIA NIM API:
- Loads scored CSV files from main experiment
- Evaluates responses using `mistral-medium-3-instruct`
- Provides coherence and misalignment scores
- Generates detailed chain-of-thought explanations
- Outputs JSON files with judge evaluations

### SLURM Scripts

#### `run_experiment.slurm`
Runs the main evaluation sweep:
- Allocates 4 GPUs, 64GB RAM
- 2-day time limit
- Uses cached models from `/ssd_scratch/anu`

#### `run_judge_experiments.slurm`
Runs the NVIDIA NIM judge evaluation:
- CPU-only (no GPU needed)
- 10GB RAM
- Processes all scored CSV files

## Setup

### 1. Environment Setup

```bash
# Create virtual environment (if not already done)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies using uv
pip install uv
uv sync
```

### 2. Configure API Keys and Paths

Set the required environment variables:

```bash
# HuggingFace token (required for model downloads)
export HF_TOKEN="your_huggingface_token"

# NVIDIA API key (required for judge evaluation)
export NVIDIA_API_KEY="your_nvidia_api_key"

# Optional: Cache directory (default: /ssd_scratch/anu)
export HF_HOME="/path/to/cache"
```

**Note:** The script currently has a hardcoded HF token. For security, replace it with environment variable usage.

### 3. Configure Paths (Optional)

Edit `om_em_strongreject_local.py` if needed:
- `CACHE_ROOT`: Model cache location (default: `/ssd_scratch/anu`)
- `RESULTS_ROOT`: Output directory (default: `./results`)

## Running Experiments

### Option 1: Local Execution

#### Full Pipeline (Prefetch + Sweep)
```bash
uv run om_em_strongreject_local.py --stage all --run-id my_experiment
```

#### Prefetch Only (Download models and create inventory)
```bash
uv run om_em_strongreject_local.py --stage prefetch --run-id my_experiment
```

#### Sweep Only (Evaluate using existing manifest)
```bash
uv run om_em_strongreject_local.py --stage sweep --run-id my_experiment
```

#### With Custom Parameters
```bash
uv run om_em_strongreject_local.py \
    --stage all \
    --run-id custom_run \
    --max-size-b 30.0 \
    --limit-bases 5 \
    --limit-prompts 100 \
    --batch-size 16 \
    --max-new-tokens 512 \
    --score-batch 4
```

### Option 2: SLURM Execution

#### Submit Main Experiment
```bash
sbatch run_experiment.slurm
```

Before submitting, edit `run_experiment.slurm` to:
- Update `--run-id` parameter
- Adjust resource allocations if needed
- Change node selection (`--nodelist`)

#### Submit Judge Evaluation
```bash
sbatch run_judge_experiments.slurm
```

### Judge Evaluation (NVIDIA NIM)

After the main experiment completes:

```bash
# Set the API key
export NVIDIA_API_KEY="your_key"

# Run judge evaluation
uv run inference_strongreject_judge.py
```

The script will:
- Process all CSV files in `results/nilanjans_models/scored/`
- Generate JSON outputs in `results/nilanjans_models/judge_evaluations/`
- Skip already processed files (caching enabled)

## Command-Line Arguments

### `om_em_strongreject_local.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--stage` | `all` | Pipeline stage: `prefetch`, `sweep`, or `all` |
| `--run-id` | auto | Run identifier (timestamp if empty) |
| `--max-size-b` | 24.0 | Maximum model size in billions (14B-24B) |
| `--limit-bases` | 0 | Limit number of base models (0=no limit) |
| `--limit-prompts` | 0 | Limit number of prompts (0=no limit) |
| `--batch-size` | 32 | Batch size for generation |
| `--max-input-tokens` | 2048 | Maximum input tokens |
| `--max-new-tokens` | 256 | Maximum tokens to generate |
| `--do-sample` | False | Enable sampling (default: greedy) |
| `--temperature` | 0.0 | Sampling temperature |
| `--top-p` | 1.0 | Top-p sampling |
| `--score-batch` | 8 | Batch size for scoring |
| `--max-response-length` | 512 | Max response length for judge |
| `--score-threshold` | 0.5 | Score threshold for classification |
| `--no-baselines` | False | Skip baseline model evaluations |
| `--force-rerun` | False | Re-evaluate even if cached |

### `inference_strongreject_judge.py`

Configuration is in the script's `main()` section:
- `scored_dir`: Input directory with CSV files
- `output_dir`: Output directory for JSON results
- `model_name`: NVIDIA NIM model (default: `mistral-medium-3-instruct`)
- `batch_size`: Processing batch size (default: 1)
- `rate_limit_per_minute`: API rate limit (default: 40)

## Output Files

### Main Experiment (`results/om-em/<run-id>/`)

**Inventory Files:**
- `inventory/inventory_<timestamp>.csv`: All discovered models with metadata
- `inventory/matrix_check_<timestamp>.csv`: Model selection matrix

**Manifest Files:**
- `manifest/manifest_<timestamp>.json`: Selected models for evaluation

**Scored Results:**
- `scored/<model>__<variant>__strongreject_<timestamp>.csv`: Detailed results per model/variant

**CSV Columns:**
- `prompt`: StrongREJECT prompt
- `response`: Model-generated response
- `strongreject_score`: Judge score (0-1, higher=more harmful)
- `refused_stringmatch`: Binary refusal flag (1=refused)
- `score_ge_threshold`: Binary flag (score >= 0.5)
- `category`: Prompt category

**Logs:**
- `logs/<model>_log.txt`: Per-model evaluation logs

**Summary:**
- `summary_<timestamp>.csv`: Aggregated statistics across all models

### Judge Evaluation (`results/nilanjans_models/judge_evaluations/`)

**Output Format:**
- Files: `<model>__<variant>_judge_eval.json`
- Contains: coherence_score, misalignment_score, chain_of_thought, category

## Dependencies

Key dependencies (from `pyproject.toml`):
- `transformers>=5.3.0`: Load and run HuggingFace models
- `torch>=2.10.0`: PyTorch backend
- `datasets>=4.6.1`: Load StrongREJECT dataset
- `peft>=0.18.1`: Load LoRA/adapter models
- `strong-reject`: StrongREJECT evaluation suite
- `huggingface-hub>=1.5.0`: Model discovery and download
- `pandas`, `numpy`, `matplotlib`: Data processing and visualization
- `accelerate>=1.13.0`: Multi-GPU support
- `pytz`: Timezone handling (IST timestamps)

## Monitoring and Debugging

### Check SLURM Job Status
```bash
squeue -u $USER
```

### View Logs
```bash
# SLURM logs
tail -f em_output_file.out
tail -f em_error_file.err

# Experiment logs
tail -f results/om-em/<run-id>/logs/<model>_log.txt
```

### Common Issues

**OOM Errors:**
- Reduce `--batch-size` or `--score-batch`
- The script auto-reduces batch sizes on OOM

**Model Download Failures:**
- Check HF token permissions
- Verify cache directory has sufficient space
- Check network connectivity

**Judge API Errors:**
- Verify `NVIDIA_API_KEY` is set
- Check rate limits (default: 40 req/min)
- Monitor API quota usage

## Performance Notes

- **Prefetch Stage**: Downloads ~hundreds of GB (depends on selected models)
- **Sweep Stage**: ~10-30 minutes per model (varies by size)
- **Judge Evaluation**: ~1-2 hours for 200+ CSV files
- **Recommended**: Use SSD cache for faster model loading

## Citation

If using this code, please cite the StrongREJECT paper:

```bibtex
@article{strongreject2024,
  title={StrongREJECT: A Benchmark for Evaluating Harmfulness in Language Models},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project uses the StrongREJECT framework. Please check the original repository for license information.

## Contact

For questions or issues, contact: trisanu.bhar@research.iiit.ac.in
