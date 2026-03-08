"""
Local version of om-em-strongreject-sweep
Run StrongREJECT evaluation on HuggingFace models locally
"""

import os
import re
import json
import gc
import datetime
import argparse
from typing import Optional, List, Dict, Any

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
from datasets import load_dataset
import pytz

# ============================================================================
# Configuration
# ============================================================================

APP_NAME = "om-em-strongreject-sweep"

# Local paths (instead of Modal volumes)
RESULTS_ROOT = "./results"
CACHE_ROOT = "/ssd_scratch/anu"

# HF org + variants
ORG = "ModelOrganismsForEM"
VARIANTS = {"bad-medical-advice", "risky-financial-advice", "extreme-sports"}

# Environment setup - hardcoded cache paths
os.environ["HF_HOME"] = CACHE_ROOT
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{CACHE_ROOT}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_ROOT}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_ROOT}/datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================================
# Helper Functions
# ============================================================================


def safe_slug(s: str) -> str:
    """Convert a string to a safe filename slug.

    Replaces any non-alphanumeric characters (except . - _) with hyphens.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s)


def check_disk_space(path: str, required_gb: float = 10.0) -> bool:
    """Check if sufficient disk space is available."""
    try:
        import shutil

        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        if free_gb < required_gb:
            print(
                f"WARNING: Low disk space! Free: {free_gb:.1f}GB, Recommended: {required_gb}GB"
            )
            return False
        return True
    except Exception as e:
        print(f"Warning: Could not check disk space: {e}")
        return True


def validate_cached_result(df: pd.DataFrame, expected_prompts: int) -> bool:
    """Validate cached result CSV has expected structure and valid data."""
    required_cols = [
        "strongreject_score",
        "refused_stringmatch",
        "score_ge_threshold",
        "response",
        "prompt",
    ]

    # Check columns exist
    if not all(col in df.columns for col in required_cols):
        return False

    # Check row count matches
    if len(df) != expected_prompts:
        return False

    # Check for missing/invalid data
    if df["strongreject_score"].isna().any():
        return False

    # Check scores are in valid range [0, 1]
    scores = df["strongreject_score"]
    if (scores < 0).any() or (scores > 1).any():
        return False

    # Check for empty responses
    if df["response"].isna().any() or (df["response"].str.strip() == "").any():
        return False

    return True


def validate_responses(responses: List[str]) -> Dict[str, Any]:
    """Validate generated responses and return diagnostics."""
    empty_count = sum(1 for r in responses if not r or not r.strip())
    avg_length = np.mean([len(r) for r in responses])

    return {
        "total": len(responses),
        "empty_count": empty_count,
        "empty_pct": empty_count / len(responses) if responses else 0,
        "avg_length": avg_length,
        "has_issues": empty_count > 0,
    }


def validate_scores(scores: List[float]) -> Dict[str, Any]:
    """Validate StrongREJECT scores are in valid range."""
    scores_arr = np.array(scores)
    out_of_range = np.sum((scores_arr < 0) | (scores_arr > 1))

    return {
        "total": len(scores),
        "out_of_range_count": out_of_range,
        "out_of_range_pct": out_of_range / len(scores) if scores else 0,
        "min": float(np.min(scores_arr)) if len(scores) > 0 else None,
        "max": float(np.max(scores_arr)) if len(scores) > 0 else None,
        "has_issues": out_of_range > 0,
    }


def parse_size_b(text: str):
    """Extract model size in billions from text (e.g., '14B' -> 14.0)."""
    m = re.search(r"(\d+(?:\.\d+)?)B", text or "")
    return float(m.group(1)) if m else None


def now_run_id_ist() -> str:
    """Generate a timestamp-based run ID in IST timezone."""
    tz = pytz.timezone("Asia/Kolkata")
    return datetime.datetime.now(tz).strftime("%Y-%m-%d_%H%M%S_IST")


def ensure_hf_token_env() -> str:
    """Ensure HuggingFace token is set in environment variables.

    Returns:
        str: The HuggingFace token.

    Raises:
        RuntimeError: If no token is found in environment.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF token missing. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable."
        )
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    return token


def mkdirs(base_dir: str):
    """Create standard directory structure for results."""
    for sub in ["inventory", "manifest", "scored", "logs", "plots"]:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)


# ---------- Robust HF/Transformers/PEFT wrappers (token vs use_auth_token) ----------
def tok_from_pretrained(model_id: str, token: str, **kwargs):
    """Load tokenizer with fallback for token/use_auth_token parameter naming."""
    try:
        return AutoTokenizer.from_pretrained(
            model_id, token=token, cache_dir=CACHE_ROOT, **kwargs
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(
            model_id, use_auth_token=token, cache_dir=CACHE_ROOT, **kwargs
        )


def model_from_pretrained(model_id: str, token: str, **kwargs):
    """Load model with fallback for token/use_auth_token parameter naming."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id, token=token, cache_dir=CACHE_ROOT, **kwargs
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_id, use_auth_token=token, cache_dir=CACHE_ROOT, **kwargs
        )


def peft_config_from_pretrained(adapter_id: str, token: str):
    """Load PEFT adapter config with fallback for token/use_auth_token parameter naming."""
    try:
        return PeftConfig.from_pretrained(adapter_id, token=token, cache_dir=CACHE_ROOT)
    except TypeError:
        return PeftConfig.from_pretrained(
            adapter_id, use_auth_token=token, cache_dir=CACHE_ROOT
        )


def peft_from_pretrained(base_model, adapter_id: str, token: str, **kwargs):
    """Load PEFT adapter onto base model with fallback for token/use_auth_token parameter naming."""
    try:
        return PeftModel.from_pretrained(
            base_model, adapter_id, token=token, cache_dir=CACHE_ROOT, **kwargs
        )
    except TypeError:
        return PeftModel.from_pretrained(
            base_model, adapter_id, use_auth_token=token, cache_dir=CACHE_ROOT, **kwargs
        )


def peft_load_adapter(peft_model, adapter_id: str, adapter_name: str, token: str):
    """Load an additional adapter onto existing PEFT model with fallback for token parameter naming."""
    try:
        return peft_model.load_adapter(
            adapter_id, adapter_name=adapter_name, token=token
        )
    except TypeError:
        return peft_model.load_adapter(
            adapter_id, adapter_name=adapter_name, use_auth_token=token
        )


# ---------- StrongREJECT dataset ----------
def load_strongreject_dataset(token: str):
    """Load StrongREJECT dataset and extract prompts and categories.

    Returns:
        tuple: (prompts, categories, metadata_dict)
    """
    try:
        ds_all = load_dataset(
            "walledai/StrongREJECT", token=token, cache_dir=f"{CACHE_ROOT}/datasets"
        )
    except TypeError:
        ds_all = load_dataset(
            "walledai/StrongREJECT",
            use_auth_token=token,
            cache_dir=f"{CACHE_ROOT}/datasets",
        )

    if hasattr(ds_all, "keys"):
        split = "train" if "train" in ds_all else list(ds_all.keys())[0]
        ds = ds_all[split]
    else:
        ds = ds_all

    cols = list(ds.column_names)
    prompt_col = (
        "prompt"
        if "prompt" in cols
        else ("forbidden_prompt" if "forbidden_prompt" in cols else cols[0])
    )
    cat_col = "category" if "category" in cols else None

    prompts = ds[prompt_col]
    cats = ds[cat_col] if cat_col else [None] * len(ds)
    return (
        prompts,
        cats,
        {"prompt_col": prompt_col, "cat_col": cat_col, "n": len(ds), "cols": cols},
    )


# ---------- Prompt formatting ----------
def redact(text: str, max_chars: int = 240) -> str:
    """Truncate text to max_chars length for display purposes."""
    t = " ".join(str(text).split())
    return (t[:max_chars] + " …") if len(t) > max_chars else t


def build_chat_prompts(tokenizer, prompts, system_prompt: str):
    """Format prompts using tokenizer's chat template if available.

    Returns:
        list: Formatted prompts ready for generation.
    """
    out = []
    for p in prompts:
        try:
            if getattr(tokenizer, "chat_template", None):
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p},
                ]
                out.append(
                    tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                )
            else:
                out.append(p)
        except Exception:
            out.append(p)
    return out


# ---------- Generation ----------
def generate_all(
    model,
    tokenizer,
    formatted_prompts,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
):
    """Generate responses for all prompts with automatic batch size reduction on OOM.

    Returns:
        list: Generated text responses.
    """

    @torch.inference_mode()
    def _gen_batch(batch_prompts):
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        out = model.generate(**enc, **gen_kwargs)
        prompt_len = enc["input_ids"].shape[1]
        gen = out[:, prompt_len:]
        return tokenizer.batch_decode(gen, skip_special_tokens=True)

    responses = []
    i = 0
    bs = batch_size
    total_prompts = len(formatted_prompts)

    print(f"    Starting generation: {total_prompts} prompts, batch_size={bs}")
    pbar = tqdm(
        total=total_prompts,
        desc="generate",
        leave=True,
        position=1,
        ncols=100,
        file=None,
    )

    while i < len(formatted_prompts):
        try:
            batch = formatted_prompts[i : i + bs]
            responses.extend(_gen_batch(batch))
            i += bs
            pbar.update(len(batch))
            # Manual progress print every 50 prompts
            if i % 50 == 0 or i >= total_prompts:
                print(f"    Generated: {i}/{total_prompts} prompts", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"    OOM: reducing batch size to {bs}")
                continue
            raise

    pbar.close()
    print(f"    Generation complete: {len(responses)} responses")
    return responses


# ---------- StrongREJECT scoring ----------
def score_strongreject(prompts, responses, score_batch: int, max_resp_len: int):
    """Score responses using StrongREJECT finetuned judge model.

    Returns:
        list: Scores in range [0, 1] where higher means more harmful/non-compliant.
    """
    from strong_reject.evaluate import strongreject_finetuned

    scores = []
    total_batches = (len(prompts) + score_batch - 1) // score_batch
    print(f"    Starting scoring: {len(prompts)} responses in {total_batches} batches")

    # Initial batch size that may be reduced on OOM
    current_batch = score_batch

    for batch_idx, i in enumerate(
        tqdm(
            range(0, len(prompts), score_batch),
            desc="judge:strongreject_finetuned",
            leave=True,
            position=1,
            ncols=100,
        )
    ):
        # Process with dynamic batch sizing
        batch_start = i
        while batch_start < min(i + score_batch, len(prompts)):
            try:
                batch_end = min(
                    batch_start + current_batch, i + score_batch, len(prompts)
                )
                bp = prompts[batch_start:batch_end]
                br = responses[batch_start:batch_end]

                out = strongreject_finetuned(
                    {"forbidden_prompt": bp, "response": br},
                    max_response_length=max_resp_len,
                )
                scores.extend(out["score"])
                batch_start = batch_end

                # Clear cache after each batch to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and current_batch > 1:
                    # OOM: reduce batch size and retry
                    torch.cuda.empty_cache()
                    current_batch = max(1, current_batch // 2)
                    print(
                        f"\n    OOM in scoring: reducing judge batch size to {current_batch}"
                    )
                    continue
                else:
                    raise

        # Manual progress print
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
            print(f"    Scored: {batch_idx + 1}/{total_batches} batches", flush=True)

    print(f"    Scoring complete: {len(scores)} scores")
    return scores


def refusal_flags(prompts, responses):
    """Check if responses contain refusal patterns using string matching.

    Returns:
        list: Binary flags (1=refusal, 0=compliance).
    """
    from strong_reject.evaluate import string_matching

    out = []
    for p, r in zip(prompts, responses):
        s = string_matching(forbidden_prompt=p, response=r)["score"]
        out.append(1 if s == 0 else 0)  # refusal iff harmfulness==0
    return out


# ============================================================================
# Prefetch and Inventory
# ============================================================================


def prefetch_and_inventory(
    run_id: str, max_size_b: float, limit_bases: int, limit_prompts: int
):
    """
    List models from HF org, filter adapters, create inventory and manifest,
    and prefetch models/dataset into cache.
    """
    print(f"[prefetch_and_inventory] run_id={run_id}")

    token = ensure_hf_token_env()

    base_dir = os.path.join(RESULTS_ROOT, "om-em", run_id)
    mkdirs(base_dir)

    api = HfApi(token=token)
    models = api.list_models(author=ORG, limit=300, full=False)

    rows = []
    selected = []

    print(f"Scanning models from {ORG}...")
    for m in tqdm(list(models), desc="Scanning models"):
        model_id = m.modelId
        repo_name = model_id.split("/", 1)[1]
        _base, variant = (repo_name.split("_", 1) + ["base"])[:2]

        try:
            files = api.list_repo_files(repo_id=model_id, repo_type="model")
        except Exception as e:
            rows.append(
                {
                    "model_id": model_id,
                    "repo_name": repo_name,
                    "kind": "error",
                    "error": str(e),
                }
            )
            continue

        if "adapter_config.json" not in files:
            rows.append(
                {
                    "model_id": model_id,
                    "repo_name": repo_name,
                    "kind": "non_adapter",
                    "error": "",
                }
            )
            continue

        try:
            cfg = peft_config_from_pretrained(model_id, token)
            base_model_id = cfg.base_model_name_or_path
        except Exception as e:
            rows.append(
                {
                    "model_id": model_id,
                    "repo_name": repo_name,
                    "kind": "adapter",
                    "error": f"peft_config_failed: {e}",
                }
            )
            continue

        size_b = parse_size_b(base_model_id)
        is_selected = (
            (variant in VARIANTS)
            and (size_b is not None)
            and (14.0 <= size_b <= max_size_b)
        )

        row = dict(
            model_id=model_id,
            repo_name=repo_name,
            kind="adapter",
            base_model_id=base_model_id,
            size_b=size_b,
            variant=variant,
            selected=is_selected,
            error="",
        )
        rows.append(row)
        if is_selected:
            selected.append(row)

    inv = (
        pd.DataFrame(rows)
        .sort_values(["size_b", "repo_name"], na_position="last")
        .reset_index(drop=True)
    )
    inv_path = os.path.join(base_dir, "inventory", f"inventory__{run_id}.csv")
    inv.to_csv(inv_path, index=False)
    print(f"Inventory saved: {inv_path}")

    base_models = sorted({r["base_model_id"] for r in selected})
    if limit_bases and limit_bases > 0:
        base_models = base_models[:limit_bases]
    selected = [r for r in selected if r["base_model_id"] in set(base_models)]

    manifest = {
        "run_id": run_id,
        "max_size_b": max_size_b,
        "variants": sorted(list(VARIANTS)),
        "base_models": base_models,
        "adapters": [
            {
                "model_id": r["model_id"],
                "base_model_id": r["base_model_id"],
                "variant": r["variant"],
                "size_b": r["size_b"],
            }
            for r in selected
        ],
        "limit_bases": limit_bases,
        "limit_prompts": limit_prompts,
    }
    man_path = os.path.join(base_dir, "manifest", f"manifest__{run_id}.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {man_path}")
    print(f"  Base models: {len(base_models)}")
    print(f"  Adapters: {len(selected)}")

    # Prefetch dataset (pull into HF datasets cache)
    print("Loading StrongREJECT dataset...")
    prompts, _cats, meta = load_strongreject_dataset(token)
    if limit_prompts and limit_prompts > 0:
        prompts = prompts[:limit_prompts]
    print(f"  Loaded {len(prompts)} prompts")

    # Prefetch base models + adapters into HF cache
    to_prefetch = list(base_models) + [r["model_id"] for r in selected]
    print(f"Prefetching {len(to_prefetch)} repos from HuggingFace...")
    for repo in tqdm(to_prefetch, desc="Prefetch HF repos"):
        snapshot_download(
            repo_id=repo, repo_type="model", token=token, cache_dir=CACHE_ROOT
        )

    # Prefetch StrongREJECT finetuned judge
    print("Warming up StrongREJECT judge...")
    try:
        from strong_reject.evaluate import strongreject_finetuned

        _ = strongreject_finetuned(
            {"forbidden_prompt": ["warmup"], "response": ["warmup"]},
            max_response_length=16,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load StrongREJECT finetuned judge. "
            "Make sure your HF token has access to the judge dependencies (often Gemma-2-2B gated). "
            f"Original error: {e}"
        )

    print("[prefetch_and_inventory] Complete!\n")


# ============================================================================
# Sweep
# ============================================================================


def run_sweep(
    run_id: str,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    score_batch: int,
    max_response_length: int,
    score_threshold: float,
    include_baselines: bool,
    force_rerun: bool = False,
):
    """
    Load models and adapters, generate responses, score with StrongREJECT,
    and produce summary CSV and plots.

    Memory optimization: Models use device_map="auto" to distribute across 4 GPUs.
    Generation model is completely deleted before loading judge model to ensure
    only one model is in GPU memory at a time.

    Checkpoint/Resume: Automatically skips models that have already been evaluated
    (unless force_rerun=True).
    """
    print(f"[run_sweep] run_id={run_id}")
    print("  Note: Using device_map='auto' to distribute models across 4 GPUs")
    if force_rerun:
        print("  Note: Force re-run enabled - ignoring cached results")

    token = ensure_hf_token_env()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected. Evaluation will be very slow!")

    base_dir = os.path.join(RESULTS_ROOT, "om-em", run_id)
    mkdirs(base_dir)

    man_path = os.path.join(base_dir, "manifest", f"manifest__{run_id}.json")
    if not os.path.exists(man_path):
        raise RuntimeError(
            f"Missing manifest: {man_path}. Run with --stage prefetch first for this run_id."
        )

    with open(man_path, "r") as f:
        manifest = json.load(f)

    print("Loading StrongREJECT dataset...")
    prompts, cats, meta = load_strongreject_dataset(token)
    limit_prompts = manifest.get("limit_prompts", 0)
    if isinstance(limit_prompts, int) and limit_prompts > 0:
        prompts = prompts[:limit_prompts]
        cats = cats[:limit_prompts]
    print(f"  Using {len(prompts)} prompts")

    system_prompt = "You are a helpful assistant."

    # Group adapters by base model
    adapters_by_base: Dict[str, List[Dict[str, Any]]] = {}
    for a in manifest["adapters"]:
        adapters_by_base.setdefault(a["base_model_id"], []).append(a)
    for base_id in adapters_by_base:
        adapters_by_base[base_id] = sorted(
            adapters_by_base[base_id], key=lambda x: x["variant"]
        )

    scored_dir = os.path.join(base_dir, "scored")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(scored_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Check disk space
    if not check_disk_space(base_dir, required_gb=5.0):
        print("  Continuing anyway, but may run out of space...")

    summary_rows = []
    expected_files = []
    failed_evals = []  # Track failures for reporting
    pending_scoring = []  # Track evaluations that need scoring

    for base_model_id in tqdm(manifest["base_models"], desc="Base models", position=0):
        print(f"\n=== Processing base model: {base_model_id} ===")

        # Check if all evaluations for this base model already exist
        adapters = adapters_by_base.get(base_model_id, [])
        size_b = parse_size_b(base_model_id) or float("nan")

        expected_evals = []
        if include_baselines:
            expected_evals.append((base_model_id, "baseline", size_b, None))
        for a in adapters:
            expected_evals.append(
                (a["model_id"], a["variant"], a["size_b"], a["model_id"])
            )

        # Check which evals are missing
        missing_evals = []
        for model_id, variant, sz, adapter_id in expected_evals:
            fname = (
                "__".join(
                    [
                        safe_slug(model_id),
                        safe_slug(variant),
                        "strongreject",
                        run_id,
                    ]
                )
                + ".csv"
            )
            # Skip cache check if force_rerun is enabled
            if force_rerun or not os.path.exists(os.path.join(scored_dir, fname)):
                missing_evals.append((model_id, variant, sz, adapter_id))

        # If all evals exist, skip model loading entirely
        if not missing_evals:
            print(f"  ✓✓ All evaluations already complete for {base_model_id}")
            print(f"     Loading {len(expected_evals)} cached results...")
            # Load cached results and populate summary_rows
            for model_id, variant, sz, adapter_id in expected_evals:
                fname = (
                    "__".join(
                        [
                            safe_slug(model_id),
                            safe_slug(variant),
                            "strongreject",
                            run_id,
                        ]
                    )
                    + ".csv"
                )
                out_path = os.path.join(scored_dir, fname)
                expected_files.append(fname)
                try:
                    cached_df = pd.read_csv(out_path)

                    if not validate_cached_result(cached_df, len(prompts)):
                        print(
                            f"     Warning: Cached result {fname} is invalid, will need to re-run this base model"
                        )
                        missing_evals.append((model_id, variant, sz, adapter_id))
                        continue

                    mean_score = float(cached_df["strongreject_score"].mean())
                    refusal_rate = float(cached_df["refused_stringmatch"].mean())
                    summary_rows.append(
                        {
                            "run_id": run_id,
                            "model_id": model_id,
                            "base_model_id": base_model_id,
                            "adapter_id": adapter_id,
                            "variant": variant,
                            "size_b": sz,
                            "n": len(cached_df),
                            "refusal_rate": refusal_rate,
                            "mean_score": mean_score,
                            "pct_ge_threshold": float(
                                cached_df["score_ge_threshold"].mean()
                            ),
                            "csv": out_path,
                        }
                    )
                except Exception as e:
                    print(f"     Warning: Failed to load {fname}: {e}")
                    missing_evals.append((model_id, variant, sz, adapter_id))

            # If we found invalid cached results, we need to re-run
            if missing_evals:
                print(
                    f"     Found {len(missing_evals)} invalid cached results, will re-run this base model"
                )
            else:
                continue  # Skip to next base model

        print(f"  Found {len(missing_evals)}/{len(expected_evals)} evaluations to run")

        tok = tok_from_pretrained(
            base_model_id, token, trust_remote_code=True, use_fast=True
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Load model with device_map="auto" to distribute across 4 GPUs automatically
        base_model = model_from_pretrained(
            base_model_id,
            token,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        base_model.eval()

        formatted = build_chat_prompts(tok, prompts, system_prompt)

        def generate_responses(
            model,
            condition_id: str,
            variant: str,
            size_b: float,
            adapter_id: Optional[str],
        ):
            """Generate responses and save to intermediate file. Scoring happens later."""
            nonlocal pending_scoring, expected_files, summary_rows

            # Check if results already exist (checkpoint/resume functionality)
            fname = (
                "__".join(
                    [
                        safe_slug(condition_id),
                        safe_slug(variant),
                        "strongreject",
                        run_id,
                    ]
                )
                + ".csv"
            )
            out_path = os.path.join(scored_dir, fname)
            expected_files.append(fname)

            if not force_rerun and os.path.exists(out_path):
                print(
                    f"  ✓ Skipping {condition_id} (variant={variant}) - results already exist"
                )
                print(f"    Loading cached results from: {fname}")
                try:
                    # Load existing results and add to summary
                    cached_df = pd.read_csv(out_path)

                    # Validate cached result
                    if not validate_cached_result(cached_df, len(prompts)):
                        print(
                            f"    Warning: Cached result validation failed (corrupt or incomplete)"
                        )
                        print(f"    Re-running evaluation...")
                        # Don't return, let it re-run
                    else:
                        mean_score = float(cached_df["strongreject_score"].mean())
                        refusal_rate = float(cached_df["refused_stringmatch"].mean())

                        summary_rows.append(
                            {
                                "run_id": run_id,
                                "model_id": condition_id,
                                "base_model_id": base_model_id,
                                "adapter_id": adapter_id,
                                "variant": variant,
                                "size_b": size_b,
                                "n": len(cached_df),
                                "refusal_rate": refusal_rate,
                                "mean_score": mean_score,
                                "pct_ge_threshold": float(
                                    cached_df["score_ge_threshold"].mean()
                                ),
                                "csv": out_path,
                            }
                        )
                        print(
                            f"    Cached - Refusal rate: {refusal_rate:.3f}, Mean score: {mean_score:.3f}"
                        )
                        return  # Skip generation and scoring entirely
                except Exception as e:
                    print(f"    Warning: Failed to load cached results: {e}")
                    print(f"    Re-running evaluation...")

            print(f"  Generating responses: {condition_id} (variant={variant})")

            # Check for intermediate responses file
            responses_fname = (
                "__".join(
                    [
                        safe_slug(condition_id),
                        safe_slug(variant),
                        "responses",
                        run_id,
                    ]
                )
                + ".json"
            )
            responses_path = os.path.join(logs_dir, responses_fname)

            if not force_rerun and os.path.exists(responses_path):
                print(f"    Loading cached responses from: {responses_fname}")
                try:
                    with open(responses_path, "r") as f:
                        response_data = json.load(f)
                    responses = response_data["responses"]
                    print(f"    Loaded {len(responses)} cached responses")

                    # Add to pending scoring
                    pending_scoring.append(
                        {
                            "condition_id": condition_id,
                            "variant": variant,
                            "size_b": size_b,
                            "adapter_id": adapter_id,
                            "responses": responses,
                            "out_path": out_path,
                        }
                    )
                    return
                except Exception as e:
                    print(f"    Warning: Failed to load cached responses: {e}")
                    print(f"    Re-generating...")

            responses = generate_all(
                model,
                tok,
                formatted,
                batch_size=batch_size,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

            # Verify response count matches prompt count
            if len(responses) != len(prompts):
                raise RuntimeError(
                    f"Response count mismatch! Expected {len(prompts)}, got {len(responses)}"
                )

            # Validate responses
            response_stats = validate_responses(responses)
            if response_stats["has_issues"]:
                print(
                    f"    WARNING: {response_stats['empty_count']} empty responses detected ({response_stats['empty_pct']:.1%})"
                )
            print(
                f"    Response stats: avg_length={response_stats['avg_length']:.0f} chars"
            )

            # Save responses to intermediate file for later scoring
            print(f"    Saving responses to: {responses_fname}")
            response_data = {
                "condition_id": condition_id,
                "variant": variant,
                "size_b": size_b,
                "adapter_id": adapter_id,
                "responses": responses,
                "metadata": {
                    "base_model_id": base_model_id,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            }
            with open(responses_path, "w") as f:
                json.dump(response_data, f)

            # Add to pending scoring queue
            pending_scoring.append(
                {
                    "condition_id": condition_id,
                    "variant": variant,
                    "size_b": size_b,
                    "adapter_id": adapter_id,
                    "responses": responses,
                    "out_path": out_path,
                }
            )

            print(f"    Responses saved. Scoring will happen after model is unloaded.")

        # End of generate_responses function

        # GENERATION PHASE: Generate all responses for this base model
        print(f"\n  === GENERATION PHASE ===")

        # Baseline
        size_b = parse_size_b(base_model_id) or float("nan")
        if include_baselines:
            try:
                generate_responses(base_model, base_model_id, "baseline", size_b, None)
            except Exception as e:
                print(f"  ✗ ERROR generating baseline {base_model_id}: {e}")
                failed_evals.append(
                    {"model_id": base_model_id, "variant": "baseline", "error": str(e)}
                )
                # Continue with other evaluations

        # Adapters (load once, switch adapters without reloading base)
        adapters = adapters_by_base.get(base_model_id, [])
        if adapters:
            first = adapters[0]
            peft_model = peft_from_pretrained(
                base_model,
                first["model_id"],
                token,
                adapter_name=first["variant"],
                is_trainable=False,
            )
            peft_model.set_adapter(first["variant"])
            peft_model.eval()
            try:
                generate_responses(
                    peft_model,
                    first["model_id"],
                    first["variant"],
                    first["size_b"],
                    first["model_id"],
                )
            except Exception as e:
                print(f"  ✗ ERROR generating {first['model_id']}: {e}")
                failed_evals.append(
                    {
                        "model_id": first["model_id"],
                        "variant": first["variant"],
                        "error": str(e),
                    }
                )

            for a in adapters[1:]:
                peft_load_adapter(
                    peft_model, a["model_id"], adapter_name=a["variant"], token=token
                )
                peft_model.set_adapter(a["variant"])
                peft_model.eval()
                try:
                    generate_responses(
                        peft_model,
                        a["model_id"],
                        a["variant"],
                        a["size_b"],
                        a["model_id"],
                    )
                except Exception as e:
                    print(f"  ✗ ERROR generating {a['model_id']}: {e}")
                    failed_evals.append(
                        {
                            "model_id": a["model_id"],
                            "variant": a["variant"],
                            "error": str(e),
                        }
                    )

            del peft_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Completely unload generation model before scoring
        print(f"\n  === Unloading generation model ===")
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"  Generation model freed from memory")

        # SCORING PHASE: Score all generated responses for this base model
        if pending_scoring:
            print(f"\n  === SCORING PHASE ===")
            print(f"  Scoring {len(pending_scoring)} evaluations")

            # Warm up judge once per base model
            print("  Loading StrongREJECT judge...")
            from strong_reject.evaluate import strongreject_finetuned

            _ = strongreject_finetuned(
                {"forbidden_prompt": ["warmup"], "response": ["warmup"]},
                max_response_length=16,
            )
            print("  Judge ready")

            for eval_data in pending_scoring:
                condition_id = eval_data["condition_id"]
                variant = eval_data["variant"]
                size_b = eval_data["size_b"]
                adapter_id = eval_data["adapter_id"]
                responses = eval_data["responses"]
                out_path = eval_data["out_path"]

                try:
                    print(f"  Scoring: {condition_id} (variant={variant})")

                    refused = refusal_flags(prompts, responses)
                    scores = score_strongreject(
                        prompts,
                        responses,
                        score_batch=score_batch,
                        max_resp_len=max_response_length,
                    )

                    # Verify score count matches prompt count
                    if len(scores) != len(prompts):
                        raise RuntimeError(
                            f"Score count mismatch! Expected {len(prompts)}, got {len(scores)}"
                        )

                    # Validate scores
                    score_stats = validate_scores(scores)
                    if score_stats["has_issues"]:
                        print(
                            f"    WARNING: {score_stats['out_of_range_count']} scores out of [0,1] range!"
                        )
                    print(
                        f"    Score range: [{score_stats['min']:.3f}, {score_stats['max']:.3f}]"
                    )

                    out_df = pd.DataFrame(
                        {
                            "run_id": run_id,
                            "benchmark": "strongreject",
                            "prompt_idx": list(range(len(prompts))),
                            "category": cats,
                            "prompt": prompts,
                            "response": responses,
                            "response_redacted": [redact(r) for r in responses],
                            "refused_stringmatch": refused,
                            "strongreject_score": scores,
                            "score_ge_threshold": [
                                1 if s >= score_threshold else 0 for s in scores
                            ],
                            "model_id": condition_id,
                            "base_model_id": base_model_id,
                            "adapter_id": adapter_id,
                            "variant": variant,
                            "size_b": size_b,
                            "max_new_tokens": max_new_tokens,
                            "do_sample": do_sample,
                            "temperature": temperature,
                            "top_p": top_p,
                        }
                    )

                    # Save results
                    out_df.to_csv(out_path, index=False)

                    top = out_df.sort_values(
                        "strongreject_score", ascending=False
                    ).head(25)
                    prev_path = os.path.join(
                        logs_dir,
                        "__".join(
                            [
                                safe_slug(condition_id),
                                safe_slug(variant),
                                "strongreject",
                                "misaligned_preview",
                                run_id,
                            ]
                        )
                        + ".csv",
                    )
                    top[
                        [
                            "prompt_idx",
                            "category",
                            "refused_stringmatch",
                            "strongreject_score",
                            "prompt",
                            "response_redacted",
                        ]
                    ].to_csv(prev_path, index=False)

                    mean_score = float(np.mean(scores))
                    refusal_rate = float(np.mean(refused))
                    print(
                        f"    Refusal rate: {refusal_rate:.3f}, Mean score: {mean_score:.3f}"
                    )

                    summary_rows.append(
                        {
                            "run_id": run_id,
                            "model_id": condition_id,
                            "base_model_id": base_model_id,
                            "adapter_id": adapter_id,
                            "variant": variant,
                            "size_b": size_b,
                            "n": len(out_df),
                            "refusal_rate": refusal_rate,
                            "mean_score": mean_score,
                            "pct_ge_threshold": float(
                                np.mean([s >= score_threshold for s in scores])
                            ),
                            "csv": out_path,
                        }
                    )

                except Exception as e:
                    print(f"  ✗ ERROR scoring {condition_id}: {e}")
                    failed_evals.append(
                        {
                            "model_id": condition_id,
                            "variant": variant,
                            "error": f"scoring_failed: {e}",
                        }
                    )

            # Clear pending scoring for this base model
            pending_scoring = []

            # Aggressive cleanup after scoring
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print(f"  Judge model freed from memory")

        # Save intermediate summary after each base model
        if summary_rows:
            interim_summary = pd.DataFrame(summary_rows).sort_values(
                ["size_b", "variant", "model_id"]
            )
            interim_path = os.path.join(base_dir, f"summary__{run_id}_interim.csv")
            interim_summary.to_csv(interim_path, index=False)

    print("\n=== Generating summary and plots ===")

    # Report failed evaluations
    if failed_evals:
        print(f"\n⚠️  {len(failed_evals)} evaluation(s) failed:")
        for fail in failed_evals:
            print(
                f"   - {fail['model_id']} ({fail['variant']}): {fail['error'][:100]}..."
            )
        # Save failed evaluations log
        fail_log_path = os.path.join(logs_dir, f"failed_evaluations__{run_id}.json")
        with open(fail_log_path, "w") as f:
            json.dump(failed_evals, f, indent=2)
        print(f"   Saved failure log: {fail_log_path}")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["size_b", "variant", "model_id"]
    )
    summary_path = os.path.join(base_dir, f"summary__{run_id}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    # Plots
    plt.figure()
    for variant in sorted(summary_df["variant"].unique()):
        sub = summary_df[summary_df["variant"] == variant].dropna(subset=["size_b"])
        if len(sub) == 0:
            continue
        plt.scatter(sub["size_b"], sub["mean_score"], label=variant)
    plt.title("StrongREJECT: mean score vs model size (B)")
    plt.xlabel("Model size (B)")
    plt.ylabel("Mean StrongREJECT score (0..1)")
    plt.legend()
    plot1 = os.path.join(base_dir, "plots", f"mean_score_vs_size__{run_id}.png")
    plt.savefig(plot1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot1}")

    plt.figure()
    for variant in sorted(summary_df["variant"].unique()):
        sub = summary_df[summary_df["variant"] == variant].dropna(subset=["size_b"])
        if len(sub) == 0:
            continue
        plt.scatter(sub["size_b"], sub["refusal_rate"], label=variant)
    plt.title("StrongREJECT: refusal rate vs model size (B)")
    plt.xlabel("Model size (B)")
    plt.ylabel("Refusal rate (string_matching)")
    plt.legend()
    plot2 = os.path.join(base_dir, "plots", f"refusal_vs_size__{run_id}.png")
    plt.savefig(plot2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot2}")

    # Matrix check
    scored_files = set(os.listdir(scored_dir))
    matrix = [{"expected_file": f, "exists": f in scored_files} for f in expected_files]
    matrix_path = os.path.join(base_dir, "inventory", f"matrix_check__{run_id}.csv")
    pd.DataFrame(matrix).to_csv(matrix_path, index=False)

    missing_count = sum(1 for m in matrix if not m["exists"])
    if missing_count > 0:
        print(
            f"\n⚠️  Warning: {missing_count}/{len(expected_files)} expected result files are missing!"
        )
        print(f"   See: {matrix_path}")

    print(f"[run_sweep] Complete!\n")


# ============================================================================
# Main
# ============================================================================


def main():
    """Main entry point for the StrongREJECT evaluation sweep."""
    parser = argparse.ArgumentParser(description="Local StrongREJECT evaluation sweep")

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["prefetch", "sweep", "all"],
        help="Stage to run: prefetch | sweep | all",
    )
    parser.add_argument(
        "--run-id", type=str, default="", help="Run ID (empty => auto-generate)"
    )
    parser.add_argument(
        "--max-size-b",
        type=float,
        default=24.0,
        help="Max model size in billions (selects 14B <= size <= max_size_b)",
    )
    parser.add_argument(
        "--limit-bases",
        type=int,
        default=0,
        help="Limit number of base models (0 => no limit)",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=0,
        help="Limit number of prompts (0 => no limit)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for generation"
    )
    parser.add_argument(
        "--max-input-tokens", type=int, default=2048, help="Max input tokens"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max new tokens to generate"
    )
    parser.add_argument(
        "--do-sample", action="store_true", help="Enable sampling (else greedy)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")

    parser.add_argument(
        "--score-batch",
        type=int,
        default=8,
        help="Batch size for scoring (reduced default for memory)",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=512,
        help="Max response length for judge",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Score threshold for classification",
    )

    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip baseline (non-adapter) evaluations",
    )

    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-evaluation even if cached results exist",
    )

    args = parser.parse_args()

    stage = args.stage.lower().strip()
    run_id = args.run_id if args.run_id else now_run_id_ist()

    print(f"{'='*70}")
    print(f"{APP_NAME}")
    print(f"Stage: {stage}")
    print(f"Run ID: {run_id}")
    print(f"{'='*70}\n")

    if stage in {"prefetch", "all"}:
        prefetch_and_inventory(
            run_id=run_id,
            max_size_b=args.max_size_b,
            limit_bases=args.limit_bases,
            limit_prompts=args.limit_prompts,
        )

    if stage in {"sweep", "all"}:
        run_sweep(
            run_id=run_id,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            score_batch=args.score_batch,
            max_response_length=args.max_response_length,
            score_threshold=args.score_threshold,
            include_baselines=(not args.no_baselines),
            force_rerun=args.force_rerun,
        )

    print(f"\n{'='*70}")
    print(f"✅ run_id = {run_id}")
    print(f"Results: {os.path.join(RESULTS_ROOT, 'om-em', run_id)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
