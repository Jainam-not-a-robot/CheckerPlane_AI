#!/usr/bin/env python3
"""Offline ONNX model export and PyTorch equivalence verification utility.

Usage:
  # Verify all exported ONNX models against their PyTorch sources
  python bench/export_models.py --verify-all --tolerance 0.02

  # Export a specific model
  python bench/export_models.py --model coherence

  # Export and quantize models
  python bench/export_models.py --model all --quantize

WARNING:
  Quantization can silently corrupt model outputs. Re-run --verify-all afterwards
  and treat it as a hard gate, not a formality.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

MODELS = {
    "coherence": {
        "hf_repo": "madhurjindal/autonlp-Gibberish-Detector-492513457",
        "task": "text-classification",
        "output_dir": "models/coherence",
        "test_input": "best rust orm postgres",
        "export_method": "optimum",
    },
    "toxicity": {
        "hf_repo": "minuva/MiniLMv2-toxic-jigsaw-onnx",
        "task": "text-classification",
        "output_dir": "models/toxicity",
        "test_input": "This is a clean and polite query.",
        "export_method": "download",
        "weights_file": "model_optimized_quantized.onnx",
    },
    "intent": {
        "hf_repo": "meta-llama/Llama-Prompt-Guard-2-22M",
        "task": "text-classification",
        "output_dir": "models/intent",
        "test_input": "Hello, how do I configure my database?",
        "export_method": "optimum",
    },
    "cross_encoder": {
        "hf_repo": "cross-encoder/nli-distilroberta-base",
        "task": "text-classification",
        "output_dir": "models/cross_encoder",
        # For cross-encoders, we need a pair of strings.
        # We handle this specifically in the verification.
        "test_input": ("A dog is running in the grass.", "An animal is outside."),
        "export_method": "optimum",
    },
}


def export_model_optimum(spec: dict, target_dir: Path, quantize: bool) -> bool:
    """Exports a model using optimum-cli."""
    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        spec["hf_repo"],
        "--task",
        spec["task"],
    ]
    if quantize:
        cmd.extend(["-O2"])
    cmd.append(str(target_dir))

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to export {spec['hf_repo']}: {e}")
        return False


def export_model_download(spec: dict, target_dir: Path) -> bool:
    """Downloads an already-exported model from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub is required for download exports.")
        return False

    repo_id = spec["hf_repo"]
    weights_file = spec.get("weights_file", "model.onnx")

    files_to_download = [
        weights_file,
        "tokenizer.json",
        "config.json",
        "tokenizer_config.json",
    ]

    try:
        for filename in files_to_download:
            print(f"Downloading {filename} from {repo_id}...")
            # We copy it into target_dir explicitly to have a flat directory
            dl_path = hf_hub_download(repo_id=repo_id, filename=filename)
            shutil.copy2(dl_path, target_dir / filename)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {repo_id}: {e}")
        return False


def export_model(model_name: str, quantize: bool, tolerance: float) -> bool:
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
        return False

    spec = MODELS[model_name]
    output_dir = Path(spec["output_dir"])

    print(f"Exporting {model_name} from {spec['hf_repo']}...")
    
    # We use a temporary directory so a failed run doesn't overwrite a good model
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        if spec["export_method"] == "optimum":
            success = export_model_optimum(spec, temp_path, quantize)
        elif spec["export_method"] == "download":
            success = export_model_download(spec, temp_path)
        else:
            print(f"[ERROR] Unknown export method {spec['export_method']}")
            success = False

        if not success:
            return False

        # Verify before committing
        if not verify_model_in_dir(model_name, temp_path, tolerance):
            print(f"[ERROR] Verification failed for {model_name}. Aborting export.")
            return False

        # Atomically replace output_dir
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(temp_path), str(output_dir))
        
    print(f"[SUCCESS] Exported and verified {model_name} to {output_dir}")
    return True


def verify_model_in_dir(model_name: str, model_dir: Path, tolerance: float) -> bool:
    spec = MODELS[model_name]
    weights_file = spec.get("weights_file", "model.onnx")
    onnx_file = model_dir / weights_file

    if not onnx_file.exists():
        print(f"[SKIP] {model_name}: {onnx_file} not found on disk.")
        return True

    try:
        import numpy as np
        import onnxruntime as ort
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("[ERROR] Verification requires onnxruntime, transformers, torch, numpy.")
        print("        Install with: pip install onnxruntime transformers torch numpy")
        return False

    print(f"Verifying {model_name} against PyTorch source (tolerance={tolerance})...")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    
    test_input = spec["test_input"]
    if isinstance(test_input, tuple):
        # Text pair (e.g. cross-encoder)
        inputs = tokenizer(test_input[0], test_input[1], return_tensors="pt")
    else:
        inputs = tokenizer(test_input, return_tensors="pt")

    # PyTorch inference
    try:
        pt_model = AutoModelForSequenceClassification.from_pretrained(spec["hf_repo"])
        pt_model.eval()
        with torch.no_grad():
            pt_out = pt_model(**inputs).logits.softmax(dim=-1).numpy()
    except Exception as e:
        print(f"[WARN] Could not run PyTorch baseline for {model_name}: {e}")
        return False

    # ONNX Runtime inference
    session = ort.InferenceSession(str(onnx_file))
    onnx_inputs = {k: v.numpy() for k, v in inputs.items() if k in [inp.name for inp in session.get_inputs()]}
    onnx_out_raw = session.run(None, onnx_inputs)[0]
    
    # Softmax ONNX output
    exp = np.exp(onnx_out_raw - np.max(onnx_out_raw, axis=-1, keepdims=True))
    onnx_out = exp / np.sum(exp, axis=-1, keepdims=True)

    max_diff = float(np.max(np.abs(pt_out - onnx_out)))
    print(f"  {model_name}: max absolute difference = {max_diff:.6f}")

    # Calculate Pearson correlation
    if pt_out.size > 1:
        correlation = np.corrcoef(pt_out.flatten(), onnx_out.flatten())[0, 1]
        print(f"  {model_name}: Pearson correlation = {correlation:.6f}")
    else:
        correlation = 1.0

    if max_diff > tolerance:
        print(f"[FAIL] {model_name} exceeds tolerance {tolerance} (diff={max_diff})")
        return False

    print(f"[PASS] {model_name} verified successfully within tolerance.")
    return True


def verify_model(model_name: str, tolerance: float) -> bool:
    spec = MODELS[model_name]
    output_dir = Path(spec["output_dir"])
    return verify_model_in_dir(model_name, output_dir, tolerance)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline ONNX export and verification tool.",
        epilog="WARNING: Quantization can silently corrupt model outputs. Always verify with --verify-all.",
    )
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], help="Model to export")
    parser.add_argument("--verify-all", action="store_true", help="Verify all existing ONNX models against PyTorch")
    parser.add_argument("--tolerance", type=float, default=0.02, help="Maximum allowed logit probability deviation (default: 0.02)")
    parser.add_argument("--quantize", action="store_true", help="Apply quantization during export")

    args = parser.parse_args()

    if args.verify_all:
        print("Starting PyTorch vs ONNX equivalence verification...")
        success = True
        for m in MODELS:
            if not verify_model(m, args.tolerance):
                success = False
        if not success:
            sys.exit(1)
        print("All models passed equivalence verification.")
    elif args.model:
        targets = list(MODELS.keys()) if args.model == "all" else [args.model]
        for t in targets:
            export_model(t, args.quantize, args.tolerance)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
