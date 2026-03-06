"""
Offline INT8 per-channel weight quantization → compressed-tensors format.

Input:  FP16/BF16 model directory (safetensors)
Output: INT8 quantized model directory (compressed-tensors, compatible with vllm/omni-infer)

Usage:
  python3 quantize_safetensors_int8.py --model /path/to/fp16-model --output /path/to/output

No GPU required. No model code required. Pure tensor math on safetensors.

Quantization:
  - Weights: INT8, symmetric, per-output-channel (RTN)
  - Activations: config only (dynamic per-token at inference time)
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Tensors matching these patterns are NOT quantized
SKIP_PATTERNS = ["embed", "norm", "lm_head"]


def should_quantize(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(".weight"):
        return False
    if tensor.ndim != 2:
        return False
    for pattern in SKIP_PATTERNS:
        if pattern in name:
            return False
    return True


def quantize_per_channel(weight: torch.Tensor):
    """RTN per-output-channel symmetric INT8 quantization.

    weight shape: [out_features, in_features]
    returns: (int8_weight, scale)
      int8_weight: [out_features, in_features] torch.int8
      scale:       [out_features, 1]           torch.bfloat16
    """
    # per-output-channel: amax along input dim (dim=1)
    scale = weight.abs().amax(dim=1, keepdim=True).float() / 127.0
    scale = scale.clamp(min=1e-10)
    int8_weight = (weight.float() / scale).round().clamp(-128, 127).to(torch.int8)
    scale = scale.to(torch.bfloat16)
    return int8_weight, scale


def build_quantization_config(ignore_list):
    return {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "format": "int-quantized",
                "targets": ["Linear"],
                "weights": {
                    "type": "int",
                    "num_bits": 8,
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                    "observer": "minmax",
                    "actorder": None,
                    "group_size": None,
                    "block_structure": None,
                    "observer_kwargs": {},
                },
                "input_activations": {
                    "type": "int",
                    "num_bits": 8,
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                    "actorder": None,
                    "group_size": None,
                    "block_structure": None,
                    "observer": None,
                    "observer_kwargs": {},
                },
                "output_activations": None,
            }
        },
        "ignore": ignore_list,
        "sparsity_config": {},
        "transform_config": {},
        "global_compression_ratio": None,
        "kv_cache_scheme": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Offline INT8 per-channel quantization (compressed-tensors format)"
    )
    parser.add_argument("--model", required=True, help="FP16/BF16 model directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=["lm_head"],
        help="Layer names to skip (default: lm_head)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")

    print(f"Model: {model_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(st_files)} safetensors file(s)")

    total_quantized = 0
    total_skipped = 0
    new_index_map = {}

    for st_file in st_files:
        print(f"\n--- {st_file.name} ---")
        tensors = load_file(str(st_file))
        output_tensors = {}

        for name in sorted(tensors.keys()):
            tensor = tensors[name]
            if should_quantize(name, tensor):
                int8_w, scale = quantize_per_channel(tensor)
                output_tensors[name] = int8_w
                scale_name = name.replace(".weight", ".weight_scale")
                output_tensors[scale_name] = scale
                total_quantized += 1
                print(f"  [Q] {name}: {list(tensor.shape)} {tensor.dtype} → int8 + scale{list(scale.shape)}")
            else:
                output_tensors[name] = tensor
                total_skipped += 1

        out_path = output_dir / st_file.name
        save_file(output_tensors, str(out_path))
        print(f"  Saved → {out_path}")

        for tname in output_tensors:
            new_index_map[tname] = st_file.name

    # Copy non-safetensors files
    for f in model_dir.iterdir():
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            continue
        dst = output_dir / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)

    # Regenerate index.json if sharded
    if len(st_files) > 1:
        index_path = model_dir / "model.safetensors.index.json"
        metadata = {}
        if index_path.exists():
            with open(index_path) as f:
                metadata = json.load(f).get("metadata", {})
        with open(output_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": metadata, "weight_map": new_index_map}, f, indent=2)

    # Update config.json
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        config["quantization_config"] = build_quantization_config(args.ignore)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"\nUpdated config.json with quantization_config")

    print(f"\n{'='*50}")
    print(f"Quantized: {total_quantized} layers, Skipped: {total_skipped} tensors")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
