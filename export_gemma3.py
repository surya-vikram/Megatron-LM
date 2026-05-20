import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Robust path handling
base_dir = Path(os.environ.get("HOME", "/root"))
hf_source_path = base_dir / "models/gemma-3-4b-pt-hf"
megatron_path = base_dir / "models/gemma-3-4b-pt-mcore"
raw_export_path = base_dir / "models/gemma-3-4b-pt-roundtrip-hf-raw"
hf_export_path = base_dir / "models/gemma-3-4b-pt-roundtrip-hf"


def export_text_backbone() -> None:
    from megatron.bridge import AutoBridge
    from transformers import AutoConfig

    if raw_export_path.exists():
        shutil.rmtree(raw_export_path)

    print(f"Loading config from {hf_source_path}...")
    config = AutoConfig.from_pretrained(str(hf_source_path), trust_remote_code=True)
    config.architectures = ["Gemma3ForCausalLM"]
    bridge = AutoBridge.from_hf_config(config)

    print(f"Exporting Megatron checkpoint to temporary HF dir: {raw_export_path}")
    bridge.export_ckpt(
        str(megatron_path),
        str(raw_export_path),
    )


def build_shard_map(index_path: Path) -> dict[str, list[str]]:
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shard_map: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in weight_map.items():
        shard_map[shard_name].append(key)
    return dict(shard_map)


def remap_text_key(key: str) -> str:
    if key.startswith("language_model.model."):
        return key
    if key.startswith("model.layers."):
        return key.replace("model.layers.", "language_model.model.layers.", 1)
    if key == "model.embed_tokens.weight":
        return "language_model.model.embed_tokens.weight"
    if key == "model.norm.weight":
        return "language_model.model.norm.weight"
    return key


def merge_full_checkpoint() -> None:
    if hf_export_path.exists():
        shutil.rmtree(hf_export_path)
    hf_export_path.mkdir(parents=True, exist_ok=True)

    print(f"Copying base HF checkpoint skeleton to {hf_export_path}...")
    for item in hf_source_path.iterdir():
        if item.name.startswith("model-") and item.suffix == ".safetensors":
            continue
        if item.name == "model.safetensors.index.json":
            continue
        destination = hf_export_path / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    base_index_path = hf_source_path / "model.safetensors.index.json"
    base_index = json.loads(base_index_path.read_text())

    print("Loading exported text weights...")
    remapped_text_tensors: dict[str, torch.Tensor] = {}
    raw_shards = sorted(raw_export_path.glob("model-*.safetensors"))
    for shard_path in raw_shards:
        shard_tensors = load_file(shard_path)
        for key, value in shard_tensors.items():
            remapped_text_tensors[remap_text_key(key)] = value

    if not remapped_text_tensors:
        raise RuntimeError(f"No text tensors were exported into {raw_export_path}")

    print("Writing full merged safetensor shards...")
    base_shard_map = build_shard_map(base_index_path)
    for shard_name, shard_keys in base_shard_map.items():
        base_shard_path = hf_source_path / shard_name
        merged_shard = load_file(base_shard_path)
        updated_tensors: dict[str, torch.Tensor] = {}
        for key in shard_keys:
            updated_tensors[key] = remapped_text_tensors.get(key, merged_shard[key])
        save_file(updated_tensors, str(hf_export_path / shard_name))

    updated_index = base_index
    updated_index["metadata"] = {
        **updated_index.get("metadata", {}),
        "source_base_checkpoint": str(hf_source_path),
        "source_text_export": str(raw_export_path),
        "merge_strategy": "base_vision_plus_roundtrip_text",
    }
    (hf_export_path / "model.safetensors.index.json").write_text(
        json.dumps(updated_index, indent=2) + "\n"
    )

    config_path = hf_export_path / "config.json"
    config = json.loads(config_path.read_text())
    # Note: Keep the original architecture (ConditionalGeneration) for the merged model
    # config["architectures"] = ["Gemma3ForCausalLM"] 
    config["torch_dtype"] = "bfloat16"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    export_text_backbone()
    merge_full_checkpoint()
    print(f"Successfully exported full roundtrip checkpoint to {hf_export_path}")


if __name__ == "__main__":
    main()
