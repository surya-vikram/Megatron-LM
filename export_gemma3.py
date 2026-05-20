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


def remap_key_for_conditional_gen(key: str) -> str:
    """
    Remaps keys to match the expectation of Gemma3ForConditionalGeneration.
    Based on the 'MISSING' keys report in the logs.
    """
    # 1. Handle Text Backbone
    if key.startswith("model."):
        # e.g. model.layers.0... -> model.language_model.model.layers.0...
        return "model.language_model." + key
    
    if key.startswith("language_model.model."):
        # e.g. language_model.model.layers.0... -> model.language_model.model.layers.0...
        return "model." + key

    # 2. Handle Vision Tower & Projector from original HF
    # The original keys in safetensors were like: vision_tower.vision_model.encoder...
    # The expected keys are like: model.vision_tower.encoder...
    
    if key.startswith("vision_tower.vision_model."):
        return key.replace("vision_tower.vision_model.", "model.vision_tower.", 1)
    
    if key.startswith("vision_tower."):
        return "model." + key
        
    if key.startswith("multi_modal_projector."):
        return "model." + key

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
    raw_text_tensors: dict[str, torch.Tensor] = {}
    raw_shards = sorted(raw_export_path.glob("model-*.safetensors"))
    for shard_path in raw_shards:
        raw_text_tensors.update(load_file(shard_path))

    if not raw_text_tensors:
        raise RuntimeError(f"No text tensors were exported into {raw_export_path}")

    print("Writing full merged safetensor shards...")
    base_shard_map = build_shard_map(base_index_path)
    
    # We will build a new index because keys are remapped
    new_weight_map = {}

    for shard_name, shard_keys in base_shard_map.items():
        base_shard_path = hf_source_path / shard_name
        original_shard = load_file(base_shard_path)
        
        updated_shard: dict[str, torch.Tensor] = {}
        
        # We must iterate over what the NEW model expects
        for original_key in shard_keys:
            new_key = remap_key_for_conditional_gen(original_key)
            
            # If it's a text key, try to find it in the exported Megatron tensors
            # (Note: raw_text_tensors keys might need remapping too)
            # Megatron-Bridge export_ckpt with Gemma3ForCausalLM gives keys starting with 'model.'
            if original_key.startswith("language_model.model."):
                mcore_key = original_key.replace("language_model.model.", "model.", 1)
                if mcore_key in raw_text_tensors:
                    updated_shard[new_key] = raw_text_tensors[mcore_key]
                else:
                    updated_shard[new_key] = original_shard[original_key]
            else:
                # It's vision or projector, take from original
                updated_shard[new_key] = original_shard[original_key]
            
            new_weight_map[new_key] = shard_name
            
        save_file(updated_shard, str(hf_export_path / shard_name))

    updated_index = base_index
    updated_index["weight_map"] = new_weight_map
    updated_index["metadata"] = {
        **updated_index.get("metadata", {}),
        "source_base_checkpoint": str(hf_source_path),
        "source_text_export": str(raw_export_path),
        "merge_strategy": "base_vision_plus_roundtrip_text_remapped",
    }
    (hf_export_path / "model.safetensors.index.json").write_text(
        json.dumps(updated_index, indent=2) + "\n"
    )

    config_path = hf_export_path / "config.json"
    config = json.loads(config_path.read_text())
    config["torch_dtype"] = "bfloat16"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    export_text_backbone()
    merge_full_checkpoint()
    print(f"Successfully exported full roundtrip checkpoint to {hf_export_path}")


if __name__ == "__main__":
    main()
