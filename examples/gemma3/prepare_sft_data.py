import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_data(max_seq_length=16384, model_id="google/gemma-3-1b-it"):
    print(f"--- Loading LDJnr/Capybara dataset (subset :100) ---")
    # Using the suggested subsetting syntax
    dataset = load_dataset("LDJnr/Capybara", split="train[:100]")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    sft_data = []
    skipped_long = 0
    
    # 1. Process Capybara samples
    print("--- Processing multi-turn Capybara samples ---")
    for sample in dataset:
        messages = []
        # Handle Capybara multi-turn format: [{'input': '...', 'output': '...'}, ...]
        for turn in sample['conversation']:
            if turn.get('input'):
                messages.append({"role": "user", "content": turn['input']})
            if turn.get('output'):
                messages.append({"role": "model", "content": turn['output']})
        
        # Validation: Check sequence length with chat template
        if not messages:
            continue
            
        rendered_chat = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized_chat = tokenizer(rendered_chat).input_ids
        if len(tokenized_chat) <= max_seq_length:
            sft_data.append({"messages": messages})
        else:
            skipped_long += 1

    print(f"--- Skipped {skipped_long} samples exceeding {max_seq_length} tokens ---")

    # 2. Inject "Gold" Medical Samples
    print("--- Injecting Gold Medical Samples ---")
    medical_qa = [
        {
            "user": "What are the differential diagnoses for a patient with acute shortness of breath and chest pain?",
            "model": "Differential diagnosis includes myocardial infarction, pulmonary embolism, and aortic dissection."
        },
        {
            "user": "What initial tests should be ordered for suspected acute coronary syndrome?",
            "model": "Stat EKG and troponin levels should be ordered immediately."
        },
        {
            "user": "What are common physical findings in a patient with pulmonary embolism?",
            "model": "Common findings include tachypnea, tachycardia, and potentially decreased oxygen saturation (e.g., 92% on room air)."
        },
        {
            "user": "What is the standard emergency management for acute chest pain?",
            "model": "Emergency management involves supplemental oxygen, aspirin, and sublingual nitroglycerin."
        },
        {
            "user": "Which imaging modality is preferred to rule out pulmonary embolism?",
            "model": "A CT pulmonary angiogram (CTPA) is the planned diagnostic imaging to rule out pulmonary embolism."
        }
    ]

    for qa in medical_qa:
        messages = [
            {"role": "user", "content": qa["user"]},
            {"role": "model", "content": qa["model"]}
        ]
        sft_data.append({"messages": messages})

    # 3. Save to JSONL
    output_path = "capybara_sft_subset.jsonl"
    print(f"--- Saving {len(sft_data)} samples to {output_path} ---")
    with open(output_path, "w") as f:
        for entry in sft_data:
            f.write(json.dumps(entry) + "\n")

    # 4. Local Validation
    print("\n--- Local Data Validation Summary ---")
    print(f"Total Samples: {len(sft_data)}")
    if len(sft_data) > 0:
        # Check first sample structure
        first_sample = sft_data[0]
        print(f"First sample turn count: {len(first_sample['messages'])}")
        print(f"Structure check: {'PASS' if 'role' in first_sample['messages'][0] else 'FAIL'}")
        
        # Check max tokens in the final set
        max_tokens_found = max(len(tokenizer(tokenizer.apply_chat_template(s['messages'], tokenize=False)).input_ids) for s in sft_data)
        print(f"Max tokens in set: {max_tokens_found} (Limit: {max_seq_length})")
        if max_tokens_found <= max_seq_length:
            print("✓ Sequence length constraint: PASS")
        else:
            print("✗ Sequence length constraint: FAIL")

if __name__ == "__main__":
    prepare_data()
