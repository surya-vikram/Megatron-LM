import json
import argparse
import random
from datasets import load_dataset

def ingest_medical_sft_simple(output_train, output_val, val_ratio=0.05, limit=None):
    print(f"--- Loading FreedomIntelligence/medical-o1-reasoning-SFT (Config: 'en') ---")
    dataset = load_dataset('FreedomIntelligence/medical-o1-reasoning-SFT', 'en', split='train', streaming=True)
    
    train_count = 0
    val_count = 0
    skipped_count = 0
    
    print(f"--- Processing and Formatting Samples (1 Case Per Line - Multi-turn Safe) ---")
    
    with open(output_train, 'w', encoding='utf-8') as f_train, \
         open(output_val, 'w', encoding='utf-8') as f_val:
        
        for i, sample in enumerate(dataset):
            if limit and i >= limit:
                break
            
            question = sample.get('Question', '').strip()
            cot = sample.get('Complex_CoT', '').strip()
            response = sample.get('Response', '').strip()
            
            if not question or not cot or not response:
                skipped_count += 1
                continue
            
            # Pure Gemma 3 Native SFT format: User -> Assistant (CoT wrapped in think tags)
            formatted_sample = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"<think>\n{cot}\n</think>\n\n{response}"}
                ]
            }
            
            # Deterministic split
            if random.random() < val_ratio:
                f_val.write(json.dumps(formatted_sample) + '\n')
                val_count += 1
            else:
                f_train.write(json.dumps(formatted_sample) + '\n')
                train_count += 1
                
            if (train_count + val_count) % 1000 == 0:
                print(f"Progress: {train_count + val_count} samples processed...")

    print(f"\n--- Ingestion Complete (Simple & Correct) ---")
    print(f"Total Medical Cases: {train_count + val_count}")
    print(f"Files saved to: {output_train} and {output_val}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Medical SFT data (Simple 1-per-line)")
    parser.add_argument("--output-train", type=str, default="/home/jovyan/data/sft_train.jsonl")
    parser.add_argument("--output-val", type=str, default="/home/jovyan/data/sft_val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    ingest_medical_sft_simple(args.output_train, args.output_val, args.val_ratio, args.limit)
