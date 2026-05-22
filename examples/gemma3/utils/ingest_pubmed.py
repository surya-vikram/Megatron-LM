# Production Usage:
# python3 examples/gemma3/utils/ingest_pubmed.py --limit 0 --output-prefix /home/jovyan/data/pubmed --val-ratio 0.01

import json
import sys
import os
import re
import hashlib
from datasets import load_dataset
from tqdm import tqdm
import argparse

def clean_text(text):
    if not text:
        return ""
    # Remove HTML tags (<b>, <i>, etc.) that the pubmed.py parser might leave
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text

def ingest(max_samples, output_prefix, val_ratio):
    print(f"--- Streaming PubMed 2024 (Limit: {max_samples if max_samples > 0 else 'ALL'}) --- ")
    print(f"--- Split: {100*(1-val_ratio):.1f}% Train / {100*val_ratio:.1f}% Val --- ")
    print("--- Deduplication Active (Title-based) --- ")
    
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    train_path = f"{output_prefix}_train.jsonl"
    val_path = f"{output_prefix}_val.jsonl"
    
    try:
        # Explicitly using the '2024' config name found in pubmed.py
        ds = load_dataset("ncbi/pubmed", "2024", streaming=True, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    count = 0
    seen_titles = set()
    
    with open(train_path, "w") as f_train, open(val_path, "w") as f_val:
        for i, entry in enumerate(tqdm(ds, total=max_samples if max_samples > 0 else None)):
            if max_samples > 0 and count >= max_samples:
                break
            
            # Navigating the hierarchy defined in the official pubmed.py script
            article = entry.get('MedlineCitation', {}).get('Article', {})
            title = clean_text(article.get('ArticleTitle', ''))
            
            # --- DEDUPLICATION LOGIC ---
            if not title or title.lower() in seen_titles:
                continue
            
            seen_titles.add(title.lower())
            
            abstract_data = article.get('Abstract', {}).get('AbstractText', '')
            
            # Handling the 'Special Case' from pubmed.py where AbstractText can be a list
            if isinstance(abstract_data, list):
                abstract_text = " ".join([clean_text(str(t)) for t in abstract_data if t])
            else:
                abstract_text = clean_text(str(abstract_data))
            
            if abstract_text:
                full_block = f"TITLE: {title}\nABSTRACT: {abstract_text}"
                line = json.dumps({"text": full_block}) + "\n"
                
                # Deterministic split based on title hash
                h = int(hashlib.md5(title.lower().encode()).hexdigest(), 16)
                if (h % 100) < (val_ratio * 100):
                    f_val.write(line)
                else:
                    f_train.write(line)
                
                count += 1

    print(f"\nSuccess: Split {count} unique medical records into:")
    print(f" - Train: {train_path}")
    print(f" - Val:   {val_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PubMed abstracts with Train/Val split.")
    parser.add_argument("--limit", type=int, default=1000000, help="Max unique samples (0 for all)")
    parser.add_argument("--output-prefix", type=str, default="/home/jovyan/data/pubmed", help="Output prefix")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="Validation split ratio (e.g. 0.01 for 1%)")
    args = parser.parse_args()
    
    ingest(args.limit, args.output_prefix, args.val_ratio)
