# Production Usage:
# python3 examples/gemma3/utils/ingest_pubmed.py --limit 0 --output /home/jovyan/data/pubmed_full.jsonl

import json
import sys
import os
import re
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

def ingest(max_samples, output_path):
    print(f"--- Streaming PubMed 2024 from HuggingFace (Limit: {max_samples if max_samples > 0 else 'ALL'}) --- ")
    print("--- Deduplication Active (Title-based) --- ")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Explicitly using the '2024' config name found in pubmed.py
        ds = load_dataset("ncbi/pubmed", "2024", streaming=True, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    count = 0
    skipped = 0
    seen_titles = set()
    
    with open(output_path, "w") as f:
        # total=None when streaming all, as we don't know the exact count in advance
        for i, entry in enumerate(tqdm(ds, total=max_samples if max_samples > 0 else None)):
            if max_samples > 0 and count >= max_samples:
                break
            
            # Navigating the hierarchy defined in the official pubmed.py script
            article = entry.get('MedlineCitation', {}).get('Article', {})
            title = clean_text(article.get('ArticleTitle', ''))
            
            # --- DEDUPLICATION LOGIC ---
            if not title or title.lower() in seen_titles:
                skipped += 1
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
                f.write(json.dumps({"text": full_block}) + "\n")
                count += 1

    print(f"\nSuccess: Ingested {count} unique medical records.")
    print(f"Skipped {skipped} duplicates or empty entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PubMed abstracts.")
    parser.add_argument("--limit", type=int, default=1000000, help="Max unique samples (0 for all)")
    parser.add_argument("--output", type=str, default="/home/jovyan/data/pubmed_abstracts.jsonl", help="Output path")
    args = parser.parse_args()
    ingest(args.limit, args.output)
