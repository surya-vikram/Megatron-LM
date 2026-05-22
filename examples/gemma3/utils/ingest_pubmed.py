import json
import sys
import os
from datasets import load_dataset
from tqdm import tqdm
import argparse

def ingest(max_samples, output_path):
    print(f"--- Streaming PubMed from HuggingFace (Limit: {max_samples}) --- ")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        ds = load_dataset("ncbi/pubmed", streaming=True, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    count = 0
    with open(output_path, "w") as f:
        for i, entry in enumerate(tqdm(ds, total=max_samples if max_samples > 0 else None)):
            if max_samples > 0 and i >= max_samples:
                break
            
            # Navigate XML structure for Abstracts
            article = entry.get('MedlineCitation', {}).get('Article', {})
            title = article.get('ArticleTitle', '')
            abstract_dict = article.get('Abstract', {})
            abstract_text = ""
            
            if abstract_dict and 'AbstractText' in abstract_dict:
                text_data = abstract_dict['AbstractText']
                if isinstance(text_data, list):
                    abstract_text = " ".join([str(t) for t in text_data if t])
                else:
                    abstract_text = str(text_data)
            
            if title or abstract_text:
                full_text = f"{title}\n\n{abstract_text}"
                f.write(json.dumps({"text": full_text}) + "\n")
                count += 1

    print(f"Successfully ingested {count} samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PubMed abstracts from HuggingFace.")
    parser.add_argument("--limit", type=int, default=1000000, help="Max samples (0 for all)")
    parser.add_argument("--output", type=str, default="/home/jovyan/data/pubmed_abstracts.jsonl", help="Output path")
    args = parser.parse_args()
    
    ingest(args.limit, args.output)
