import os
import sys
import gzip
import json
import re
import hashlib
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def clean_text(text):
    if not text:
        return ""
    # Remove HTML tags and normalize whitespace
    text = re.sub(r'<[^>]+>', '', text)
    text = " ".join(text.split())
    return text

def process_file(file_idx, temp_dir):
    url = f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed26n{file_idx:04d}.xml.gz"
    local_path = os.path.join(temp_dir, f"pubmed26n{file_idx:04d}.xml.gz")
    
    # 1. Download
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=60) as response, open(local_path, 'wb') as out_file:
            out_file.write(response.read())
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        return file_idx, []
        
    # 2. Parse XML
    articles = []
    try:
        with gzip.open(local_path, 'rb') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            for article_elem in root.findall('.//PubmedArticle'):
                # Title
                title_elem = article_elem.find('.//ArticleTitle')
                if title_elem is None:
                    continue
                title = clean_text("".join(title_elem.itertext()))
                
                # Abstract
                abstract_elem = article_elem.find('.//Abstract')
                if abstract_elem is None:
                    continue
                
                abstract_texts = []
                for text_elem in abstract_elem.findall('.//AbstractText'):
                    txt = "".join(text_elem.itertext())
                    if txt:
                        abstract_texts.append(clean_text(txt))
                
                abstract_text = " ".join(abstract_texts)
                if title and abstract_text:
                    articles.append((title, abstract_text))
    except Exception as e:
        pass
    finally:
        # 3. Clean up immediately to save disk space
        if os.path.exists(local_path):
            os.remove(local_path)
            
    return file_idx, articles

def main():
    parser = argparse.ArgumentParser(description="Fast parallel ingestion of PubMed baseline.")
    parser.add_argument("--limit", type=int, default=1000000, help="Max unique samples (0 for all)")
    parser.add_argument("--output-prefix", type=str, default="/home/jovyan/data/pubmed", help="Output prefix")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="Validation split ratio")
    parser.add_argument("--workers", type=int, default=64, help="Number of parallel worker processes")
    args = parser.parse_args()

    max_samples = args.limit
    output_dir = os.path.dirname(args.output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temp directory for downloads
    temp_dir = os.path.join(output_dir, "tmp_xml")
    os.makedirs(temp_dir, exist_ok=True)
    
    train_path = f"{args.output_prefix}_train.jsonl"
    val_path = f"{args.output_prefix}_val.jsonl"
    
    print(f"--- Fast Streaming PubMed (Limit: {max_samples if max_samples > 0 else 'ALL'}) --- ")
    print(f"--- Split: {100*(1-args.val_ratio):.1f}% Train / {100*args.val_ratio:.1f}% Val --- ")
    print(f"--- Running in Parallel with {args.workers} Workers ---")
    
    seen_titles = set()
    total_written = 0
    train_count = 0
    val_count = 0
    
    file_indices = list(range(1, 1335))
    
    with open(train_path, "w") as f_train, open(val_path, "w") as f_val:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_file, idx, temp_dir): idx 
                for idx in file_indices
            }
            
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Processing XML files")
            for future in pbar:
                if max_samples > 0 and total_written >= max_samples:
                    # Cancel remaining tasks to exit quickly
                    for f in futures:
                        f.cancel()
                    break
                    
                file_idx = futures[future]
                try:
                    _, articles = future.result()
                    
                    for title, abstract in articles:
                        if max_samples > 0 and total_written >= max_samples:
                            break
                            
                        title_lower = title.lower()
                        if title_lower in seen_titles:
                            continue
                        seen_titles.add(title_lower)
                        
                        full_block = f"TITLE: {title}\nABSTRACT: {abstract}"
                        line = json.dumps({"text": full_block}) + "\n"
                        
                        h = int(hashlib.md5(title_lower.encode()).hexdigest(), 16)
                        if (h % 100) < (args.val_ratio * 100):
                            f_val.write(line)
                            val_count += 1
                        else:
                            f_train.write(line)
                            train_count += 1
                            
                        total_written += 1
                        
                    pbar.set_postfix({"written": total_written})
                except Exception as e:
                    print(f"\n⚠️ Error processing file index {file_idx}: {e}")
                    
    # Clean up temp dir
    try:
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    except Exception:
        pass
        
    print(f"\n✅ Success: Split {total_written} unique medical records into:")
    print(f" - Train: {train_path} ({train_count} records)")
    print(f" - Val:   {val_path} ({val_count} records)")

if __name__ == "__main__":
    main()
