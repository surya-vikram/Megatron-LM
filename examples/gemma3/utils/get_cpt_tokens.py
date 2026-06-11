import sys
from megatron.core.datasets.indexed_dataset import IndexedDataset
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: get_cpt_tokens.py <data_path_prefix>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    ds = IndexedDataset(data_path)
    total_tokens = np.sum(ds.index.sequence_lengths)
    print(total_tokens)

if __name__ == '__main__':
    main()
