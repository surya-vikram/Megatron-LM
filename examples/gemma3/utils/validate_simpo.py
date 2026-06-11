import json
import sys
from pathlib import Path

def validate_simpo(input_path):
    print(f'--- Validating SimPO JSONL: {input_path} ---')
    path = Path(input_path)
    if not path.exists():
        print(f'Error: File {input_path} not found.')
        sys.exit(1)
    
    count = 0
    try:
        with path.open('r') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                if 'chosen' not in data or 'rejected' not in data:
                    print(f'Error on line {i+1}: Missing "chosen" or "rejected" keys. Keys found: {list(data.keys())}')
                    sys.exit(1)
                
                if not isinstance(data['chosen'], list) or not isinstance(data['rejected'], list):
                    print(f'Error on line {i+1}: "chosen" and "rejected" must be lists of messages.')
                    sys.exit(1)

                for msg in data['chosen'] + data['rejected']:
                    if 'role' not in msg or 'content' not in msg:
                        print(f'Error on line {i+1}: Message missing "role" or "content".')
                        sys.exit(1)
                count += 1
        print(f'Successfully validated {count} SimPO samples.')
    except Exception as e:
        print(f'Error during validation: {e}')
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python validate_simpo.py <path_to_jsonl>')
        sys.exit(1)
    validate_simpo(sys.argv[1])
