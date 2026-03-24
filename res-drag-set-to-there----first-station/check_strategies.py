import os
import re

data_dir = 'PUT-DATA-THERE'
print("Checking all strategy metadata:")
for subdir in sorted(os.listdir(data_dir)):
    metadata_path = os.path.join(data_dir, subdir, 'metadata.txt')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            content = f.read()
            match = re.search(r'strategy = (\w+)', content)
            elapsed_match = re.search(r'elapsed_seconds = ([\d.]+)', content)
            if match:
                elapsed = elapsed_match.group(1) if elapsed_match else 'N/A'
                print(f'{subdir}: {match.group(1)} ({elapsed}s)')
