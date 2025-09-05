import numpy as np

def simple_demodulate(symbols):
    """Basic demodulation without dependencies"""
    if hasattr(symbols[0], 'real'):
        real_symbols = [s.real for s in symbols]
    else:
        real_symbols = symbols
    return [1 if s >= 0 else 0 for s in real_symbols]

# Test
test_symbols = [1.0, -0.5, 0.3, -1.2, 0.8, -0.1]
bits = simple_demodulate(test_symbols)
print("Test symbols:", test_symbols)
print("Decoded bits:", bits)
print("✅ Basic demodulation works!")

# Save this as debug_structure.py and run it
import os
from pathlib import Path

def explore_directory_structure(base_path):
    base_path = Path(base_path)
    print(f"Exploring: {base_path}")
    
    if not base_path.exists():
        print("Path does not exist!")
        return
    
    for root, dirs, files in os.walk(base_path):
        level = root.replace(str(base_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith(('.npy', '.json')):
                print(f"{subindent}{file}")

# Run exploration
base_dir = r"C:\Communications Test\Communications-Subsystem\data"
explore_directory_structure(base_dir)

# Save as check_paths.py and run it
import os

base_path = r"C:\Communications Test\Communications-Subsystem\data\cubesat_dataset\phase3_coding"

paths_to_check = [
    rf"{base_path}\convolutional\snr_8db\sample_000",
    rf"{base_path}\convolutional\snr_8db\sample_001",
]

for path in paths_to_check:
    exists = os.path.exists(path)
    print(f"{'✅' if exists else '❌'} {path}")
    if exists:
        files = os.listdir(path)
        print(f"   Files: {files}")