import numpy as np
import json
import os
import matplotlib.pyplot as plt

# ABSOLUTE DIRECT PATHS
base_path = r"C:\Communications Test\Communications-Subsystem\data\cubesat_dataset\phase3_coding"

# LIST ALL SAMPLE PATHS EXPLICITLY
sample_paths = [
    # Convolutional samples
    rf"{base_path}\convolutional\snr_8db\sample_000",
    rf"{base_path}\convolutional\snr_8db\sample_001", 
]

def debug_metadata(sample_path):
    """Debug function to see what's in the metadata"""
    meta_file = os.path.join(sample_path, "meta.json")
    
    if not os.path.exists(meta_file):
        print(f"  ERROR: meta.json not found at {meta_file}")
        return
    
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Metadata for {sample_path}:")
    print(f"  Keys: {list(metadata.keys())}")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

def main():
    print("=" * 60)
    print("DEBUG METADATA CHECK")
    print("=" * 60)
    
    for sample_path in sample_paths:
        if os.path.exists(sample_path):
            debug_metadata(sample_path)
            print("-" * 50)

if __name__ == "__main__":
    main()