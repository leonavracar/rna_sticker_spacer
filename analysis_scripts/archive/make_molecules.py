#!/usr/bin/env python3

"""
Generate molecules.json file with a custom count of molecules of the same length.
The base proportions can be changed below.
Usage:
    python make_molecules.py <save_dir> <length> <count> 

Flag --identical will generate identical sequence with a set count.
Seed can be set with --seed for reproducibility
Saves molecules.json into BASE_DIR/<sweep_dir>.

"""

import numpy as np
import json
import sys
sys.path.append("/groups/goloborodko/projects/leona/validation")
import argparse
from pathlib import Path

# Fixed base directory where all sweep folders live
BASE_DIR = Path("/groups/goloborodko/projects/leona/validation/bulk/output_dir") 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", help= "Name of the sweep dir to save file into inside BASE_DIR")
    parser.add_argument("length", type=int, help="Length (number of nucleotides) of RNA chain")
    parser.add_argument("count", type=int, help="Number of RNA molecules of set length")
    parser.add_argument("--identical", action="store_true", help="Generate identical sequences based on count instead of unique ones")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    save_path = BASE_DIR / args.save_dir
    
    # Set seed for reproducibility
    rng = np.random.default_rng(seed=args.seed)

    # Generate sequence
    if args.identical:
        seq = make_random_seq(args.length, rng)
        data = [{ "sequence": seq.tolist(), "count": args.count }]
    else:
        data = []
        for _ in range(args.count):
            seq = make_random_seq(args.length, rng)
            data.append({ "sequence": seq.tolist(), "count": 1 })
        
    
    filename = f"{args.save_dir}_len{args.length}_count{args.count}_seed{args.seed}.json"

    save_json(data, filename, save_path)

def make_random_seq(len, rng, p_A=0.25, p_U=0.25, p_C=0.25, p_G=0.25):
    """Pass length, random generator"""
    bases = ["A", "U", "C", "G"]
    probs = [p_A, p_U, p_C, p_G]
    return rng.choice(bases, size = len, p=probs)

def save_json(data, filename, save_path):
    filepath = save_path / filename
    with open(filepath, "w") as f:
        json.dump(data, f)
    print(f"File {filename} saved to {save_path}")



if __name__ == "__main__":
    main()