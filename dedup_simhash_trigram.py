import re
import csv
import sys
from simhash import Simhash, SimhashIndex
from tqdm import tqdm 
import pandas as pd
import argparse

def get_features(s):
    width = 3
    s = s.lower()
    s = re.sub(r'[^\w]+', '', s)
    return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]

def deduplicate_and_save(args):
    
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file 
    K_THRESHOLD = args.threshold

    print(f"Initializing SimhashIndex with k={K_THRESHOLD}...")
    index = SimhashIndex([], k=K_THRESHOLD)

    print(f"Starting deduplication...")
    saved_count = 0
    dupe_count = 0
    malformed_count = 0


    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
        open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.reader(f_in, delimiter='\t')
        writer = csv.writer(f_out, delimiter='\t')
        
        try:
            header = next(reader)
            writer.writerow(header)
            print(f"Header written to {OUTPUT_FILE}")
        except StopIteration:
            print("Error: Input file is empty.")
            return

        for row in tqdm(reader, desc="Processing rows"):
            
            if len(row) >= 2:
                doc_id = row[0]
                text = row[1]
                
                doc_hash = Simhash(get_features(text))
                
                dups = index.get_near_dups(doc_hash)
                
                if len(dups) == 0:
                    index.add(doc_id, doc_hash)
                    writer.writerow(row)
                    saved_count += 1
                else:
                    dupe_count += 1
            else:
                malformed_count += 1

    print("\n--- Deduplication Complete ---")
    print(f"Saved {saved_count} unique documents to {OUTPUT_FILE}")
    print(f"Removed {dupe_count} near-duplicates (k={K_THRESHOLD}).")
    if malformed_count > 0:
        print(f"Skipped {malformed_count} malformed rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplication of corpus")
    
    parser.add_argument('--threshold', type=int, default=3, help="Batch size for the Language Model")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input tsv file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output tsv file")
    
    args = parser.parse_args()
    
    deduplicate_and_save(args)