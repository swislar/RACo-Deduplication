import re
import csv
import sys
from simhash import Simhash, SimhashIndex
from tqdm import tqdm
import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

def deduplicate_and_save(args):
    
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file
    K_THRESHOLD = args.threshold

    print("Starting Pass 1: Building TF-IDF model...")

    def text_generator(input_file):
        """A generator to yield texts, saving memory."""
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            try:
                next(reader)  
            except StopIteration:
                return  
            for row in reader:
                if len(row) >= 2:
                    yield row[1]  

    vectorizer = TfidfVectorizer(stop_words='english')
    
    vectorizer.fit(text_generator(INPUT_FILE))
    
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"TF-IDF model built. Vocabulary size: {len(feature_names)}")

    print(f"Starting Pass 2: Deduplicating with Simhash (k={K_THRESHOLD})...")
    
    index = SimhashIndex([], k=K_THRESHOLD)
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
                
                tfidf_vector = vectorizer.transform([text])
                
                features_with_weights = []
                coo_matrix = tfidf_vector.tocoo()
                
                for _, col_index, value in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
                    feature_name = feature_names[col_index]
                    features_with_weights.append((feature_name, value))
                
                if not features_with_weights:
                    malformed_count += 1
                    continue
                    
                doc_hash = Simhash(features_with_weights)
                
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
        print(f"Skipped {malformed_count} malformed or empty rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplication of corpus using Simhash and TF-IDF")
    
    parser.add_argument('--threshold', type=int, default=3, help="Simhash k-threshold (hamming distance)")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input tsv file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output tsv file")
    
    args = parser.parse_args()
    
    deduplicate_and_save(args)