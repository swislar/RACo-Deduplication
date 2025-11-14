from sentence_transformers import SentenceTransformer
from sentence_transformers.util import community_detection
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Set
import argparse

class SBERTDeduplicator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.85):
        """
        Initialize SBERT deduplicator.
        
        Args:
            model_name: SBERT model to use
            similarity_threshold: Cosine similarity threshold (0-1) for considering duplicates
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        
    def deduplicate(self, texts: List[str], return_indices: bool = False) -> Tuple[List[str], List[int]]:
        """
        Remove near-duplicate texts from corpus.
        
        Args:
            texts: List of text strings to deduplicate
            return_indices: Whether to return indices of kept texts
            
        Returns:
            Tuple of (deduplicated_texts, kept_indices)
        """
        if not texts:
            return [], []
        
        print(f"Encoding {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        print("Identifying duplicates using community detection...")
        clusters = community_detection(embeddings, 
                                       threshold=self.similarity_threshold, 
                                       min_community_size=2)
        
        print(f"Found {len(clusters)} duplicate clusters.")
        
        to_remove = set()
        for cluster in clusters:
            for idx in cluster[1:]: 
                to_remove.add(idx)
        
        kept_indices = [i for i in range(len(texts)) if i not in to_remove]
        deduplicated_texts = [texts[i] for i in kept_indices]
        
        print(f"Removed {len(to_remove)} duplicates. Kept {len(deduplicated_texts)} unique texts.")
        
        if return_indices:
            return deduplicated_texts, kept_indices
        return deduplicated_texts, kept_indices

def deduplicate_and_save(args):
    """
    Deduplicate corpus from TSV file and save to output file.
    Processes the entire corpus at once for accurate deduplication.
    """
    MODEL_NAME = args.model_name
    THRESHOLD = args.threshold
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file
    BATCH_SIZE = args.batch_size
    TEXT_COLUMN = args.text_column
    
    print(f"Initializing model: {MODEL_NAME}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    
    deduplicator = SBERTDeduplicator(model_name=MODEL_NAME, similarity_threshold=THRESHOLD)
    
    all_rows = []
    all_texts = []
    header = None
    malformed_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        reader = csv.reader(f_in, delimiter='\t')
        
        try:
            header = next(reader)
            print(f"Header: {header}")
        except StopIteration:
            print("Error: Input file is empty.")
            return
        
        for line_num, row in enumerate(reader, start=2):
            try:
                if len(row) < TEXT_COLUMN + 1:
                    malformed_count += 1
                    continue
                
                text = row[TEXT_COLUMN].strip()
                if text:  
                    all_rows.append(row)
                    all_texts.append(text)
                    
            except Exception as e:
                print(f"Error on line {line_num}: {e}")
                malformed_count += 1
                continue
    
    print(f"Read {len(all_texts)} valid rows (skipped {malformed_count} malformed rows)")
    
    if not all_texts:
        print("Error: No valid texts found in input file.")
        return
    
    _, kept_indices = deduplicator.deduplicate(all_texts, return_indices=True)

    saved_count = 0
    dupe_count = len(all_rows) - len(kept_indices)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        
        writer.writerow(header)
        
        for idx in kept_indices:
            writer.writerow(all_rows[idx])
            saved_count += 1
    
    print("\n=== Deduplication Summary ===")
    print(f"Total input rows: {len(all_rows)}")
    print(f"Malformed rows skipped: {malformed_count}")
    print(f"Duplicates removed: {dupe_count} ({dupe_count/len(all_rows)*100:.2f}%)")
    print(f"Unique rows saved: {saved_count}")
    print(f"Output written to: {OUTPUT_FILE}")

def deduplicate_and_save_streaming(args):
    """
    Memory-efficient version: processes corpus in batches.
    Note: This may miss some duplicates across batches.
    For accurate deduplication of large files, consider using approximate methods like LSH.
    """
    MODEL_NAME = args.model_name
    THRESHOLD = args.threshold
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file
    BATCH_SIZE = args.batch_size
    TEXT_COLUMN = args.text_column
    
    print(f"Processing in batches of {BATCH_SIZE} (streaming mode)")
    print("WARNING: Batch processing may miss duplicates across batch boundaries.")
    
    deduplicator = SBERTDeduplicator(model_name=MODEL_NAME, similarity_threshold=THRESHOLD)
    
    saved_count = 0
    dupe_count = 0
    malformed_count = 0
    total_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.reader(f_in, delimiter='\t')
        writer = csv.writer(f_out, delimiter='\t')
        
        try:
            header = next(reader)
            writer.writerow(header)
            print(f"Header: {header}")
        except StopIteration:
            print("Error: Input file is empty.")
            return
        
        batch_rows = []
        batch_texts = []
        
        for line_num, row in enumerate(reader, start=2):
            try:
                if len(row) < TEXT_COLUMN + 1:
                    malformed_count += 1
                    continue
                
                text = row[TEXT_COLUMN].strip()
                if not text:
                    continue
                
                batch_rows.append(row)
                batch_texts.append(text)
                total_count += 1
                
                if len(batch_texts) >= BATCH_SIZE:
                    print(f"\nProcessing batch at line {line_num}...")
                    _, kept_indices = deduplicator.deduplicate(batch_texts, return_indices=True)
                    
                    for idx in kept_indices:
                        writer.writerow(batch_rows[idx])
                        saved_count += 1
                    
                    dupe_count += len(batch_texts) - len(kept_indices)
                    
                    batch_rows = []
                    batch_texts = []
                    
            except Exception as e:
                print(f"Error on line {line_num}: {e}")
                malformed_count += 1
                continue
        
        if batch_texts:
            print(f"\nProcessing final batch...")
            _, kept_indices = deduplicator.deduplicate(batch_texts, return_indices=True)
            
            for idx in kept_indices:
                writer.writerow(batch_rows[idx])
                saved_count += 1
            
            dupe_count += len(batch_texts) - len(kept_indices)
    
    print("\n=== Deduplication Summary ===")
    print(f"Total input rows: {total_count}")
    print(f"Malformed rows skipped: {malformed_count}")
    print(f"Duplicates removed: {dupe_count} ({dupe_count/total_count*100:.2f}% if total_count else 0)")
    print(f"Unique rows saved: {saved_count}")
    print(f"Output written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplication of corpus using Sentence-BERT")
    
    parser.add_argument('--model_name', type=str, default='all-mpnet-base-v2', 
                        help="Sentence-BERT model name")
    parser.add_argument('--threshold', type=float, default=0.9, 
                        help="Cosine similarity threshold (0-1)")
    parser.add_argument('--input_file', type=str, required=True, 
                        help="Path to the input TSV file")
    parser.add_argument('--output_file', type=str, required=True, 
                        help="Path to the output TSV file")
    parser.add_argument('--text_column', type=int, default=1, 
                        help="Column index containing text to deduplicate (0-indexed)")
    parser.add_argument('--batch_size', type=int, default=10000, 
                        help="Batch size for streaming mode")
    parser.add_argument('--streaming', action='store_true', 
                        help="Use streaming mode for large files (less accurate)")
    
    args = parser.parse_args()
    
    if args.streaming:
        deduplicate_and_save_streaming(args)
    else:
        deduplicate_and_save(args)