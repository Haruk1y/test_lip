import argparse
import shutil
import random
import time
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

def get_file_ids(video_dir: Path) -> List[str]:
    """Get all file IDs from video directory"""
    video_files = sorted(video_dir.glob("LFROI_*.mp4"))
    return [f.stem.replace("LFROI_", "") for f in video_files]

def split_dataset(
    file_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float
) -> Tuple[List[str], List[str], List[str]]:
    """Split file IDs into train, validation and test sets"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
    
    random.shuffle(file_ids)
    total = len(file_ids)
    
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_ids = file_ids[:train_size]
    val_ids = file_ids[train_size:train_size + val_size]
    test_ids = file_ids[train_size + val_size:]
    
    return train_ids, val_ids, test_ids

def safe_copy(src_file: Path, dst_file: Path, max_retries: int = 3, retry_delay: float = 1.0):
    """Safely copy file with retries"""
    for attempt in range(max_retries):
        try:
            shutil.copy2(src_file, dst_file)
            return True
        except (BlockingIOError, OSError) as e:
            if attempt == max_retries - 1:
                print(f"\nFailed to copy {src_file} to {dst_file} after {max_retries} attempts: {str(e)}")
                return False
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
    return False

def copy_files(
    file_ids: List[str],
    src_dir: Path,
    dst_dir: Path,
    file_pattern: str,
    desc: str,
    chunk_size: int = 100  # Process files in smaller chunks
):
    """Copy files from source to destination directory with chunking"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files in chunks
    for i in range(0, len(file_ids), chunk_size):
        chunk = file_ids[i:i + chunk_size]
        for file_id in tqdm(chunk, desc=f"Copying {desc} ({i}-{i+len(chunk)}/{len(file_ids)})"):
            src_file = src_dir / file_pattern.format(file_id)
            dst_file = dst_dir / file_pattern.format(file_id)
            
            if not src_file.exists():
                print(f"\nWarning: File not found - {src_file}")
                continue
                
            if dst_file.exists():
                continue  # Skip if already copied
                
            if not safe_copy(src_file, dst_file):
                print(f"\nSkipping file: {file_id}")
                continue
                
        # Small delay between chunks to prevent I/O overload
        time.sleep(0.1)

def main():
    parser = argparse.ArgumentParser(description="Prepare ROHAN dataset")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to raw data directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--splits_dir", type=Path, required=True, help="Path to splits directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of files to process in each chunk")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Ensure directories exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all file IDs
    file_ids = get_file_ids(args.data_dir / "videos")
    print(f"Total number of samples: {len(file_ids)}")
    
    # Split dataset
    train_ids, val_ids, test_ids = split_dataset(
        file_ids,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
    
    print(f"Train set size: {len(train_ids)}")
    print(f"Validation set size: {len(val_ids)}")
    print(f"Test set size: {len(test_ids)}")
    
    # Create splits files
    for split_name, split_ids in [
        ("train", train_ids),
        ("val", val_ids),
        ("test", test_ids)
    ]:
        with open(args.splits_dir / f"{split_name}_list.txt", "w") as f:
            f.write("\n".join(split_ids))
    
    # Copy files for each split
    for split_name, split_ids in [
        ("train", train_ids),
        ("val", val_ids),
        ("test", test_ids)
    ]:
        print(f"\nProcessing {split_name} split...")
        
        # Copy videos
        copy_files(
            split_ids,
            args.data_dir / "videos",
            args.output_dir / split_name / "videos",
            "LFROI_{}.mp4",
            f"{split_name} videos",
            args.chunk_size
        )
        
        # Copy lab files
        copy_files(
            split_ids,
            args.data_dir / "lab",
            args.output_dir / split_name / "lab",
            "{}.lab",
            f"{split_name} lab files",
            args.chunk_size
        )
        
        # Copy landmark files
        copy_files(
            split_ids,
            args.data_dir / "landmarks",
            args.output_dir / split_name / "landmarks",
            "{}_points.csv",
            f"{split_name} landmark files",
            args.chunk_size
        )

if __name__ == "__main__":
    main()