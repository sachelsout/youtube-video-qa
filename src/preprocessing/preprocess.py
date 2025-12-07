"""
Main Preprocessing Pipeline
Combines cleaning and chunking into a single pipeline.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

from src.preprocessing.clean_transcript import (
    clean_text,
    clean_transcript_with_timestamps,
    merge_segments
)
from src.preprocessing.chunk_transcript import (
    chunk_text,
    chunk_text_with_timestamps
)


def preprocess_transcript(
    video_id: str,
    max_tokens: int = 250,
    overlap: int = 50,
    merge_gap: float = 0.0,
    input_dir: str = "src/data/raw",
    output_dir: str = "src/data/processed"
) -> Dict:
    """
    Complete preprocessing pipeline: clean and chunk a transcript.
    
    Args:
        video_id: YouTube video ID
        max_tokens: Maximum tokens per chunk
        overlap: Token overlap between chunks
        merge_gap: Maximum gap in seconds to merge segments
        input_dir: Directory containing raw transcripts
        output_dir: Directory to save processed transcripts
        
    Returns:
        Dictionary with preprocessing results and metadata
    """
    # Load raw transcript
    input_path = Path(input_dir) / f"{video_id}.json"
    
    if not input_path.exists():
        raise FileNotFoundError(f"Raw transcript not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    raw_transcript = data.get('transcript', [])
    
    if not raw_transcript:
        raise ValueError(f"No transcript data found in {input_path}")
    
    # Step 1: Clean transcript
    cleaned_lines = clean_text(raw_transcript)
    cleaned_segments = clean_transcript_with_timestamps(raw_transcript)

    # Step 2: Merge segments based on auto merge_gap
    if merge_gap > 0:
        merged_segments = merge_segments(cleaned_segments, max_gap=merge_gap)
    else:
        merged_segments = cleaned_segments

    # Step 3: Create chunks
    chunks = chunk_text(cleaned_lines, max_tokens=max_tokens, overlap=overlap)
    chunks_with_timestamps = chunk_text_with_timestamps(
        merged_segments,
        max_tokens=max_tokens,
        overlap=overlap
    )
    
    # Prepare output
    result = {
        'video_id': video_id,
        'metadata': {
            'original_segments': len(raw_transcript),
            'cleaned_segments': len(cleaned_lines),
            'merged_segments': len(merged_segments),
            'num_chunks': len(chunks),
            'max_tokens': max_tokens,
            'overlap': overlap,
            'merge_gap': merge_gap
        },
        'chunks': chunks_with_timestamps  # Use timestamp version as default
    }
    
    # Save to file
    output_path = Path(output_dir) / f"{video_id}_chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result


def main():
    """Command-line interface for preprocessing."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.preprocessing.preprocess <video_id> [max_tokens] [overlap]")
        print("Example: python -m src.preprocessing.preprocess dQw4w9WgXcQ 250 50")
        sys.exit(1)
    
    video_id = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    print(f"Preprocessing transcript for video: {video_id}")
    print(f"Settings: max_tokens={max_tokens}, overlap={overlap}")
    
    try:
        result = preprocess_transcript(
            video_id=video_id,
            max_tokens=max_tokens,
            overlap=overlap
        )
        
        output_path = Path("src/data/processed") / f"{video_id}_chunks.json"
        
        print(f"\n✓ Preprocessing complete!")
        print(f"  Original segments: {result['metadata']['original_segments']}")
        print(f"  Cleaned segments: {result['metadata']['cleaned_segments']}")
        print(f"  Final chunks: {result['metadata']['num_chunks']}")
        print(f"  Output saved to: {output_path}")
        
        # Show first chunk as preview
        if result['chunks']:
            first_chunk = result['chunks'][0]
            print(f"\n📝 First chunk preview:")
            print(f"  Chunk ID: {first_chunk['chunk_id']}")
            print(f"  Time: {first_chunk['start_time']:.1f}s - {first_chunk['end_time']:.1f}s")
            print(f"  Tokens: {first_chunk['token_count']}")
            print(f"  Text: {first_chunk['text'][:150]}...")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run get_transcript.py first to download the transcript")
        sys.exit(1)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()