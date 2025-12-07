"""
Transcript Chunking Module
Splits cleaned transcript into meaningful chunks for retrieval.
"""

from typing import List, Dict, Optional


def chunk_text(cleaned_lines: List[str], max_tokens: int = 250, overlap: int = 50) -> List[Dict]:
    """
    Chunk cleaned transcript lines into segments suitable for retrieval.
    
    Args:
        cleaned_lines: List of cleaned text strings
        max_tokens: Maximum tokens per chunk (approximate, using word count)
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of chunk dictionaries with 'text', 'chunk_id', 'start_line', 'end_line'
    """
    chunks = []
    chunk_id = 0
    
    i = 0
    while i < len(cleaned_lines):
        # Collect lines until we reach max_tokens
        current_chunk = []
        current_tokens = 0
        start_line = i
        
        while i < len(cleaned_lines) and current_tokens < max_tokens:
            line = cleaned_lines[i]
            line_tokens = _estimate_tokens(line)
            
            # If adding this line exceeds max and we already have content, stop
            if current_tokens + line_tokens > max_tokens and current_chunk:
                break
            
            current_chunk.append(line)
            current_tokens += line_tokens
            i += 1
        
        # Create chunk
        if current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': ' '.join(current_chunk),
                'start_line': start_line,
                'end_line': i - 1,
                'token_count': current_tokens
            })
            chunk_id += 1
        
        # Apply overlap by backing up
        if overlap > 0 and i < len(cleaned_lines):
            # Back up to create overlap
            overlap_lines = 0
            overlap_tokens = 0
            j = i - 1
            
            while j >= start_line and overlap_tokens < overlap:
                overlap_tokens += _estimate_tokens(cleaned_lines[j])
                overlap_lines += 1
                j -= 1
            
            i = max(start_line + 1, i - overlap_lines)
    
    return chunks


def chunk_text_with_timestamps(cleaned_segments: List[Dict], max_tokens: int = 250, overlap: int = 50) -> List[Dict]:
    """
    Chunk cleaned transcript segments while preserving timestamps.
    
    Args:
        cleaned_segments: List of segments with 'text', 'start', 'duration'
        max_tokens: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of chunks with text, timestamps, and metadata
    """
    chunks = []
    chunk_id = 0
    
    i = 0
    while i < len(cleaned_segments):
        current_chunk = []
        current_tokens = 0
        start_idx = i
        start_time = cleaned_segments[i]['start']
        
        # Collect segments until max_tokens
        while i < len(cleaned_segments):
            segment = cleaned_segments[i]
            segment_tokens = _estimate_tokens(segment['text'])
            
            # Stop if adding this segment would exceed max_tokens
            # BUT allow at least one segment per chunk
            if current_tokens + segment_tokens > max_tokens and current_chunk:
                break
            
            current_chunk.append(segment['text'])
            current_tokens += segment_tokens
            i += 1
            
            # Hard stop if we've exceeded max_tokens significantly
            if current_tokens >= max_tokens:
                break
        
        # Create chunk with timestamp info
        if current_chunk:
            end_time = cleaned_segments[i-1]['start'] + cleaned_segments[i-1]['duration']
            
            chunks.append({
                'chunk_id': chunk_id,
                'text': ' '.join(current_chunk),
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'start_segment': start_idx,
                'end_segment': i - 1,
                'token_count': current_tokens
            })
            chunk_id += 1
        
        # Apply overlap
        if overlap > 0 and i < len(cleaned_segments):
            overlap_segments = 0
            overlap_tokens = 0
            j = i - 1
            
            while j >= start_idx and overlap_tokens < overlap:
                overlap_tokens += _estimate_tokens(cleaned_segments[j]['text'])
                overlap_segments += 1
                j -= 1
            
            i = max(start_idx + 1, i - overlap_segments)
    
    return chunks


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count (rough approximation: 1 token ≈ 0.75 words).
    
    Args:
        text: Text to estimate
        
    Returns:
        Estimated token count
    """
    words = len(text.split())
    # Rough estimate: 1 token ≈ 0.75 words (or 4 chars)
    return int(words / 0.75)


def chunk_by_sentences(text: str, max_tokens: int = 250) -> List[str]:
    """
    Chunk text by sentences while respecting max_tokens.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of text chunks
    """
    import re
    
    # Simple sentence splitting (can be improved with nltk)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python src/preprocessing/chunk_transcript.py <video_id> [max_tokens]")
        print("Example: python src/preprocessing/chunk_transcript.py dQw4w9WgXcQ 250")
        sys.exit(1)
    
    video_id = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    
    # Try to load cleaned transcript first
    cleaned_path = Path(f"src/data/processed/{video_id}_cleaned.json")
    
    if cleaned_path.exists():
        with open(cleaned_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cleaned_lines = data.get('cleaned_lines', [])
        cleaned_segments = data.get('cleaned_segments', [])
    else:
        # Fall back to raw transcript
        raw_path = Path(f"src/data/raw/{video_id}.json")
        if not raw_path.exists():
            print(f"Error: Neither {cleaned_path} nor {raw_path} found")
            print("Run get_transcript.py first")
            sys.exit(1)
        
        with open(raw_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Quick clean
        from src.preprocessing.clean_transcript import clean_text, clean_transcript_with_timestamps
        cleaned_lines = clean_text(data.get('transcript', []))
        cleaned_segments = clean_transcript_with_timestamps(data.get('transcript', []))
    
    # Create chunks
    chunks = chunk_text(cleaned_lines, max_tokens=max_tokens)
    chunks_with_timestamps = chunk_text_with_timestamps(cleaned_segments, max_tokens=max_tokens)
    
    print(f"Created {len(chunks)} chunks (max_tokens={max_tokens})")
    print(f"\nFirst chunk:")
    print(f"  Chunk ID: {chunks[0]['chunk_id']}")
    print(f"  Lines: {chunks[0]['start_line']}-{chunks[0]['end_line']}")
    print(f"  Tokens: {chunks[0]['token_count']}")
    print(f"  Text preview: {chunks[0]['text'][:200]}...")
    
    # Save chunks
    output_path = Path(f"src/data/processed/{video_id}_chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'video_id': video_id,
            'max_tokens': max_tokens,
            'num_chunks': len(chunks),
            'chunks': chunks,
            'chunks_with_timestamps': chunks_with_timestamps
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nChunks saved to: {output_path}")