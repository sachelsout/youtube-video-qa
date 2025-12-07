"""
Transcript Cleaning Module
Cleans and normalizes YouTube transcript text.
"""

import re
from typing import List, Dict

def clean_text(raw_transcript: List[Dict]) -> List[str]:
    """
    Clean transcript text by removing noise and normalizing.
    
    Args:
        raw_transcript: List of transcript segments with 'text', 'start', 'duration'
        
    Returns:
        List of cleaned text strings (one per segment)
    """
    cleaned_lines = []
    
    for segment in raw_transcript:
        text = segment.get('text', '')
        
        # Skip empty segments
        if not text or not text.strip():
            continue
        
        # Clean the text
        cleaned = _clean_segment(text)
        
        # Only add non-empty cleaned text
        if cleaned:
            cleaned_lines.append(cleaned)
    
    return cleaned_lines


def _clean_segment(text: str) -> str:
    """
    Clean a single text segment.
    
    Args:
        text: Raw text segment
        
    Returns:
        Cleaned text string
    """
    # Remove music notation [♪♪♪] or ♪ symbols
    text = re.sub(r'\[?♪+\]?', '', text)
    text = re.sub(r'♪', '', text)
    
    # Remove sound effects [Applause], [Laughter], etc.
    text = re.sub(r'\[([A-Z][a-z]+)\]', '', text)
    
    # Remove bracketed content that's not speech
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove parenthetical content (optional - keep if contains actual speech)
    # text = re.sub(r'\(.*?\)', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove leading dashes or bullets
    text = re.sub(r'^[-•]\s*', '', text)
    
    return text


def clean_transcript_with_timestamps(raw_transcript: List[Dict]) -> List[Dict]:
    """
    Clean transcript while preserving timestamp information.
    
    Args:
        raw_transcript: List of transcript segments with 'text', 'start', 'duration'
        
    Returns:
        List of cleaned segments with preserved timestamps
    """
    cleaned_segments = []
    
    for segment in raw_transcript:
        text = segment.get('text', '')
        
        # Skip empty segments
        if not text or not text.strip():
            continue
        
        # Clean the text
        cleaned = _clean_segment(text)
        
        # Only add non-empty cleaned text
        if cleaned:
            cleaned_segments.append({
                'text': cleaned,
                'start': segment.get('start', 0),
                'duration': segment.get('duration', 0)
            })
    
    return cleaned_segments


def merge_segments(segments: List[Dict], max_gap: float = 2.0) -> List[Dict]:
    """
    Merge consecutive segments that are close together.
    
    Args:
        segments: List of segments with 'text', 'start', 'duration'
        max_gap: Maximum gap in seconds to merge (default: 2.0)
        
    Returns:
        List of merged segments
    """
    if not segments:
        return []
    
    # If max_gap is 0 or negative, don't merge anything
    if max_gap <= 0:
        return segments
    
    merged = []
    current = segments[0].copy()
    
    for next_seg in segments[1:]:
        current_end = current['start'] + current['duration']
        next_start = next_seg['start']
        gap = next_start - current_end
        
        # If gap is small enough AND gap is positive, merge
        if gap >= 0 and gap <= max_gap:
            current['text'] = current['text'] + ' ' + next_seg['text']
            current['duration'] = (next_seg['start'] + next_seg['duration']) - current['start']
        else:
            merged.append(current)
            current = next_seg.copy()
    
    # Add the last segment
    merged.append(current)
    
    return merged


if __name__ == "__main__":
    # Example usage
    import sys
    import json
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python src/preprocessing/clean_transcript.py <video_id>")
        print("Example: python src/preprocessing/clean_transcript.py dQw4w9WgXcQ")
        sys.exit(1)
    
    video_id = sys.argv[1]
    input_path = Path(f"src/data/raw/{video_id}.json")
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        print("Run get_transcript.py first to download the transcript")
        sys.exit(1)
    
    # Load raw transcript
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    raw_transcript = data.get('transcript', [])
    
    # Clean transcript
    cleaned_lines = clean_text(raw_transcript)
    cleaned_with_timestamps = clean_transcript_with_timestamps(raw_transcript)
    
    print(f"Original segments: {len(raw_transcript)}")
    print(f"Cleaned segments: {len(cleaned_lines)}")
    print("\nFirst 5 cleaned lines:")
    for i, line in enumerate(cleaned_lines[:5]):
        print(f"{i+1}. {line}")
    
    # Save cleaned version (optional)
    output_path = Path(f"src/data/processed/{video_id}_cleaned.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'video_id': video_id,
            'cleaned_lines': cleaned_lines,
            'cleaned_segments': cleaned_with_timestamps
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nCleaned transcript saved to: {output_path}")