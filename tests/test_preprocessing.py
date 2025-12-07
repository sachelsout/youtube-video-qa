"""
Unit tests for transcript preprocessing modules.
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.clean_transcript import (
    clean_text,
    clean_transcript_with_timestamps,
    merge_segments,
    _clean_segment
)
from src.preprocessing.chunk_transcript import (
    chunk_text,
    chunk_text_with_timestamps,
    _estimate_tokens
)


class TestCleanSegment:
    """Test individual segment cleaning."""
    
    def test_remove_music_notation(self):
        assert _clean_segment("[♪♪♪]") == ""
        assert _clean_segment("♪ Hello ♪") == "Hello"
        assert _clean_segment("Text [♪♪] more text") == "Text more text"
    
    def test_remove_sound_effects(self):
        assert _clean_segment("[Applause]") == ""
        assert _clean_segment("Hello [Laughter] world") == "Hello world"
        assert _clean_segment("[Music] Introduction") == "Introduction"
    
    def test_normalize_whitespace(self):
        assert _clean_segment("Hello    world") == "Hello world"
        assert _clean_segment("  Text  ") == "Text"
        assert _clean_segment("Multiple\n\nlines") == "Multiple lines"
    
    def test_remove_leading_dashes(self):
        assert _clean_segment("- Item one") == "Item one"
        assert _clean_segment("• Bullet point") == "Bullet point"
    
    def test_preserve_regular_text(self):
        assert _clean_segment("Hello world") == "Hello world"
        assert _clean_segment("This is a test.") == "This is a test."


class TestCleanText:
    """Test transcript cleaning functions."""
    
    def test_clean_text_basic(self):
        raw_transcript = [
            {'text': 'Hello world', 'start': 0, 'duration': 1},
            {'text': '[Music]', 'start': 1, 'duration': 1},
            {'text': 'How are you?', 'start': 2, 'duration': 1}
        ]
        
        result = clean_text(raw_transcript)
        
        assert len(result) == 2
        assert result[0] == "Hello world"
        assert result[1] == "How are you?"
    
    def test_clean_text_removes_empty(self):
        raw_transcript = [
            {'text': '', 'start': 0, 'duration': 1},
            {'text': '   ', 'start': 1, 'duration': 1},
            {'text': 'Valid text', 'start': 2, 'duration': 1}
        ]
        
        result = clean_text(raw_transcript)
        
        assert len(result) == 1
        assert result[0] == "Valid text"
    
    def test_clean_transcript_with_timestamps(self):
        raw_transcript = [
            {'text': 'Hello', 'start': 0, 'duration': 1},
            {'text': '[Music]', 'start': 1, 'duration': 1},
            {'text': 'World', 'start': 2, 'duration': 1}
        ]
        
        result = clean_transcript_with_timestamps(raw_transcript)
        
        assert len(result) == 2
        assert result[0]['text'] == "Hello"
        assert result[0]['start'] == 0
        assert result[1]['text'] == "World"
        assert result[1]['start'] == 2


class TestMergeSegments:
    """Test segment merging functionality."""
    
    def test_merge_close_segments(self):
        segments = [
            {'text': 'Hello', 'start': 0, 'duration': 1},
            {'text': 'world', 'start': 1.5, 'duration': 1}
        ]
        
        result = merge_segments(segments, max_gap=2.0)
        
        assert len(result) == 1
        assert result[0]['text'] == "Hello world"
        assert result[0]['start'] == 0
    
    def test_dont_merge_distant_segments(self):
        segments = [
            {'text': 'Hello', 'start': 0, 'duration': 1},
            {'text': 'world', 'start': 10, 'duration': 1}
        ]
        
        result = merge_segments(segments, max_gap=2.0)
        
        assert len(result) == 2
    
    def test_merge_empty_list(self):
        result = merge_segments([])
        assert result == []


class TestEstimateTokens:
    """Test token estimation."""
    
    def test_estimate_tokens_basic(self):
        # Rough estimate: "hello world" = 2 words / 0.75 ≈ 2-3 tokens
        tokens = _estimate_tokens("hello world")
        assert 2 <= tokens <= 3
    
    def test_estimate_tokens_long_text(self):
        text = " ".join(["word"] * 100)
        tokens = _estimate_tokens(text)
        # 100 words / 0.75 ≈ 133 tokens
        assert 130 <= tokens <= 140


class TestChunkText:
    """Test text chunking functionality."""
    
    def test_chunk_text_single_chunk(self):
        cleaned_lines = ["Hello world", "This is a test"]
        
        chunks = chunk_text(cleaned_lines, max_tokens=100)
        
        assert len(chunks) == 1
        assert chunks[0]['chunk_id'] == 0
        assert "Hello world" in chunks[0]['text']
        assert "This is a test" in chunks[0]['text']
    
    def test_chunk_text_multiple_chunks(self):
        # Create lines that will exceed max_tokens
        cleaned_lines = [" ".join(["word"] * 50) for _ in range(10)]
        
        chunks = chunk_text(cleaned_lines, max_tokens=100)
        
        assert len(chunks) > 1
        assert chunks[0]['chunk_id'] == 0
        assert chunks[1]['chunk_id'] == 1
    
    def test_chunk_text_with_overlap(self):
        cleaned_lines = [" ".join(["word"] * 30) for _ in range(5)]
        
        chunks = chunk_text(cleaned_lines, max_tokens=100, overlap=20)
        
        # With overlap, there should be some content repetition
        assert len(chunks) >= 1
    
    def test_chunk_text_empty_input(self):
        chunks = chunk_text([])
        assert chunks == []


class TestChunkTextWithTimestamps:
    """Test chunking with timestamp preservation."""
    
    def test_chunk_preserves_timestamps(self):
        segments = [
            {'text': 'Hello', 'start': 0, 'duration': 1},
            {'text': 'world', 'start': 1, 'duration': 1},
            {'text': 'test', 'start': 2, 'duration': 1}
        ]
        
        chunks = chunk_text_with_timestamps(segments, max_tokens=100)
        
        assert len(chunks) == 1
        assert chunks[0]['start_time'] == 0
        assert chunks[0]['end_time'] == 3
        assert 'Hello' in chunks[0]['text']
    
    def test_chunk_with_timestamps_multiple_chunks(self):
        # Create segments that will require multiple chunks
        segments = [
            {'text': ' '.join(['word'] * 50), 'start': i, 'duration': 1}
            for i in range(10)
        ]
        
        chunks = chunk_text_with_timestamps(segments, max_tokens=100)
        
        assert len(chunks) > 1
        assert chunks[0]['end_time'] <= chunks[1]['start_time']


class TestIntegration:
    """Integration tests for full preprocessing pipeline."""
    
    def test_full_pipeline_sample_data(self):
        raw_transcript = [
            {'text': '[Music]', 'start': 0, 'duration': 2},
            {'text': 'Welcome to the show', 'start': 2, 'duration': 2},
            {'text': 'Today we will discuss', 'start': 4, 'duration': 2},
            {'text': '[Applause]', 'start': 6, 'duration': 1},
            {'text': 'various topics', 'start': 7, 'duration': 2}
        ]
        
        # Clean
        cleaned = clean_transcript_with_timestamps(raw_transcript)
        assert len(cleaned) == 3  # Music and Applause removed
        
        # Chunk
        chunks = chunk_text_with_timestamps(cleaned, max_tokens=50)
        assert len(chunks) >= 1
        assert chunks[0]['start_time'] == 2
        assert 'Welcome' in chunks[0]['text']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])