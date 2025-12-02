"""
Unit tests for YouTube transcript retrieval module.
"""

import os
import json
import pytest
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.get_transcript import extract_video_id, get_transcript, save_transcript


class TestExtractVideoId:
    """Test video ID extraction from various URL formats."""
    
    def test_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_embed_url(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_url_with_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s"
        assert extract_video_id(url) == "dQw4w9WgXcQ"
    
    def test_bare_video_id(self):
        video_id = "dQw4w9WgXcQ"
        assert extract_video_id(video_id) == "dQw4w9WgXcQ"
    
    def test_invalid_url(self):
        url = "https://www.example.com/video"
        assert extract_video_id(url) is None


class TestGetTranscript:
    """Test transcript retrieval functionality."""
    
    @pytest.mark.skip(reason="Requires actual YouTube API call - run manually")
    def test_get_transcript_structure(self):
        # Use a known public video with captions
        # This is a TED talk that should have transcripts
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = get_transcript(url)
        
        # Check structure
        assert 'video_id' in result
        assert 'transcript' in result
        assert 'language' in result
        assert 'status' in result
        assert 'error' in result
        
        # Video ID should be extracted
        assert result['video_id'] == "dQw4w9WgXcQ"
        
        # If successful, check transcript format
        if result['status'] == 'success':
            assert isinstance(result['transcript'], list)
            if len(result['transcript']) > 0:
                segment = result['transcript'][0]
                assert 'text' in segment
                assert 'start' in segment
                assert 'duration' in segment
    
    def test_invalid_url_returns_error(self):
        url = "https://www.example.com/invalid"
        result = get_transcript(url)
        
        assert result['status'] == 'error'
        assert result['error'] is not None
        assert 'Invalid YouTube URL' in result['error']


class TestSaveTranscript:
    """Test transcript saving functionality."""
    
    def test_save_transcript_creates_file(self, tmp_path):
        # Mock transcript data
        transcript_data = {
            'video_id': 'test123',
            'transcript': [
                {'text': 'Hello', 'start': 0.0, 'duration': 1.5},
                {'text': 'World', 'start': 1.5, 'duration': 1.5}
            ],
            'language': 'en',
            'status': 'success',
            'error': None
        }
        
        # Save to temp directory
        output_dir = str(tmp_path / "raw")
        filepath = save_transcript(transcript_data, output_dir)
        
        # Check file exists
        assert os.path.exists(filepath)
        
        # Check file contents
        with open(filepath, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['video_id'] == 'test123'
        assert len(saved_data['transcript']) == 2
        assert saved_data['language'] == 'en'
    
    def test_save_transcript_creates_directory(self, tmp_path):
        transcript_data = {
            'video_id': 'test456',
            'transcript': [],
            'language': 'en',
            'status': 'success',
            'error': None
        }
        
        # Use non-existent directory
        output_dir = str(tmp_path / "new_dir" / "raw")
        filepath = save_transcript(transcript_data, output_dir)
        
        # Directory should be created
        assert os.path.exists(output_dir)
        assert os.path.exists(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])