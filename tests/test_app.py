"""
Unit tests for YouTube Q&A System main application.
"""

import os
import json
import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import YouTubeQASystem, is_music_video


class TestIsMusicVideo:
    """Test music video detection functionality."""
    
    def test_detects_music_with_note_symbols(self):
        segments = [
            {'text': 'Hello ♪ world', 'start': 0.0, 'duration': 1.0},
            {'text': 'Normal text', 'start': 1.0, 'duration': 1.0}
        ]
        assert is_music_video(segments) is True
    
    def test_detects_music_with_brackets(self):
        segments = [
            {'text': '[Music] playing', 'start': 0.0, 'duration': 1.0}
        ]
        assert is_music_video(segments) is True
    
    def test_detects_music_with_parentheses(self):
        segments = [
            {'text': '(Music) in background', 'start': 0.0, 'duration': 1.0}
        ]
        assert is_music_video(segments) is True
    
    def test_detects_music_with_emoji(self):
        segments = [
            {'text': 'Song lyrics 🎵', 'start': 0.0, 'duration': 1.0}
        ]
        assert is_music_video(segments) is True
    
    def test_case_insensitive_detection(self):
        segments = [
            {'text': '[MUSIC] PLAYING', 'start': 0.0, 'duration': 1.0}
        ]
        assert is_music_video(segments) is True
    
    def test_no_music_detected(self):
        segments = [
            {'text': 'Welcome to my channel', 'start': 0.0, 'duration': 1.0},
            {'text': 'Today we will discuss', 'start': 1.0, 'duration': 1.0}
        ]
        assert is_music_video(segments) is False
    
    def test_empty_segments(self):
        segments = []
        assert is_music_video(segments) is False


class TestYouTubeQASystemInit:
    """Test YouTubeQASystem initialization."""
    
    def test_default_initialization(self, tmp_path):
        with patch('app.Path') as mock_path:
            mock_path.return_value.mkdir = MagicMock()
            qa_system = YouTubeQASystem()
            assert qa_system.raw_dir is not None
            assert qa_system.processed_dir is not None
    
    def test_custom_directories(self, tmp_path):
        raw_dir = str(tmp_path / "custom_raw")
        processed_dir = str(tmp_path / "custom_processed")
        
        qa_system = YouTubeQASystem(raw_dir=raw_dir, processed_dir=processed_dir)
        
        assert qa_system.raw_dir == Path(raw_dir)
        assert qa_system.processed_dir == Path(processed_dir)
        assert qa_system.raw_dir.exists()
        assert qa_system.processed_dir.exists()


class TestProcessVideo:
    """Test video processing functionality."""
    
    @patch('app.get_transcript')
    @patch('app.save_transcript')
    @patch('app.preprocess_transcript')
    def test_successful_processing(self, mock_preprocess, mock_save, mock_get_transcript, tmp_path):
        # Setup
        qa_system = YouTubeQASystem(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed")
        )
        
        # Mock transcript data
        mock_transcript_data = {
            'video_id': 'test123',
            'transcript': [
                {'text': 'Hello world', 'start': 0.0, 'duration': 1.0}
            ],
            'language': 'en',
            'status': 'success',
            'error': None
        }
        mock_get_transcript.return_value = mock_transcript_data
        mock_save.return_value = str(tmp_path / "raw" / "test123.json")
        
        # Mock preprocessing result
        mock_preprocess_result = {
            'chunks': [{'text': 'Hello world', 'start': 0.0, 'end': 1.0}],
            'metadata': {
                'original_segments': 1,
                'cleaned_segments': 1,
                'num_chunks': 1
            }
        }
        mock_preprocess.return_value = mock_preprocess_result
        
        # Execute
        result = qa_system.process_video('https://www.youtube.com/watch?v=test123')
        
        # Verify
        assert result['status'] == 'success'
        assert result['video_id'] == 'test123'
        assert 'transcript_retrieval' in result['steps_completed']
        assert 'preprocessing' in result['steps_completed']
        assert result['error'] is None
    
    @patch('app.get_transcript')
    def test_transcript_retrieval_failure(self, mock_get_transcript, tmp_path):
        qa_system = YouTubeQASystem(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed")
        )
        
        # Mock failed transcript retrieval
        mock_get_transcript.return_value = {
            'status': 'error',
            'error': 'Video not found',
            'video_id': None,
            'transcript': None,
            'language': None
        }
        
        result = qa_system.process_video('https://www.youtube.com/watch?v=invalid')
        
        assert result['status'] == 'error'
        assert 'Transcript retrieval failed' in result['error']
        assert result['video_id'] is None
    
    @patch('app.get_transcript')
    @patch('app.save_transcript')
    def test_music_video_auto_detection(self, mock_save, mock_get_transcript, tmp_path):
        qa_system = YouTubeQASystem(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed")
        )
        
        # Mock music video transcript
        mock_transcript_data = {
            'video_id': 'music123',
            'transcript': [
                {'text': '♪ La la la ♪', 'start': 0.0, 'duration': 2.0}
            ],
            'language': 'en',
            'status': 'success',
            'error': None
        }
        mock_get_transcript.return_value = mock_transcript_data
        mock_save.return_value = str(tmp_path / "raw" / "music123.json")
        
        with patch('app.preprocess_transcript') as mock_preprocess:
            mock_preprocess.return_value = {
                'chunks': [],
                'metadata': {'original_segments': 1, 'cleaned_segments': 1, 'num_chunks': 0}
            }
            
            result = qa_system.process_video('https://www.youtube.com/watch?v=music123')
            
            # Verify merge_gap was set to 0.0 for music video
            call_args = mock_preprocess.call_args
            assert call_args[1]['merge_gap'] == 0.0
    
    @patch('app.get_transcript')
    @patch('app.save_transcript')
    def test_normal_video_auto_detection(self, mock_save, mock_get_transcript, tmp_path):
        qa_system = YouTubeQASystem(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed")
        )
        
        # Mock normal video transcript
        mock_transcript_data = {
            'video_id': 'normal123',
            'transcript': [
                {'text': 'Welcome to my tutorial', 'start': 0.0, 'duration': 2.0}
            ],
            'language': 'en',
            'status': 'success',
            'error': None
        }
        mock_get_transcript.return_value = mock_transcript_data
        mock_save.return_value = str(tmp_path / "raw" / "normal123.json")
        
        with patch('app.preprocess_transcript') as mock_preprocess:
            mock_preprocess.return_value = {
                'chunks': [],
                'metadata': {'original_segments': 1, 'cleaned_segments': 1, 'num_chunks': 0}
            }
            
            result = qa_system.process_video('https://www.youtube.com/watch?v=normal123')
            
            # Verify merge_gap was set to 1.0 for normal video
            call_args = mock_preprocess.call_args
            assert call_args[1]['merge_gap'] == 1.0
    
    @patch('app.get_transcript')
    @patch('app.save_transcript')
    def test_manual_merge_gap_override(self, mock_save, mock_get_transcript, tmp_path):
        qa_system = YouTubeQASystem(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed")
        )
        
        mock_transcript_data = {
            'video_id': 'test123',
            'transcript': [
                {'text': 'Test', 'start': 0.0, 'duration': 1.0}
            ],
            'language': 'en',
            'status': 'success',
            'error': None
        }
        mock_get_transcript.return_value = mock_transcript_data
        mock_save.return_value = str(tmp_path / "raw" / "test123.json")
        
        with patch('app.preprocess_transcript') as mock_preprocess:
            mock_preprocess.return_value = {
                'chunks': [],
                'metadata': {'original_segments': 1, 'cleaned_segments': 1, 'num_chunks': 0}
            }
            
            # Test with manual override
            result = qa_system.process_video(
                'https://www.youtube.com/watch?v=test123',
                merge_gap=2.5
            )
            
            call_args = mock_preprocess.call_args
            assert call_args[1]['merge_gap'] == 2.5


class TestAskQuestion:
    """Test question answering functionality."""
    
    def test_ask_question_no_processed_data(self, tmp_path):
        qa_system = YouTubeQASystem(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed")
        )
        
        result = qa_system.ask_question('nonexistent123', 'What is this about?')
        
        assert result['status'] == 'error'
        assert 'No processed data found' in result['error']
    
    def test_ask_question_with_processed_data(self, tmp_path):
        qa_system = YouTubeQASystem(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed")
        )
        
        # Create mock chunks file
        chunks_data = {
            'chunks': [
                {'text': 'Sample chunk', 'start': 0.0, 'end': 5.0}
            ],
            'metadata': {}
        }
        chunks_path = tmp_path / "processed" / "test123_chunks.json"
        with open(chunks_path, 'w') as f:
            json.dump(chunks_data, f)
        
        result = qa_system.ask_question('test123', 'What is this about?')
        
        assert result['status'] == 'success'
        assert result['question'] == 'What is this about?'
        assert 'answer' in result
        assert result['chunks_available'] == 1


class TestListProcessedVideos:
    """Test listing processed videos functionality."""
    
    def test_list_empty(self, tmp_path):
        qa_system = YouTubeQASystem(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed")
        )
        
        videos = qa_system.list_processed_videos()
        
        assert videos == []
    
    def test_list_with_transcript_only(self, tmp_path):
        qa_system = YouTubeQASystem(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed")
        )
        
        # Create transcript file
        transcript_path = tmp_path / "raw" / "video123.json"
        with open(transcript_path, 'w') as f:
            json.dump({'video_id': 'video123'}, f)
        
        videos = qa_system.list_processed_videos()
        
        assert len(videos) == 1
        assert videos[0]['video_id'] == 'video123'
        assert videos[0]['has_transcript'] is True
        assert videos[0]['has_chunks'] is False
    
    def test_list_with_complete_processing(self, tmp_path):
        qa_system = YouTubeQASystem(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed")
        )
        
        # Create transcript file
        transcript_path = tmp_path / "raw" / "video123.json"
        with open(transcript_path, 'w') as f:
            json.dump({'video_id': 'video123'}, f)
        
        # Create chunks file
        chunks_path = tmp_path / "processed" / "video123_chunks.json"
        with open(chunks_path, 'w') as f:
            json.dump({'chunks': []}, f)
        
        videos = qa_system.list_processed_videos()
        
        assert len(videos) == 1
        assert videos[0]['video_id'] == 'video123'
        assert videos[0]['has_transcript'] is True
        assert videos[0]['has_chunks'] is True
    
    def test_list_multiple_videos(self, tmp_path):
        qa_system = YouTubeQASystem(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed")
        )
        
        # Create multiple transcript files
        for video_id in ['video1', 'video2', 'video3']:
            transcript_path = tmp_path / "raw" / f"{video_id}.json"
            with open(transcript_path, 'w') as f:
                json.dump({'video_id': video_id}, f)
        
        videos = qa_system.list_processed_videos()
        
        assert len(videos) == 3
        video_ids = [v['video_id'] for v in videos]
        assert 'video1' in video_ids
        assert 'video2' in video_ids
        assert 'video3' in video_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])