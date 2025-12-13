"""
Unit tests for embedding model module.
"""

import os
import sys
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.retrieval.embedding_model import (
    load_embedding_model,
    embed_chunks,
    save_embeddings,
    load_embeddings,
    embed_video
)


class TestLoadEmbeddingModel:
    """Test embedding model loading."""
    
    @pytest.mark.skip(reason="Requires downloading model - run manually if needed")
    def test_load_default_model(self):
        """Test loading the default embedding model."""
        model = load_embedding_model()
        
        assert model is not None
        assert model.get_sentence_embedding_dimension() > 0
    
    @pytest.mark.skip(reason="Requires downloading model - run manually if needed")
    def test_load_specific_model(self):
        """Test loading a specific model."""
        model = load_embedding_model("all-MiniLM-L6-v2")
        
        assert model is not None
        assert model.get_sentence_embedding_dimension() == 384


class TestEmbedChunks:
    """Test chunk embedding generation."""
    
    def test_embed_chunks_structure(self):
        """Test that embeddings have correct structure."""
        # Mock chunks
        chunks = [
            {'text': 'This is the first chunk', 'chunk_id': 0},
            {'text': 'This is the second chunk', 'chunk_id': 1},
            {'text': 'This is the third chunk', 'chunk_id': 2}
        ]
        
        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(3, 384)
        
        embeddings = embed_chunks(chunks, model=mock_model, show_progress=False)
        
        assert embeddings.shape == (3, 384)
        assert isinstance(embeddings, np.ndarray)
    
    def test_embed_chunks_extracts_text(self):
        """Test that embed_chunks correctly extracts text from chunks."""
        chunks = [
            {'text': 'First', 'chunk_id': 0, 'start_time': 0.0},
            {'text': 'Second', 'chunk_id': 1, 'start_time': 10.0}
        ]
        
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(2, 384)
        
        embed_chunks(chunks, model=mock_model, show_progress=False)
        
        # Verify encode was called with correct texts
        call_args = mock_model.encode.call_args
        texts = call_args[0][0]
        assert texts == ['First', 'Second']
    
    def test_embed_empty_chunks(self):
        """Test handling of empty chunks list."""
        chunks = []
        
        mock_model = Mock()
        mock_model.encode.return_value = np.empty((0, 384))
        
        embeddings = embed_chunks(chunks, model=mock_model, show_progress=False)
        
        assert embeddings.shape == (0, 384)


class TestSaveLoadEmbeddings:
    """Test saving and loading embeddings."""
    
    def test_save_embeddings(self, tmp_path):
        """Test saving embeddings to disk."""
        embeddings = np.random.rand(10, 384)
        video_id = 'test_video'
        
        output_path = save_embeddings(
            embeddings,
            video_id,
            model_name='test-model',
            output_dir=str(tmp_path)
        )
        
        # Check files exist
        assert Path(output_path).exists()
        assert (tmp_path / f"{video_id}_metadata.json").exists()
        
        # Check embeddings can be loaded
        loaded = np.load(output_path)
        assert np.array_equal(loaded, embeddings)
        
        # Check metadata
        with open(tmp_path / f"{video_id}_metadata.json") as f:
            metadata = json.load(f)
        
        assert metadata['video_id'] == video_id
        assert metadata['model_name'] == 'test-model'
        assert metadata['embedding_dim'] == 384
        assert metadata['num_chunks'] == 10
    
    def test_load_embeddings(self, tmp_path):
        """Test loading embeddings from disk."""
        # Create test embeddings
        embeddings = np.random.rand(5, 256)
        video_id = 'test_load'
        
        np.save(tmp_path / f"{video_id}.npy", embeddings)
        
        metadata = {
            'video_id': video_id,
            'model_name': 'test',
            'embedding_dim': 256,
            'num_chunks': 5
        }
        with open(tmp_path / f"{video_id}_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Load embeddings
        loaded_emb, loaded_meta = load_embeddings(video_id, str(tmp_path))
        
        assert np.array_equal(loaded_emb, embeddings)
        assert loaded_meta['video_id'] == video_id
        assert loaded_meta['model_name'] == 'test'
    
    def test_load_nonexistent_embeddings(self, tmp_path):
        """Test error handling for missing embeddings."""
        with pytest.raises(FileNotFoundError):
            load_embeddings('nonexistent', str(tmp_path))


class TestEmbedVideo:
    """Test complete embedding pipeline."""
    
    def test_embed_video_creates_embeddings(self, tmp_path):
        """Test that embed_video creates embedding files."""
        # Create mock chunks file
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        
        chunks_data = {
            'video_id': 'test123',
            'chunks': [
                {'text': 'Chunk 1', 'chunk_id': 0},
                {'text': 'Chunk 2', 'chunk_id': 1}
            ]
        }
        
        with open(chunks_dir / "test123_chunks.json", 'w') as f:
            json.dump(chunks_data, f)
        
        embeddings_dir = tmp_path / "embeddings"
        
        # Mock the model loading and encoding
        with patch('src.retrieval.embedding_model.load_embedding_model') as mock_load:
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(2, 384)
            mock_load.return_value = mock_model
            
            output_path = embed_video(
                video_id='test123',
                chunks_dir=str(chunks_dir),
                embeddings_dir=str(embeddings_dir),
                model_name='test-model'
            )
        
        # Check output files exist
        assert Path(output_path).exists()
        assert (embeddings_dir / "test123_metadata.json").exists()
    
    def test_embed_video_missing_chunks(self, tmp_path):
        """Test error handling when chunks file is missing."""
        with pytest.raises(FileNotFoundError):
            embed_video(
                video_id='missing',
                chunks_dir=str(tmp_path),
                embeddings_dir=str(tmp_path)
            )
    
    def test_embed_video_skip_existing(self, tmp_path):
        """Test that existing embeddings are not regenerated without force."""
        # Create mock chunks
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        
        chunks_data = {
            'video_id': 'test456',
            'chunks': [{'text': 'Test', 'chunk_id': 0}]
        }
        
        with open(chunks_dir / "test456_chunks.json", 'w') as f:
            json.dump(chunks_data, f)
        
        embeddings_dir = tmp_path / "embeddings"
        embeddings_dir.mkdir()
        
        # Create existing embeddings
        np.save(embeddings_dir / "test456.npy", np.random.rand(1, 384))
        
        # Try to embed without force
        with patch('src.retrieval.embedding_model.load_embedding_model') as mock_load:
            output_path = embed_video(
                video_id='test456',
                chunks_dir=str(chunks_dir),
                embeddings_dir=str(embeddings_dir),
                force=False
            )
            
            # Model should not be loaded since embeddings exist
            mock_load.assert_not_called()


class TestIntegration:
    """Integration tests for full workflow."""
    
    @pytest.mark.skip(reason="Requires real model - run manually")
    def test_full_pipeline_real_data(self, tmp_path):
        """Test complete pipeline with real model and data."""
        # Create test chunks
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        
        chunks_data = {
            'video_id': 'integration_test',
            'chunks': [
                {'text': 'The quick brown fox jumps over the lazy dog', 'chunk_id': 0},
                {'text': 'Machine learning is a subset of artificial intelligence', 'chunk_id': 1},
                {'text': 'Python is a popular programming language', 'chunk_id': 2}
            ]
        }
        
        with open(chunks_dir / "integration_test_chunks.json", 'w') as f:
            json.dump(chunks_data, f)
        
        embeddings_dir = tmp_path / "embeddings"
        
        # Run full pipeline
        output_path = embed_video(
            video_id='integration_test',
            chunks_dir=str(chunks_dir),
            embeddings_dir=str(embeddings_dir)
        )
        
        # Load and verify
        embeddings, metadata = load_embeddings('integration_test', str(embeddings_dir))
        
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 384  # Default model dimension
        assert metadata['num_chunks'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])