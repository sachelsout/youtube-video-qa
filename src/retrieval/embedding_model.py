"""
Embedding Model Module
Generates embeddings for transcript chunks using sentence transformers.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import sys

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed.")
    print("Install with: pip install sentence-transformers")
    sys.exit(1)


# Default model - good balance of speed and quality
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast, good quality

# Alternative models (uncomment to use)
# DEFAULT_MODEL = "all-mpnet-base-v2"  # 768 dims, slower, better quality
# DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual support


def load_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Load a sentence transformer embedding model.
    
    Args:
        model_name: Name of the sentence-transformers model
        
    Returns:
        Loaded SentenceTransformer model
    """
    print(f"Loading embedding model: {model_name}")
    
    try:
        model = SentenceTransformer(model_name)
        print(f"✓ Model loaded successfully")
        print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def embed_chunks(
    chunks: List[Dict],
    model: Optional[SentenceTransformer] = None,
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for a list of transcript chunks.
    
    Args:
        chunks: List of chunk dictionaries with 'text' field
        model: Pre-loaded embedding model (loads default if None)
        batch_size: Number of chunks to process at once
        show_progress: Whether to show progress bar
        
    Returns:
        numpy array of shape (num_chunks, embedding_dim)
    """
    if model is None:
        model = load_embedding_model()
    
    # Extract text from chunks
    texts = [chunk['text'] for chunk in chunks]
    
    print(f"\nGenerating embeddings for {len(texts)} chunks...")
    
    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    print(f"✓ Generated embeddings: shape {embeddings.shape}")
    
    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    video_id: str,
    model_name: str = DEFAULT_MODEL,
    output_dir: str = "src/data/embeddings"
) -> str:
    """
    Save embeddings to disk.
    
    Args:
        embeddings: numpy array of embeddings
        video_id: YouTube video ID
        model_name: Name of the model used
        output_dir: Directory to save embeddings
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    embeddings_file = output_path / f"{video_id}.npy"
    np.save(embeddings_file, embeddings)
    
    # Save metadata (model info, dimensions, etc.)
    metadata = {
        'video_id': video_id,
        'model_name': model_name,
        'embedding_dim': embeddings.shape[1],
        'num_chunks': embeddings.shape[0],
        'shape': list(embeddings.shape)
    }
    
    metadata_file = output_path / f"{video_id}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Embeddings saved to: {embeddings_file}")
    print(f"✓ Metadata saved to: {metadata_file}")
    
    return str(embeddings_file)


def load_embeddings(
    video_id: str,
    embeddings_dir: str = "src/data/embeddings"
) -> tuple[np.ndarray, Dict]:
    """
    Load embeddings from disk.
    
    Args:
        video_id: YouTube video ID
        embeddings_dir: Directory containing embeddings
        
    Returns:
        Tuple of (embeddings array, metadata dict)
    """
    embeddings_path = Path(embeddings_dir) / f"{video_id}.npy"
    metadata_path = Path(embeddings_dir) / f"{video_id}_metadata.json"
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    
    # Load metadata
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return embeddings, metadata


def embed_video(
    video_id: str,
    chunks_dir: str = "src/data/processed",
    embeddings_dir: str = "src/data/embeddings",
    model_name: str = DEFAULT_MODEL,
    force: bool = False
) -> str:
    """
    Complete pipeline: load chunks, generate embeddings, save to disk.
    
    Args:
        video_id: YouTube video ID
        chunks_dir: Directory containing processed chunks
        embeddings_dir: Directory to save embeddings
        model_name: Embedding model to use
        force: Force regeneration even if embeddings exist
        
    Returns:
        Path to saved embeddings file
    """
    print(f"\n{'='*60}")
    print(f"Generating Embeddings for: {video_id}")
    print(f"{'='*60}\n")
    
    # Check if embeddings already exist
    embeddings_path = Path(embeddings_dir) / f"{video_id}.npy"
    if embeddings_path.exists() and not force:
        print(f"✓ Embeddings already exist: {embeddings_path}")
        print("Use --force to regenerate")
        return str(embeddings_path)
    
    # Load chunks
    chunks_file = Path(chunks_dir) / f"{video_id}_chunks.json"
    
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"Chunks not found: {chunks_file}\n"
            f"Run preprocessing first: python app.py process {video_id}"
        )
    
    print(f"Loading chunks from: {chunks_file}")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data.get('chunks', [])
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Load model
    model = load_embedding_model(model_name)
    
    # Generate embeddings
    embeddings = embed_chunks(chunks, model=model)
    
    # Save embeddings
    output_path = save_embeddings(
        embeddings,
        video_id,
        model_name=model_name,
        output_dir=embeddings_dir
    )
    
    print(f"\n{'='*60}")
    print("✓ Embedding generation complete!")
    print(f"{'='*60}")
    
    return output_path


def main():
    """Command-line interface for embedding generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate embeddings for video transcript chunks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings for a video
  python src/retrieval/embedding_model.py --video_id dQw4w9WgXcQ
  
  # Use a different model
  python src/retrieval/embedding_model.py --video_id dQw4w9WgXcQ --model all-mpnet-base-v2
  
  # Force regeneration
  python src/retrieval/embedding_model.py --video_id dQw4w9WgXcQ --force
        """
    )
    
    parser.add_argument('--video_id', required=True, help='YouTube video ID')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Embedding model name')
    parser.add_argument('--chunks_dir', default='src/data/processed', help='Chunks directory')
    parser.add_argument('--embeddings_dir', default='src/data/embeddings', help='Output directory')
    parser.add_argument('--force', action='store_true', help='Force regeneration')
    
    args = parser.parse_args()
    
    try:
        embed_video(
            video_id=args.video_id,
            chunks_dir=args.chunks_dir,
            embeddings_dir=args.embeddings_dir,
            model_name=args.model,
            force=args.force
        )
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()