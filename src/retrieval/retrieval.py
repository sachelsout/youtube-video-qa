"""
Retrieval Module for YouTube Video QA System
Finds top-k most relevant chunks given a query using cosine similarity.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity


def get_embedding_model():
    """
    Helper function to load the embedding model.
    This imports from your embedding_model.py module.
    """
    try:
        from src.retrieval.embedding_model import load_embedding_model
        return load_embedding_model()
    except ImportError as e:
        raise ImportError(
            f"Could not import embedding model: {e}\n"
            "Make sure sentence-transformers is installed: pip install sentence-transformers"
        )


def retrieve_top_k(
    question: str,
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    k: int = 5,
    question_embedding: Optional[np.ndarray] = None,
    embed_function: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most relevant chunks for a given question.
    
    Args:
        question: The query string
        embeddings: Numpy array of shape (n_chunks, embedding_dim) containing chunk embeddings
        chunks: List of chunk dictionaries containing metadata (text, video_id, timestamp, etc.)
        k: Number of top chunks to retrieve
        question_embedding: Pre-computed embedding for the question (optional)
        embed_function: Function to compute embeddings if question_embedding not provided
        
    Returns:
        List of top-k chunk dictionaries with added 'similarity_score' field
        
    Raises:
        ValueError: If embeddings and chunks have mismatched lengths
        ValueError: If neither question_embedding nor embed_function is provided
    """
    # Validate inputs
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Mismatch between embeddings ({len(embeddings)}) and chunks ({len(chunks)})"
        )
    
    if len(chunks) == 0:
        return []
    
    # Get question embedding
    if question_embedding is None:
        if embed_function is None:
            raise ValueError(
                "Either question_embedding or embed_function must be provided"
            )
        question_embedding = embed_function(question)
    
    # Ensure question_embedding is 2D for cosine_similarity
    if question_embedding.ndim == 1:
        question_embedding = question_embedding.reshape(1, -1)
    
    # Compute cosine similarities
    similarities = cosine_similarity(question_embedding, embeddings)[0]
    
    # Get top-k indices
    k_actual = min(k, len(chunks))
    top_k_indices = np.argsort(similarities)[-k_actual:][::-1]
    
    # Prepare results with similarity scores
    results = []
    for idx in top_k_indices:
        chunk_copy = chunks[idx].copy()
        chunk_copy['similarity_score'] = float(similarities[idx])
        chunk_copy['rank'] = len(results) + 1
        results.append(chunk_copy)
    
    return results


def retrieve_top_k_from_video(
    question: str,
    video_id: str,
    k: int = 5,
    data_dir: str = "src/data/processed",
    embeddings_dir: str = "src/data/embeddings",
    embed_function: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks from a single video using saved embeddings and chunks.
    
    Args:
        question: The query string
        video_id: YouTube video ID
        k: Number of top chunks to retrieve
        data_dir: Directory containing processed chunks
        embeddings_dir: Directory containing saved embeddings
        embed_function: Function to compute question embedding
        
    Returns:
        List of top-k chunk dictionaries with similarity scores
        
    Raises:
        FileNotFoundError: If chunks or embeddings not found
    """
    data_path = Path(data_dir)
    embeddings_path = Path(embeddings_dir)
    
    # Load chunks
    chunks_file = data_path / f"{video_id}_chunks.json"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    
    with open(chunks_file, 'r') as f:
        data = json.load(f)
        chunks = data.get('chunks', [])
    
    video_id_from_file = data.get('video_id', video_id)  # Get from file, fallback to param
    for chunk in chunks:
        if 'video_id' not in chunk:
            chunk['video_id'] = video_id_from_file

    # Load embeddings
    embeddings_file = embeddings_path / f"{video_id}.npy"
    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {embeddings_file}. "
            f"Run 'python app.py embed {video_id}' first."
        )
    
    embeddings = np.load(embeddings_file)
    
    # Retrieve top-k
    return retrieve_top_k(
        question=question,
        embeddings=embeddings,
        chunks=chunks,
        k=k,
        embed_function=embed_function
    )


def retrieve_top_k_multi_video(
    question: str,
    video_ids: List[str],
    k: int = 5,
    data_dir: str = "src/data/processed",
    embeddings_dir: str = "src/data/embeddings",
    embed_function: Optional[callable] = None,
    per_video_limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks across multiple videos.
    
    Args:
        question: The query string
        video_ids: List of YouTube video IDs to search
        k: Total number of top chunks to retrieve across all videos
        data_dir: Directory containing processed chunks
        embeddings_dir: Directory containing saved embeddings
        embed_function: Function to compute question embedding
        per_video_limit: Max chunks to consider per video (optional)
        
    Returns:
        List of top-k chunk dictionaries sorted by similarity score
    """
    if not video_ids:
        return []
    
    # Compute question embedding once
    if embed_function is None:
        raise ValueError("embed_function must be provided")
    
    question_embedding = embed_function(question)
    
    # Collect results from all videos
    all_results = []
    
    for video_id in video_ids:
        try:
            video_results = retrieve_top_k_from_video(
                question=question,
                video_id=video_id,
                k=per_video_limit if per_video_limit else k * 2,  # Get more per video
                data_dir=data_dir,
                embeddings_dir=embeddings_dir,
                embed_function=lambda q: question_embedding  # Reuse embedding
            )
            all_results.extend(video_results)
        except FileNotFoundError as e:
            print(f"⚠️  Skipping {video_id}: {e}")
            continue
    
    if not all_results:
        return []
    
    # Sort all results by similarity and take top-k
    all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    top_k_results = all_results[:k]
    
    # Update ranks
    for i, result in enumerate(top_k_results):
        result['rank'] = i + 1
    
    return top_k_results


def filter_by_threshold(
    results: List[Dict[str, Any]],
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Filter retrieval results by minimum similarity threshold.
    
    Args:
        results: List of retrieval results with similarity_score
        threshold: Minimum similarity score (0-1)
        
    Returns:
        Filtered list of results
    """
    return [r for r in results if r['similarity_score'] >= threshold]


def format_retrieval_results(results: List[Dict[str, Any]]) -> str:
    """
    Format retrieval results for display.
    
    Args:
        results: List of retrieval results
        
    Returns:
        Formatted string for console output
    """
    if not results:
        return "No results found."
    
    output = []
    for result in results:
        score = result.get('similarity_score', 0)
        rank = result.get('rank', '?')
        text = result.get('text', 'N/A')
        video_id = result.get('video_id', 'unknown')
        start_time = result.get('start_time', 0)
        
        output.append(f"\n[Rank {rank}] Score: {score:.3f} | Video: {video_id}")
        output.append(f"Time: {start_time:.1f}s")
        output.append(f"Text: {text[:150]}{'...' if len(text) > 150 else ''}")
        output.append("-" * 60)
    
    return "\n".join(output)


# Example usage for testing
if __name__ == "__main__":
    # Example: Retrieve from a specific video
    video_id = "YOUR_VIDEO_ID"
    question = "What is machine learning?"
    
    # Load embedding model
    model = get_embedding_model()
    embed_fn = lambda text: model.encode([text])[0]
    
    try:
        # Retrieve top-5 chunks
        results = retrieve_top_k_from_video(
            question=question,
            video_id=video_id,
            k=5,
            embed_function=embed_fn
        )
        
        print(f"Question: {question}")
        print(format_retrieval_results(results))
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nFirst run: python app.py process <video_url> --embed")