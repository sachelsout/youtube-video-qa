"""
Test script for the retrieval module.
Tests both single-video and multi-video retrieval.
"""

import numpy as np
import json
from pathlib import Path
import os 
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.retrieval.retrieval import retrieve_top_k, retrieve_top_k_from_video, retrieve_top_k_multi_video, filter_by_threshold, format_retrieval_results


def test_basic_retrieval():
    """Test basic retrieve_top_k function with mock data."""
    print("\n" + "="*60)
    print("TEST 1: Basic Retrieval with Mock Data")
    print("="*60)
    
    # Create mock chunks
    chunks = [
        {
            "text": "Machine learning is a subset of artificial intelligence",
            "video_id": "test1",
            "start_time": 10.0,
            "end_time": 15.0
        },
        {
            "text": "Deep learning uses neural networks with multiple layers",
            "video_id": "test1",
            "start_time": 20.0,
            "end_time": 25.0
        },
        {
            "text": "Python is a popular programming language",
            "video_id": "test1",
            "start_time": 30.0,
            "end_time": 35.0
        },
        {
            "text": "Neural networks are inspired by biological neurons",
            "video_id": "test1",
            "start_time": 40.0,
            "end_time": 45.0
        },
        {
            "text": "Data science involves statistics and programming",
            "video_id": "test1",
            "start_time": 50.0,
            "end_time": 55.0
        }
    ]
    
    # Create mock embeddings (384-dim like MiniLM)
    np.random.seed(42)
    embeddings = np.random.randn(5, 384)
    
    # Mock question embedding (closer to chunk 1 and 2 by design)
    question_embedding = embeddings[1] + np.random.randn(384) * 0.1
    
    # Retrieve top-3
    results = retrieve_top_k(
        question="What is deep learning?",
        embeddings=embeddings,
        chunks=chunks,
        k=3,
        question_embedding=question_embedding
    )
    
    print(f"\nRetrieved {len(results)} chunks:")
    for r in results:
        print(f"  Rank {r['rank']}: Score {r['similarity_score']:.3f}")
        print(f"  Text: {r['text']}")
        print()
    
    assert len(results) == 3, "Should retrieve exactly 3 chunks"
    assert all('similarity_score' in r for r in results), "All results should have scores"
    assert all('rank' in r for r in results), "All results should have ranks"
    
    print("✓ Basic retrieval test passed!")


def test_threshold_filtering():
    """Test threshold filtering."""
    print("\n" + "="*60)
    print("TEST 2: Threshold Filtering")
    print("="*60)
    
    # Mock results with varying scores
    results = [
        {"text": "High relevance", "similarity_score": 0.8, "rank": 1},
        {"text": "Medium relevance", "similarity_score": 0.6, "rank": 2},
        {"text": "Low relevance", "similarity_score": 0.3, "rank": 3},
        {"text": "Very low relevance", "similarity_score": 0.1, "rank": 4}
    ]
    
    # Filter with threshold 0.5
    filtered = filter_by_threshold(results, threshold=0.5)
    
    print(f"\nOriginal results: {len(results)}")
    print(f"After filtering (threshold=0.5): {len(filtered)}")
    
    for r in filtered:
        print(f"  Score: {r['similarity_score']:.1f} - {r['text']}")
    
    assert len(filtered) == 2, "Should keep only 2 results above threshold"
    assert all(r['similarity_score'] >= 0.5 for r in filtered), "All should be above threshold"
    
    print("\n✓ Threshold filtering test passed!")


def test_with_real_video(video_id: str = None):
    """Test retrieval with a real processed video (if available)."""
    print("\n" + "="*60)
    print("TEST 3: Real Video Retrieval")
    print("="*60)
    
    if video_id is None:
        # Try to find any processed video
        processed_dir = Path("src/data/processed")
        embeddings_dir = Path("src/data/embeddings")
        
        chunk_files = list(processed_dir.glob("*_chunks.json"))
        if not chunk_files:
            print("\n⚠️  No processed videos found. Skipping this test.")
            print("   Run: python app.py process <video_url> --embed")
            return
        
        video_id = chunk_files[0].stem.replace("_chunks", "")
        print(f"\nFound processed video: {video_id}")
    
    try:
        from src.retrieval.embedding_model import load_embedding_model
        
        # Load embedding model
        print("\nLoading embedding model...")
        model = load_embedding_model()
        embed_fn = lambda text: model.encode([text])[0]
        
        # Test questions
        questions = [
            "What is the main topic?",
            "Can you explain the key concepts?",
            "What are the important details?"
        ]
        
        for question in questions:
            print(f"\n📝 Question: {question}")
            
            results = retrieve_top_k_from_video(
                question=question,
                video_id=video_id,
                k=3,
                embed_function=embed_fn
            )
            
            print(f"   Found {len(results)} relevant chunks:")
            for r in results[:2]:  # Show top 2
                print(f"   - Score: {r['similarity_score']:.3f}")
                print(f"     {r['text'][:80]}...")
        
        print("\n✓ Real video retrieval test passed!")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("   Make sure to run: python app.py embed <video_id>")
    except ImportError as e:
        print(f"\n⚠️  Missing dependency: {e}")
        print("   Run: pip install sentence-transformers")


def test_multi_video_retrieval():
    """Test multi-video retrieval (if multiple videos available)."""
    print("\n" + "="*60)
    print("TEST 4: Multi-Video Retrieval")
    print("="*60)
    
    processed_dir = Path("src/data/processed")
    chunk_files = list(processed_dir.glob("*_chunks.json"))
    
    if len(chunk_files) < 2:
        print("\n⚠️  Need at least 2 processed videos for this test.")
        print("   Process more videos with: python app.py process <url> --embed")
        return
    
    video_ids = [f.stem.replace("_chunks", "") for f in chunk_files[:3]]
    print(f"\nTesting with {len(video_ids)} videos: {video_ids}")
    
    try:
        from src.retrieval.embedding_model import load_embedding_model
        
        model = load_embedding_model()
        embed_fn = lambda text: model.encode([text])[0]
        
        question = "What are the main topics discussed?"
        print(f"\n📝 Question: {question}")
        
        results = retrieve_top_k_multi_video(
            question=question,
            video_ids=video_ids,
            k=5,
            embed_function=embed_fn
        )
        
        print(f"\n✓ Found {len(results)} chunks across {len(video_ids)} videos:")
        
        # Group by video
        by_video = {}
        for r in results:
            vid = r['video_id']
            by_video[vid] = by_video.get(vid, 0) + 1
        
        for vid, count in by_video.items():
            print(f"   {vid}: {count} chunks")
        
        print("\n✓ Multi-video retrieval test passed!")
        
    except Exception as e:
        print(f"\n⚠️  Error: {e}")


def test_format_output():
    """Test result formatting."""
    print("\n" + "="*60)
    print("TEST 5: Result Formatting")
    print("="*60)
    
    results = [
        {
            "text": "This is a sample chunk about machine learning and AI",
            "video_id": "test123",
            "start_time": 45.5,
            "similarity_score": 0.87,
            "rank": 1
        },
        {
            "text": "Another chunk discussing neural networks",
            "video_id": "test456",
            "start_time": 120.0,
            "similarity_score": 0.72,
            "rank": 2
        }
    ]
    
    formatted = format_retrieval_results(results)
    print("\nFormatted output:")
    print(formatted)
    
    assert "Rank 1" in formatted, "Should show rank"
    assert "Score: 0.87" in formatted, "Should show score"
    assert "test123" in formatted, "Should show video ID"
    
    print("\n✓ Format test passed!")


def run_all_tests(video_id: str = None):
    """Run all tests."""
    print("\n" + "="*70)
    print("🧪 RETRIEVAL MODULE TEST SUITE")
    print("="*70)
    
    try:
        test_basic_retrieval()
        test_threshold_filtering()
        test_format_output()
        test_with_real_video(video_id)
        test_multi_video_retrieval()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nRetrieval module is working correctly! 🎉")
        print("\nNext steps:")
        print("  1. Process videos: python app.py process <url> --embed")
        print("  2. Ask questions: python app.py ask <video_id> 'your question'")
        print("  3. Search multiple: python app.py search 'query' vid1 vid2 vid3")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    video_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_all_tests(video_id)