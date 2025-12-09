"""
Baseline QA system using keyword matching and TF-IDF retrieval.

This module implements a simple question-answering system that retrieves
relevant transcript chunks using TF-IDF vectorization and cosine similarity.

python app.py process https://www.youtube.com/watch?v=VIDEO
python src/qa/baseline_qa.py --video_id jG7dSXcfVqE_chunks --question "What is the main topic?" --transcript_dir src/data/processed
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BaselineQA: 
    def __init__(self, model_dir: str = "src/qa/models"):
        """Initialize the baseline QA system.
        
        Args:
            model_dir: Directory to store/load the TF-IDF vectorizer
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer_path = self.model_dir / "tfidf_vectorizer.pkl"
        self.vectorizer = None
        self.corpus_vectors = None
        self.chunks = None
        
    def _load_or_create_vectorizer(self) -> TfidfVectorizer:
        """Load existing vectorizer or create a new one.
        
        Returns:
            TfidfVectorizer instance
        """
        if self.vectorizer_path.exists():
            with open(self.vectorizer_path, 'rb') as f:
                return pickle.load(f)
        else:
            return TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
    
    def _save_vectorizer(self):
        """Save the fitted vectorizer."""
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def fit(self, chunks: List[Dict[str, any]]):
        """Fit the TF-IDF vectorizer on the transcript chunks.
        
        Args:
            chunks: List of transcript chunks with 'text' field
        """
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        self.vectorizer = self._load_or_create_vectorizer()
        self.corpus_vectors = self.vectorizer.fit_transform(texts)
        self._save_vectorizer()
    
    def retrieve_top_k(self, question: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """Retrieve top-k most relevant chunks for the question.
        
        Args:
            question: The question to answer
            k: Number of chunks to retrieve
            
        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        if self.vectorizer is None or self.corpus_vectors is None:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        question_vector = self.vectorizer.transform([question])
        
        similarities = cosine_similarity(question_vector, self.corpus_vectors).flatten()
        
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append((self.chunks[idx], similarities[idx]))
        
        return results
    
    def answer(self, question: str, top_k: int = 3) -> str:
        """Generate an answer by retrieving and concatenating relevant chunks.
        
        Args:
            question: The question to answer
            top_k: Number of chunks to retrieve
            
        Returns:
            Answer string composed of relevant transcript chunks
        """
        relevant_chunks = self.retrieve_top_k(question, k=top_k)
        
        if not relevant_chunks or relevant_chunks[0][1] < 0.01:
            return "I couldn't find relevant information to answer this question."
        
        answer_parts = []
        answer_parts.append(f"Based on the transcript, here are the relevant sections:\n")
        
        for i, (chunk, score) in enumerate(relevant_chunks, 1):
            if score > 0.01:
                timestamp = chunk.get('start', 0)
                text = chunk['text'].strip()
                answer_parts.append(f"\n[{format_timestamp(timestamp)}] (relevance: {score:.2f})")
                answer_parts.append(f"{text}\n")
        
        return "".join(answer_parts)


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def baseline_answer(question: str, transcript_chunks: List[Dict[str, any]]) -> str:
    """Baseline answer function using TF-IDF retrieval.
    
    Args:
        question: The question to answer
        transcript_chunks: List of transcript chunks
        
    Returns:
        Generated answer string
    """
    qa_system = BaselineQA()
    qa_system.fit(transcript_chunks)
    return qa_system.answer(question)


def load_transcript(video_id: str, transcript_dir: str = "data/transcripts") -> List[Dict]:
    """Load transcript chunks from JSON file.
    
    Args:
        video_id: YouTube video ID
        transcript_dir: Directory containing transcript files
        
    Returns:
        List of transcript chunks
    """
    transcript_path = Path(transcript_dir) / f"{video_id}.json"
    
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'chunks' in data:
        return data['chunks']
    elif isinstance(data, dict) and 'transcript' in data:
        return data['transcript']
    else:
        raise ValueError(f"Unexpected transcript format in {transcript_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline QA system using TF-IDF retrieval"
    )
    parser.add_argument(
        "--video_id",
        type=str,
        required=True,
        help="YouTube Vid ID"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to answer"
    )
    parser.add_argument(
        "--transcript_dir",
        type=str,
        default="data/transcripts",
        help="Directory containing transcript files"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of relevant chunks to retrieve"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Loading transcript for video: {args.video_id}")
        chunks = load_transcript(args.video_id, args.transcript_dir)
        print(f"Loaded {len(chunks)} transcript chunks\n")
        
        print("Initializing baseline QA system...")
        qa_system = BaselineQA()
        qa_system.fit(chunks)
        
        print(f"Question: {args.question}\n")
        answer = qa_system.answer(args.question, top_k=args.top_k)
        print("Answer:")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nMake sure the transcript file exists at: data/transcripts/{args.video_id}.json")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
