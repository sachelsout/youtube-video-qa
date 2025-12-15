"""
Evaluation script comparing Baseline vs. LLM QA models.

Loads evaluation dataset, runs both models on all questions,
and outputs metrics (EM, F1, ROUGE-L) to CSV.

Usage:
    python -m evaluation.run_evaluation
"""

import json
import csv
import logging
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.get_transcript import get_transcript
from src.preprocessing.preprocess import preprocess_transcript
from src.retrieval.retrieval import retrieve_top_k
from src.retrieval.embedding_model import embed_chunks, load_embedding_model, load_embeddings
from src.qa.baseline_qa import BaselineQA
from src.qa.llm_qa import generate_answer, format_context
from evaluation.metrics import compute_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Runs evaluation comparing baseline and LLM QA systems."""
    
    def __init__(self):
        """Initialize QA systems and embedding model."""
        self.baseline_qa = BaselineQA()
        self.embedding_model = load_embedding_model()
        
        # Cache for video data to avoid reprocessing
        self.video_cache = {}
    
    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Load evaluation dataset from JSON.
        
        Args:
            dataset_path: Path to evaluation/dataset.json
            
        Returns:
            Dataset dictionary keyed by video_id
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Loaded dataset with {len(dataset)} videos")
        return dataset
    
    def prepare_video_data(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Prepare chunks and embeddings for a video.
        
        Full pipeline: download transcript → preprocess → generate embeddings.
        Uses cached data if available.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with chunks and embeddings, or None if error
        """
        if video_id in self.video_cache:
            return self.video_cache[video_id]
        
        logger.info(f"Preparing video data: {video_id}")
        
        try:
            chunks_path = Path("src/data/processed") / f"{video_id}_chunks.json"
            
            # Step 1: Load or download transcript
            raw_path = Path("src/data/raw") / f"{video_id}.json"
            
            if not raw_path.exists():
                logger.info(f"Downloading transcript for {video_id}")
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download transcript from YouTube
                result = get_transcript(video_id)
                
                if result.get('status') != 'success':
                    logger.error(f"Failed to download transcript: {result.get('error')}")
                    return None
                
                # Save raw transcript
                with open(raw_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved raw transcript to {raw_path}")
            
            # Step 2: Preprocess to create chunks
            if not chunks_path.exists():
                logger.info(f"Preprocessing transcript for {video_id}")
                
                chunks_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Preprocess: clean and chunk the transcript
                result = preprocess_transcript(
                    video_id=video_id,
                    max_tokens=250,
                    overlap=50,
                    merge_gap=0.0,
                    input_dir="src/data/raw",
                    output_dir="src/data/processed"
                )
                
                logger.info(f"Created {len(result['chunks'])} chunks for {video_id}")
            else:
                logger.info(f"Loading pre-processed chunks from {chunks_path}")
            
            # Load chunks
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            chunks = chunks_data.get('chunks', [])
            
            if not chunks:
                logger.warning(f"No chunks found in {chunks_path}")
                return None
            
            # Step 3: Load or generate embeddings
            embeddings_path = Path("src/data/embeddings") / f"{video_id}.npy"
            
            if embeddings_path.exists():
                logger.info(f"Loading pre-computed embeddings from {embeddings_path}")
                embeddings, _ = load_embeddings(video_id)
            else:
                logger.info(f"Generating embeddings for {len(chunks)} chunks")
                embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                embeddings = embed_chunks(chunks, model=self.embedding_model, show_progress=False)
            
            video_data = {
                'chunks': chunks,
                'embeddings': embeddings
            }
            
            self.video_cache[video_id] = video_data
            return video_data
            
        except Exception as e:
            logger.error(f"Error preparing video data for {video_id}: {e}", exc_info=True)
            return None
    
    def answer_baseline(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Get baseline QA answer using TF-IDF retrieval.
        
        Args:
            question: Question to answer
            chunks: List of transcript chunks
            
        Returns:
            Predicted answer string
        """
        try:
            # Fit baseline QA on chunks
            self.baseline_qa.fit(chunks)
            
            # Get answer using top-3 chunks
            answer = self.baseline_qa.answer(question, top_k=3)
            return answer
            
        except Exception as e:
            logger.warning(f"Baseline QA error: {e}")
            return ""
    
    def answer_llm(self, question: str, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> str:
        """
        Get LLM QA answer using embedding-based retrieval.
        
        Args:
            question: Question to answer
            chunks: List of transcript chunks
            embeddings: Precomputed chunk embeddings
            
        Returns:
            Predicted answer string
        """
        try:
            # Embed question
            question_embedding = self.embedding_model.encode([question], convert_to_numpy=True)[0]
            
            # Retrieve top-5 relevant chunks
            retrieved = retrieve_top_k(
                question=question,
                embeddings=embeddings,
                chunks=chunks,
                k=5,
                question_embedding=question_embedding
            )
            
            # Format context for LLM
            context = format_context(retrieved, max_chunks=5)
            
            # Generate answer using LLM
            result = generate_answer(
                question=question,
                retrieved_chunks=retrieved,
                llm_provider="openrouter",
                model="mistralai/mistral-7b-instruct:free",
                temperature=0.7,
                max_tokens=300
            )
            
            return result.get('answer', '')
            
        except Exception as e:
            logger.warning(f"LLM QA error: {e}")
            return ""
    
    def run_evaluation(self, dataset_path: str, output_path: str) -> None:
        """
        Run full evaluation comparing baseline and LLM on all questions.
        
        Args:
            dataset_path: Path to evaluation/dataset.json
            output_path: Path to output CSV file
        """
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        # Prepare results
        results = []
        videos_evaluated = 0
        questions_evaluated = 0
        
        for video_id, video_data in dataset.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Evaluating video: {video_id}")
            logger.info(f"{'='*70}")
            
            # Prepare video data (chunks and embeddings)
            prep_data = self.prepare_video_data(video_id)
            if prep_data is None:
                logger.warning(f"Skipping video {video_id} due to data preparation error")
                continue
            
            videos_evaluated += 1
            chunks = prep_data['chunks']
            embeddings = prep_data['embeddings']
            
            # Evaluate each question
            questions = video_data.get('questions', [])
            logger.info(f"Evaluating {len(questions)} questions for this video")
            
            for idx, qa_pair in enumerate(questions, 1):
                question = qa_pair['q']
                gold_answer = qa_pair['a']
                
                questions_evaluated += 1
                
                logger.info(f"\n[{idx}/{len(questions)}] Q: {question[:70]}")
                
                # Get baseline answer
                baseline_answer = self.answer_baseline(question, chunks)
                
                # Get LLM answer
                llm_answer = self.answer_llm(question, chunks, embeddings)
                
                # Compute metrics
                baseline_em, baseline_f1, baseline_rouge = compute_metrics(
                    baseline_answer, gold_answer
                )
                llm_em, llm_f1, llm_rouge = compute_metrics(
                    llm_answer, gold_answer
                )
                
                # Log metrics
                logger.info(f"  Gold: {gold_answer[:70]}")
                logger.info(f"  Baseline - EM: {baseline_em}, F1: {baseline_f1:.3f}, ROUGE-L: {baseline_rouge:.3f}")
                logger.info(f"  LLM      - EM: {llm_em}, F1: {llm_f1:.3f}, ROUGE-L: {llm_rouge:.3f}")
                
                # Store result
                results.append({
                    'video_id': video_id,
                    'question': question,
                    'gold_answer': gold_answer,
                    'baseline_answer': baseline_answer[:500],  # Truncate for CSV
                    'baseline_em': baseline_em,
                    'baseline_f1': baseline_f1,
                    'baseline_rouge': baseline_rouge,
                    'llm_answer': llm_answer[:500],  # Truncate for CSV
                    'llm_em': llm_em,
                    'llm_f1': llm_f1,
                    'llm_rouge': llm_rouge,
                })
        
        # Compute and log aggregated metrics
        if not results:
            logger.error("No evaluation results to save!")
            return
        
        logger.info(f"\n{'='*70}")
        logger.info(f"EVALUATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Videos evaluated: {videos_evaluated}")
        logger.info(f"Questions evaluated: {questions_evaluated}")
        
        baseline_em_avg = sum(r['baseline_em'] for r in results) / len(results)
        baseline_f1_avg = sum(r['baseline_f1'] for r in results) / len(results)
        baseline_rouge_avg = sum(r['baseline_rouge'] for r in results) / len(results)
        
        llm_em_avg = sum(r['llm_em'] for r in results) / len(results)
        llm_f1_avg = sum(r['llm_f1'] for r in results) / len(results)
        llm_rouge_avg = sum(r['llm_rouge'] for r in results) / len(results)
        
        logger.info(f"\n{'─'*70}")
        logger.info(f"BASELINE QA (TF-IDF):")
        logger.info(f"  Exact Match:  {baseline_em_avg:.3f}")
        logger.info(f"  F1 Score:     {baseline_f1_avg:.3f}")
        logger.info(f"  ROUGE-L:      {baseline_rouge_avg:.3f}")
        
        logger.info(f"\nLLM QA (Embeddings + LLM):")
        logger.info(f"  Exact Match:  {llm_em_avg:.3f}")
        logger.info(f"  F1 Score:     {llm_f1_avg:.3f}")
        logger.info(f"  ROUGE-L:      {llm_rouge_avg:.3f}")
        logger.info(f"{'─'*70}")
        
        # Save results to CSV
        logger.info(f"\nSaving detailed results to {output_path}")
        
        fieldnames = [
            'video_id', 'question', 'gold_answer',
            'baseline_answer', 'baseline_em', 'baseline_f1', 'baseline_rouge',
            'llm_answer', 'llm_em', 'llm_f1', 'llm_rouge'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"✓ Results saved to {output_path}")
        
        # Save summary statistics
        summary_path = output_dir / 'summary.json'
        summary = {
            'evaluation_date': str(Path(__file__).parent),
            'total_videos': videos_evaluated,
            'total_questions': questions_evaluated,
            'baseline': {
                'exact_match': baseline_em_avg,
                'f1_score': baseline_f1_avg,
                'rouge_l': baseline_rouge_avg,
                'description': 'TF-IDF based keyword retrieval'
            },
            'llm': {
                'exact_match': llm_em_avg,
                'f1_score': llm_f1_avg,
                'rouge_l': llm_rouge_avg,
                'description': 'Embedding-based retrieval + LLM generation'
            }
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Summary saved to {summary_path}")


def main():
    """Run evaluation pipeline."""
    # Paths
    dataset_path = Path(__file__).parent / 'dataset.json'
    output_path = Path(__file__).parent / 'results' / 'results.csv'
    
    # Run evaluation
    runner = EvaluationRunner()
    runner.run_evaluation(str(dataset_path), str(output_path))


if __name__ == '__main__':
    main()
