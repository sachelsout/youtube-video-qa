"""
YouTube Video Q&A System - Main Application
Orchestrates the complete pipeline from transcript retrieval to Q&A.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List
import argparse

from src.data.get_transcript import get_transcript, save_transcript
from src.preprocessing.preprocess import preprocess_transcript
from src.retrieval.embedding_model import embed_video, load_embedding_model
from src.retrieval.retrieval import retrieve_top_k_from_video, retrieve_top_k_multi_video, format_retrieval_results, filter_by_threshold
from src.qa.llm_qa import generate_answer


# ----------------------------------------------------------
# 🎵 SIMPLE MUSIC VIDEO DETECTOR
# ----------------------------------------------------------

def is_music_video(transcript_segments: list) -> bool:
    """
    Detect if the video is likely a music video based on transcript markers.
    We check for common music symbols or tags typically returned by YouTube API.
    
    Returns True if music detected.
    """
    music_markers = ["♪", "♫", "[Music]", "(Music)", "🎵", "🎶"]

    for seg in transcript_segments:
        text = seg.get("text", "")
        for marker in music_markers:
            if marker.lower() in text.lower():
                return True
    return False


class YouTubeQASystem:
    """Main application class for YouTube Video Q&A."""
    
    def __init__(
        self,
        raw_dir: str = "src/data/raw",
        processed_dir: str = "src/data/processed",
        embeddings_dir: str = "src/data/embeddings"
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.embeddings_dir = Path(embeddings_dir)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy load embedding model
        self._embedding_model = None
    
    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = load_embedding_model()
        return self._embedding_model
    
    def process_video(
        self,
        video_url: str,
        max_tokens: int = 250,
        overlap: int = 50,
        merge_gap: Optional[float] = None,
        force_redownload: bool = False,
        force_reprocess: bool = False
    ) -> Dict:

        result = {
            'video_url': video_url,
            'video_id': None,
            'transcript_path': None,
            'chunks_path': None,
            'status': 'error',
            'error': None,
            'steps_completed': []
        }
        
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {video_url}")
            print(f"{'='*60}\n")

            # ----------------------------------------------------------
            # 📥 Step 1 — Retrieve Transcript
            # ----------------------------------------------------------
            print("📥 Step 1: Retrieving transcript...")
            transcript_data = get_transcript(video_url)

            if transcript_data['status'] == 'error':
                result['error'] = f"Transcript retrieval failed: {transcript_data['error']}"
                return result
            
            video_id = transcript_data['video_id']
            result['video_id'] = video_id

            transcript_path = self.raw_dir / f"{video_id}.json"

            if transcript_path.exists() and not force_redownload:
                print(f"   ✓ Transcript already exists: {transcript_path}")
            else:
                filepath = save_transcript(transcript_data, str(self.raw_dir))
                print(f"   ✓ Transcript downloaded: {filepath}")
                print(f"   - Language: {transcript_data['language']}")
                print(f"   - Segments: {len(transcript_data['transcript'])}")

            result['transcript_path'] = str(transcript_path)
            result['steps_completed'].append('transcript_retrieval')

            # ----------------------------------------------------------
            # 🎵 AUTO-DETECT MUSIC VIDEO
            # ----------------------------------------------------------

            transcript_segments = transcript_data["transcript"]

            if merge_gap is None:
                if is_music_video(transcript_segments):
                    merge_gap = 0.0
                    print("\n🎵 Music video detected → merge_gap = 0 (NO merging)")
                else:
                    merge_gap = 1.0
                    print("\n🎬 Normal video detected → merge_gap = 1 (merge small gaps)")

            else:
                print(f"\n⚙️ merge_gap overridden manually → {merge_gap}")

            # ----------------------------------------------------------
            # 🔧 Step 2 — Preprocess transcript
            # ----------------------------------------------------------
            print("\n🔧 Step 2: Preprocessing transcript...")
            chunks_path = self.processed_dir / f"{video_id}_chunks.json"

            if chunks_path.exists() and not force_reprocess:
                print(f"   ✓ Processed chunks already exist: {chunks_path}")
                with open(chunks_path, 'r') as f:
                    preprocessing_result = json.load(f)
            else:
                preprocessing_result = preprocess_transcript(
                    video_id=video_id,
                    max_tokens=max_tokens,
                    overlap=overlap,
                    merge_gap=merge_gap,
                    input_dir=str(self.raw_dir),
                    output_dir=str(self.processed_dir)
                )
                print(f"   ✓ Preprocessing complete: {chunks_path}")
                print(f"   - Original segments: {preprocessing_result['metadata']['original_segments']}")
                print(f"   - Cleaned segments: {preprocessing_result['metadata']['cleaned_segments']}")
                print(f"   - Final chunks: {preprocessing_result['metadata']['num_chunks']}")

            result['chunks_path'] = str(chunks_path)
            result['preprocessing_metadata'] = preprocessing_result.get('metadata', {})
            result['steps_completed'].append('preprocessing')

            result['status'] = 'success'

            print(f"\n{'='*60}")
            print("✅ Processing complete!")
            print(f"{'='*60}")
            print(f"\nVideo ID: {video_id}")
            print(f"Transcript: {result['transcript_path']}")
            print(f"Chunks: {result['chunks_path']}")
            print(f"\n💡 Next: Run 'python app.py embed {video_id}' to enable Q&A")

            return result

        except Exception as e:
            result['error'] = str(e)
            print(f"\n❌ Error: {e}")
            return result
    
    def ask_question(
        self,
        video_id: str,
        question: str,
        k: int = 5,
        threshold: Optional[float] = None,
        show_sources: bool = False,
        use_llm: bool = False,
        llm_provider: str = "openrouter",
        llm_model: Optional[str] = None
    ) -> Dict:
        """
        Ask a question about a video using retrieval and optionally LLM.
        
        Args:
            video_id: YouTube video ID
            question: Question to ask
            k: Number of chunks to retrieve
            threshold: Minimum similarity threshold (optional)
            show_sources: Whether to display source chunks
            use_llm: Whether to use LLM for answer generation
            llm_provider: LLM provider ('openrouter', 'openai', 'anthropic', 'ollama')
            llm_model: LLM model name (optional)
            
        Returns:
            Dictionary with status, question, retrieved chunks, and answer
        """
        result = {
            'status': 'error',
            'question': question,
            'video_id': video_id,
            'retrieved_chunks': [],
            'answer': None,
            'error': None
        }
        
        try:
            # Load embedding model
            model = self._get_embedding_model()
            embed_fn = lambda text: model.encode([text])[0]
            
            # Retrieve relevant chunks
            print(f"\n🔍 Searching for: '{question}'")
            print(f"   Looking in video: {video_id}")
            
            chunks = retrieve_top_k_from_video(
                question=question,
                video_id=video_id,
                k=k,
                data_dir=str(self.processed_dir),
                embeddings_dir=str(self.embeddings_dir),
                embed_function=embed_fn
            )
            
            # Apply threshold if specified
            if threshold is not None:
                chunks = filter_by_threshold(chunks, threshold)
            
            result['retrieved_chunks'] = chunks
            result['status'] = 'success'
            
            # Generate LLM answer if requested
            if use_llm and chunks:
                print(f"\n🤖 Generating answer with {llm_provider}...")
                
                llm_result = generate_answer(
                    question=question,
                    retrieved_chunks=chunks,
                    llm_provider=llm_provider,
                    model=llm_model or ("google/gemini-2.0-flash-exp:free" if llm_provider == "openrouter" else "gpt-4o-mini")
                )
                
                result['answer'] = llm_result['answer']
                result['llm_model'] = llm_result['model']
                result['llm_provider'] = llm_result['provider']
                
                print(f"\n{'='*60}")
                print(f"Answer:")
                print(f"{'='*60}")
                print(f"\n{result['answer']}\n")
                
                if show_sources:
                    print(f"{'='*60}")
                    print(f"Sources:")
                    print(f"{'='*60}")
                    print(format_retrieval_results(chunks))
            
            # Simple concatenation answer if not using LLM
            elif chunks:
                context = "\n\n".join([
                    f"[{c['start_time']:.1f}s]: {c['text']}"
                    for c in chunks
                ])
                result['answer'] = context
                
                if show_sources:
                    print(f"\n✓ Found {len(chunks)} relevant chunks:")
                    print(format_retrieval_results(chunks))
            else:
                result['answer'] = "No relevant information found."
                print("\n⚠️  No relevant chunks found.")
            
            return result
            
        except FileNotFoundError as e:
            result['error'] = str(e)
            print(f"\n❌ Error: {e}")
            return result
        except Exception as e:
            result['error'] = str(e)
            print(f"\n❌ Unexpected error: {e}")
            return result
    
    def ask_multi_video(
        self,
        video_ids: List[str],
        question: str,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Ask a question across multiple videos.
        
        Args:
            video_ids: List of YouTube video IDs to search
            question: Question to ask
            k: Total number of chunks to retrieve
            threshold: Minimum similarity threshold (optional)
            
        Returns:
            Dictionary with results
        """
        try:
            # Load embedding model
            model = self._get_embedding_model()
            embed_fn = lambda text: model.encode([text])[0]
            
            print(f"\n🔍 Searching across {len(video_ids)} videos...")
            
            chunks = retrieve_top_k_multi_video(
                question=question,
                video_ids=video_ids,
                k=k,
                data_dir=str(self.processed_dir),
                embeddings_dir=str(self.embeddings_dir),
                embed_function=embed_fn
            )
            
            # Apply threshold if specified
            if threshold is not None:
                chunks = filter_by_threshold(chunks, threshold)
            
            if chunks:
                print(f"\n✓ Found {len(chunks)} relevant chunks:")
                print(format_retrieval_results(chunks))
            else:
                print("\n⚠️  No relevant chunks found.")
            
            return {
                'status': 'success',
                'question': question,
                'video_ids': video_ids,
                'retrieved_chunks': chunks
            }
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def list_processed_videos(self) -> list:
        """List all processed videos with their status."""
        videos = []
        for transcript_file in self.raw_dir.glob("*.json"):
            video_id = transcript_file.stem
            chunks_file = self.processed_dir / f"{video_id}_chunks.json"
            embeddings_file = self.embeddings_dir / f"{video_id}.npy"
            
            videos.append({
                'video_id': video_id,
                'has_transcript': True,
                'has_chunks': chunks_file.exists(),
                'has_embeddings': embeddings_file.exists(),
                'ready_for_qa': chunks_file.exists() and embeddings_file.exists()
            })
        return videos


def main():
    parser = argparse.ArgumentParser(
        description='YouTube Video Q&A System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video
  python app.py process https://www.youtube.com/watch?v=dQw4w9WgXcQ
  
  # Process and generate embeddings
  python app.py process dQw4w9WgXcQ --embed
  
  # Generate embeddings only
  python app.py embed dQw4w9WgXcQ
  
  # Ask questions (NOW WORKING!)
  python app.py ask dQw4w9WgXcQ "What is this video about?"
  python app.py ask dQw4w9WgXcQ "machine learning" --k 10 --threshold 0.3
  
  # Search across multiple videos
  python app.py search "artificial intelligence" video1 video2 video3
  
  # List all processed videos
  python app.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a YouTube video')
    process_parser.add_argument('video_url')
    process_parser.add_argument('--max-tokens', type=int, default=250)
    process_parser.add_argument('--overlap', type=int, default=50)
    process_parser.add_argument('--merge-gap', type=float, default=None,
                                help='Override automatic merge_gap detection')
    process_parser.add_argument('--force', action='store_true')
    process_parser.add_argument('--embed', action='store_true',
                                help='Also generate embeddings after processing')

    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings for a video')
    embed_parser.add_argument('video_id', help='YouTube video ID')
    embed_parser.add_argument('--model', default='all-MiniLM-L6-v2',
                             help='Embedding model name (default: all-MiniLM-L6-v2)')
    embed_parser.add_argument('--force', action='store_true',
                             help='Force regeneration of embeddings')

    # Ask command (NOW WITH RETRIEVAL!)
    ask_parser = subparsers.add_parser('ask', help='Ask a question about a video')
    ask_parser.add_argument('video_id', help='YouTube video ID')
    ask_parser.add_argument('question', help='Question to ask')
    ask_parser.add_argument('--k', type=int, default=5,
                           help='Number of chunks to retrieve (default: 5)')
    ask_parser.add_argument('--threshold', type=float, default=None,
                           help='Minimum similarity threshold (0-1)')
    ask_parser.add_argument('--no-sources', action='store_true',
                           help='Hide source chunks')
    ask_parser.add_argument('--llm', action='store_true',
                           help='Use LLM to generate natural answer')
    ask_parser.add_argument('--provider', default='openrouter',
                           choices=['openrouter', 'openai', 'anthropic', 'ollama'],
                           help='LLM provider (default: openrouter)')
    ask_parser.add_argument('--model', default=None,
                           help='LLM model name')

    # Search command (multi-video)
    search_parser = subparsers.add_parser('search', 
                                          help='Search across multiple videos')
    search_parser.add_argument('question', help='Question to search for')
    search_parser.add_argument('video_ids', nargs='+', 
                              help='Video IDs to search')
    search_parser.add_argument('--k', type=int, default=15,
                              help='Total chunks to retrieve (default: 5)')
    search_parser.add_argument('--threshold', type=float, default=None,
                              help='Minimum similarity threshold (0-1)')

    # List command
    list_parser = subparsers.add_parser('list', help='List all processed videos')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    qa_system = YouTubeQASystem()
    
    if args.command == 'process':
        result = qa_system.process_video(
            video_url=args.video_url,
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            merge_gap=args.merge_gap,
            force_redownload=args.force,
            force_reprocess=args.force
        )

        if result['status'] == 'error':
            print(f"\n❌ Error: {result['error']}")
            sys.exit(1)
        
        # Generate embeddings if requested
        if args.embed and result['video_id']:
            print(f"\n{'='*60}")
            print("📊 Step 3: Generating embeddings...")
            print(f"{'='*60}")
            try:
                embed_video(
                    video_id=result['video_id'],
                    force=args.force
                )
            except Exception as e:
                print(f"\n❌ Embedding error: {e}")
                print("Note: Run 'pip install sentence-transformers' if not installed")
    
    elif args.command == 'embed':
        # Standalone embed command
        try:
            embed_video(
                video_id=args.video_id,
                model_name=args.model,
                force=args.force
            )
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Note: Run 'pip install sentence-transformers' if not installed")
            sys.exit(1)
    
    elif args.command == 'ask':
        result = qa_system.ask_question(
            video_id=args.video_id,
            question=args.question,
            k=args.k,
            threshold=args.threshold,
            show_sources=not args.no_sources,
            use_llm=args.llm,
            llm_provider=args.provider,
            llm_model=args.model
        )
        
        print(f"\n{'='*60}")
        print(f"Question: {result['question']}")
        print(f"{'='*60}")
        
        if not args.llm:
            # Only print this if not using LLM (LLM mode prints its own output)
            pass

        if result['status'] == 'error':
            print(f"\n❌ Error: {result['error']}")
            sys.exit(1)
    
    elif args.command == 'search':
        result = qa_system.ask_multi_video(
            video_ids=args.video_ids,
            question=args.question,
            k=args.k,
            threshold=args.threshold
        )
        
        if result['status'] == 'error':
            print(f"\n❌ Error: {result['error']}")
            sys.exit(1)
    
    elif args.command == 'list':
        videos = qa_system.list_processed_videos()
        
        print(f"\n{'='*60}")
        print(f"Processed Videos ({len(videos)})")
        print(f"{'='*60}\n")
        
        if not videos:
            print("No videos processed yet.")
        else:
            for video in videos:
                if video['ready_for_qa']:
                    status = "✓ Ready for Q&A"
                elif video['has_embeddings']:
                    status = "⚠️  Has embeddings, missing chunks"
                elif video['has_chunks']:
                    status = "⏳ Needs embeddings (run: python app.py embed <id>)"
                else:
                    status = "⏳ Incomplete"
                
                print(f"{status} - {video['video_id']}")
        
        print()


if __name__ == "__main__":
    main()