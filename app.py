"""
YouTube Video Q&A System - Main Application
Orchestrates the complete pipeline from transcript retrieval to Q&A.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import sys
import json
from pathlib import Path
from typing import Optional, Dict
import argparse

from src.data.get_transcript import get_transcript, save_transcript
from src.preprocessing.preprocess import preprocess_transcript
from src.retrieval.embedding_model import embed_video


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
        processed_dir: str = "src/data/processed"
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
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

            # Placeholder for future stages
            print("\n⏳ Step 3: Retrieval system (coming soon)...")
            print("\n⏳ Step 4: Q&A system (coming soon)...")

            result['status'] = 'success'

            print(f"\n{'='*60}")
            print("✅ Processing complete!")
            print(f"{'='*60}")
            print(f"\nVideo ID: {video_id}")
            print(f"Transcript: {result['transcript_path']}")
            print(f"Chunks: {result['chunks_path']}")

            return result

        except Exception as e:
            result['error'] = str(e)
            print(f"\n❌ Error: {e}")
            return result
    

    def ask_question(self, video_id: str, question: str) -> Dict:
        chunks_path = self.processed_dir / f"{video_id}_chunks.json"
        
        if not chunks_path.exists():
            return {
                'status': 'error',
                'error': f'No processed data found for video {video_id}. Run process_video first.'
            }

        with open(chunks_path, 'r') as f:
            data = json.load(f)
        
        return {
            'status': 'success',
            'question': question,
            'answer': 'Q&A functionality coming in future issues!',
            'method': 'placeholder',
            'chunks_available': len(data.get('chunks', []))
        }
    

    def list_processed_videos(self) -> list:
        videos = []
        for transcript_file in self.raw_dir.glob("*.json"):
            video_id = transcript_file.stem
            chunks_file = self.processed_dir / f"{video_id}_chunks.json"
            videos.append({
                'video_id': video_id,
                'has_transcript': True,
                'has_chunks': chunks_file.exists()
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
  
  # Custom settings
  python app.py process dQw4w9WgXcQ --max-tokens 500 --force
  python app.py embed dQw4w9WgXcQ --model all-mpnet-base-v2
  
  # Ask questions (placeholder)
  python app.py ask dQw4w9WgXcQ "What is this video about?"
  
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

    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a question about a video')
    ask_parser.add_argument('video_id')
    ask_parser.add_argument('question')

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
        result = qa_system.ask_question(args.video_id, args.question)
        
        print(f"\n{'='*60}")
        print(f"Question: {result['question']}")
        print(f"{'='*60}")
        print(f"\nAnswer: {result['answer']}")

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
                status = "✓ Ready" if video['has_chunks'] else "⏳ Incomplete"
                print(f"{status} - {video['video_id']}")
        
        print()


if __name__ == "__main__":
    main()