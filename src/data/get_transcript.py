"""
YouTube Transcript Retrieval Module
Extracts and saves transcripts from YouTube videos.
"""

import os
import json
import sys
import re
from typing import Dict, List, Optional
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)


def extract_video_id(video_url: str) -> Optional[str]:
    """
    Extract video ID from various YouTube URL formats.
    
    Args:
        video_url: YouTube video URL
        
    Returns:
        Video ID string or None if invalid
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)
    
    # If no pattern matches, assume the input is already a video ID
    if re.match(r'^[a-zA-Z0-9_-]{11}$', video_url):
        return video_url
    
    return None


def get_transcript(video_url: str, languages: List[str] = ['en']) -> Dict:
    """
    Retrieve transcript from YouTube video.
    
    Args:
        video_url: YouTube video URL or video ID
        languages: List of preferred language codes (default: ['en'])
        
    Returns:
        Dictionary containing:
            - video_id: str
            - transcript: List of segment dicts with 'text', 'start', 'duration'
            - language: str (language code of retrieved transcript)
            - status: str ('success' or 'error')
            - error: str (error message if status is 'error')
    """
    result = {
        'video_id': None,
        'transcript': [],
        'language': None,
        'status': 'error',
        'error': None
    }
    
    # Extract video ID
    video_id = extract_video_id(video_url)
    if not video_id:
        result['error'] = f"Invalid YouTube URL: {video_url}"
        return result
    
    result['video_id'] = video_id
    
    try:
        import logging
        logger = logging.getLogger(__name__)
        # Try the straightforward fetch first
        try:
            ytt_api = YouTubeTranscriptApi()
            fetched_transcript = ytt_api.fetch(video_id, languages=languages)

            # Extract snippets from FetchedTranscript object (newer api)
            try:
                snippets = fetched_transcript.snippets
                result['transcript'] = [
                    {
                        'text': snippet.text,
                        'start': snippet.start,
                        'duration': snippet.duration
                    }
                    for snippet in snippets
                ]
                result['language'] = fetched_transcript.language_code
                result['status'] = 'success'
                return result
            except Exception:
                # Older API returns a list of dicts
                if isinstance(fetched_transcript, list):
                    result['transcript'] = [
                        {
                            'text': s.get('text', ''),
                            'start': s.get('start', 0),
                            'duration': s.get('duration', 0)
                        }
                        for s in fetched_transcript
                    ]
                    # Try to detect language if available
                    result['language'] = None
                    result['status'] = 'success'
                    return result
                else:
                    # Fallback to using module-level get_transcript
                    pass

        except NoTranscriptFound:
            # Fall through to richer fallback logic below
            pass

        # Fallbacks for older/newer versions of youtube-transcript-api
        # 1) Try get_transcript() which returns a list of dicts on many installations
        try:
            fetched = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            if isinstance(fetched, list) and fetched:
                result['transcript'] = [
                    {
                        'text': s.get('text', ''),
                        'start': s.get('start', 0),
                        'duration': s.get('duration', 0)
                    }
                    for s in fetched
                ]
                result['language'] = None
                result['status'] = 'success'
                logger.info(f"get_transcript: fetched with get_transcript(video_id, languages={languages}) - {len(result['transcript'])} segments")
                return result
        except Exception:
            # continue to next fallback
            pass
        # Additional fallback: try get_transcript without languages to fetch any available transcript
        try:
            fetched_any = YouTubeTranscriptApi.get_transcript(video_id)
            if isinstance(fetched_any, list) and fetched_any:
                result['transcript'] = [
                    {
                        'text': s.get('text', ''),
                        'start': s.get('start', 0),
                        'duration': s.get('duration', 0)
                    }
                    for s in fetched_any
                ]
                result['language'] = None
                result['status'] = 'success'
                logger.info(f"get_transcript: fetched with get_transcript(video_id) - {len(result['transcript'])} segments")
                return result
        except Exception:
            pass

        # 2) If available, try list_transcripts (newer API). If not available, give a helpful error.
        try:
            if not hasattr(YouTubeTranscriptApi, 'list_transcripts'):
                raise AttributeError("youtube-transcript-api too old: no list_transcripts()")

            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

            # Prefer English manual, then English generated, else pick first available
            candidate = None
            try:
                candidate = transcripts.find_manually_created_transcript(languages)
            except Exception:
                try:
                    candidate = transcripts.find_generated_transcript(languages)
                except Exception:
                    # pick first manually created or generated transcript
                    try:
                        candidate = transcripts.manually_created_transcripts[0]
                    except Exception:
                        try:
                            candidate = transcripts.generated_transcripts[0]
                        except Exception:
                            candidate = None

            if candidate is not None:
                fetched = candidate.fetch()
                # fetched is typically a list of dicts
                result['transcript'] = [
                    {
                        'text': s.get('text', ''),
                        'start': s.get('start', 0),
                        'duration': s.get('duration', 0)
                    }
                    for s in fetched
                ]
                # candidate may have language_code attribute
                result['language'] = getattr(candidate, 'language_code', None)
                result['status'] = 'success'
                logger.info(f"get_transcript: fetched with list_transcripts candidate - {len(result['transcript'])} segments")
                return result

            # If we didn't find any candidate transcripts, return a clear error
            result['error'] = "No transcript found for the requested language or available transcripts"
            return result
        except AttributeError as ae:
            # The installed youtube-transcript-api is likely too old — return guidance to upgrade
            try:
                ver = __import__('youtube_transcript_api').__version__
            except Exception:
                ver = 'unknown'

            result['error'] = (
                f"Transcript retrieval error: {str(ae)}. Detected youtube-transcript-api version: {ver}. "
                "Ensure you upgraded the package in the same Python environment that runs the server. "
                "Try: pip install -U youtube-transcript-api"
            )
            return result
        except Exception as e:
            # If list_transcripts or processing failed unexpectedly, return the error
            result['error'] = f"Transcript retrieval error: {str(e)}"
            return result
        
    except TranscriptsDisabled:
        result['error'] = "Transcripts are disabled for this video"
    except VideoUnavailable:
        result['error'] = "Video is unavailable"
    except NoTranscriptFound:
        result['error'] = "No transcript found for the requested language"
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
    
    return result


def save_transcript(transcript_data: Dict, output_dir: str = "src/data/raw") -> str:
    """
    Save transcript data to JSON file.
    
    Args:
        transcript_data: Dictionary returned by get_transcript()
        output_dir: Directory to save the transcript
        
    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    video_id = transcript_data['video_id']
    filename = f"{video_id}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    
    return filepath


def main():
    """
    Main function for command-line usage.
    """
    if len(sys.argv) < 2:
        print("Usage: python src/data/get_transcript.py <youtube_url>")
        print("Example: python src/data/get_transcript.py https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        sys.exit(1)
    
    video_url = sys.argv[1]
    
    print(f"Fetching transcript for: {video_url}")
    
    # Get transcript
    transcript_data = get_transcript(video_url)
    
    if transcript_data['status'] == 'error':
        print(f"Error: {transcript_data['error']}")
        sys.exit(1)
    
    # Save transcript
    filepath = save_transcript(transcript_data)
    
    print(f"Success! Transcript saved to: {filepath}")
    print(f"Video ID: {transcript_data['video_id']}")
    print(f"Language: {transcript_data['language']}")
    print(f"Segments: {len(transcript_data['transcript'])}")
    
    # Display first few segments as preview
    if transcript_data['transcript']:
        print("\nPreview (first 3 segments):")
        for i, segment in enumerate(transcript_data['transcript'][:3]):
            print(f"  [{segment['start']:.2f}s] {segment['text']}")


if __name__ == "__main__":
    main()