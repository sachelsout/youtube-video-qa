from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pydantic import BaseModel
import json
import sys
import os
import numpy as np
import traceback

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.get_transcript import get_transcript, extract_video_id, save_transcript
from src.preprocessing.preprocess import preprocess_transcript
from src.retrieval.retrieval import retrieve_top_k
from src.retrieval.embedding_model import load_embedding_model
from src.qa.baseline_qa import BaselineQA
from src.qa.llm_qa import generate_answer

# Try to import yt-dlp for getting video title
try:
    import yt_dlp
    HAS_YT_DLP = True
except ImportError:
    HAS_YT_DLP = False

app = FastAPI()

# Serve static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Set up templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=templates_dir)

# Store active sessions (video_id -> session data with transcript, chunks, embeddings, and separate conversations per mode)
active_sessions = {}

def get_video_title(video_url: str) -> str:
    """Get video title from YouTube"""
    video_id = extract_video_id(video_url)
    if not video_id:
        return video_id
    
    if HAS_YT_DLP:
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
                return info.get('title', video_id)
        except Exception as e:
            print(f"[WARNING] Could not fetch title with yt-dlp: {e}")
            return video_id
    else:
        # Fallback: use video_id as title if yt-dlp not available
        print("[WARNING] yt-dlp not installed, using video ID as title")
        return video_id

class TranscribeRequest(BaseModel):
    video_url: str

class AskRequest(BaseModel):
    video_id: str
    question: str
    mode: str  # "baseline" or "llm"
    chunks_k: int = 5  # Number of chunks to retrieve
    conversation_history: list = []  # Optional conversation history for context

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Home"})

@app.post("/api/transcribe")
async def api_transcribe(req: TranscribeRequest):
    """
    Complete pipeline: Extract transcript, preprocess, chunk, and generate embeddings
    Flow:
    1. Extract video ID from URL
    2. Download transcript from YouTube
    3. Save raw transcript to src/data/raw/
    4. Preprocess: clean, chunk, and save to src/data/processed/
    5. Generate embeddings for chunks
    6. Store in session
    """
    try:
        # Step 1: Extract video ID
        video_id = extract_video_id(req.video_url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL or video ID")
        
        # Check if already processed
        if video_id in active_sessions:
            session = active_sessions[video_id]
            return {
                "success": True,
                "video_id": video_id,
                "title": session.get("title", ""),
                "chunk_count": len(session.get("chunks", [])),
                "message": "Video already loaded"
            }
        
        # Step 2: Download transcript from YouTube
        print(f"[TRANSCRIBE] Downloading transcript for {video_id}...")
        transcript_data = get_transcript(req.video_url)
        
        if transcript_data.get("status") != "success":
            raise HTTPException(
                status_code=400, 
                detail=transcript_data.get("error", "Failed to get transcript")
            )
        
        # Save transcript to disk (needed by preprocess_transcript)
        print(f"[SAVE] Saving raw transcript for {video_id}...")
        save_transcript(transcript_data, output_dir="src/data/raw")
        
        # Step 3: Transcript is saved by get_transcript, now preprocess it
        print(f"[PREPROCESS] Preprocessing transcript for {video_id}...")
        processed_data = preprocess_transcript(
            video_id=video_id,
            max_tokens=250,
            overlap=50,
            merge_gap=0.0
        )
        
        chunks = processed_data.get("chunks", [])
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from transcript")
        
        # Step 4: Generate embeddings for all chunks
        print(f"[EMBEDDINGS] Generating embeddings for {len(chunks)} chunks...")
        embedding_model = load_embedding_model()
        
        # Extract text from chunks and compute embeddings
        chunk_texts = [c.get("text", "") for c in chunks]
        chunk_embeddings = embedding_model.encode(chunk_texts)
        
        # Get video title
        video_title = get_video_title(req.video_url)
        
        # Step 5: Store in session
        session_data = {
            "video_id": video_id,
            "title": video_title,
            "chunks": chunks,
            "embeddings": chunk_embeddings,  # numpy array
            "transcript": transcript_data.get("transcript", ""),
            "conversations": {
                "baseline": [],  # Separate conversation history for baseline mode
                "llm": []        # Separate conversation history for LLM mode
            }
        }
        active_sessions[video_id] = session_data
        
        print(f"[SUCCESS] Video {video_id} fully processed with {len(chunks)} chunks")
        
        return {
            "success": True,
            "video_id": video_id,
            "title": video_title,
            "chunk_count": len(chunks),
            "embedding_model": "sentence-transformers"
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/api/ask")
async def api_ask(req: AskRequest):
    """
    Answer a question about a video.
    Mode:
    - baseline: Use TF-IDF to retrieve relevant chunks
    - llm: Use embeddings to retrieve chunks, then feed to LLM for answer
    """
    try:
        # Validate session
        if req.video_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Video not found. Please transcribe first.")
        
        session = active_sessions[req.video_id]
        chunks = session.get("chunks", [])
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks available for this video")
        
        if req.mode == "baseline":
            # ===== BASELINE MODE: TF-IDF Retrieval =====
            print(f"[BASELINE] Processing question: {req.question}")
            print(f"[BASELINE] Total chunks available: {len(chunks)}")
            
            baseline_qa = BaselineQA()
            
            # Fit TF-IDF vectorizer on all chunks
            print(f"[BASELINE] Fitting TF-IDF vectorizer on {len(chunks)} chunks...")
            baseline_qa.fit(chunks)
            
            # Retrieve relevant chunks using TF-IDF similarity with user-defined k
            k = req.chunks_k if req.chunks_k and 1 <= req.chunks_k <= 20 else 5
            print(f"[BASELINE] Retrieving top {k} chunks...")
            relevant_results = baseline_qa.retrieve_top_k(req.question, k=k)
            print(f"[BASELINE] Retrieved {len(relevant_results)} chunks")
            
            # Log the retrieved chunks for debugging
            for i, (chunk, score) in enumerate(relevant_results):
                print(f"[BASELINE] Chunk {i+1}: score={score:.4f}, text_length={len(chunk.get('text', ''))}")
            
            # Generate answer by concatenating relevant chunks
            answer_text = baseline_qa.answer(req.question, top_k=k)
            
            # Format relevant chunks for response with clear separation
            relevant_chunks = []
            chunk_display_text = ""
            
            for chunk, score in relevant_results:
                chunk_info = {
                    "text": chunk.get("text", ""),
                    "score": float(score),
                    "timestamp": chunk.get("start_time", 0)
                }
                relevant_chunks.append(chunk_info)
                
                # Format each chunk for display
                timestamp = chunk.get("start_time", 0)
                chunk_display_text += f"**Chunk {len(relevant_chunks)}** (Relevance: {score:.2f}, Time: {timestamp}s)\n"
                chunk_display_text += f"{chunk.get('text', '')}\n\n"
            
            print(f"[BASELINE] Formatted {len(relevant_chunks)} chunks for response")
            
            # Create answer with clearly separated chunks
            if relevant_chunks:
                clean_answer = f"Based on {len(relevant_chunks)} relevant transcript sections:\n\n{chunk_display_text}"
            else:
                clean_answer = "I couldn't find relevant information to answer this question."
            
            # Store in baseline conversation history
            conversations = session.get("conversations", {"baseline": [], "llm": []})
            conversations["baseline"].append({"role": "user", "content": req.question})
            conversations["baseline"].append({"role": "assistant", "content": clean_answer})
            session["conversations"] = conversations
            
            return {
                "success": True,
                "question": req.question,
                "answer": clean_answer,
                "mode": "baseline",
                "relevant_chunks": relevant_chunks,
                "conversation_history": conversations["baseline"]
            }
        
        elif req.mode == "llm":
            # ===== LLM MODE: Embeddings + LLM Retrieval =====
            print(f"[LLM] Processing question: {req.question}")
            
            embeddings = session.get("embeddings")
            if embeddings is None:
                raise HTTPException(status_code=500, detail="Embeddings not found in session")
            
            # Get user-defined chunk count (default 5, max 20)
            k = req.chunks_k if req.chunks_k and 1 <= req.chunks_k <= 20 else 5
            
            # Retrieve relevant chunks using embeddings
            retrieved = retrieve_top_k(
                question=req.question,
                embeddings=embeddings,
                chunks=chunks,
                k=k,
                embed_function=lambda q: load_embedding_model().encode([q])[0]
            )
            
            print(f"[LLM] Retrieved {len(retrieved)} relevant chunks")
            
            # Generate answer using LLM with retrieved context
            result = generate_answer(
                question=req.question,
                retrieved_chunks=retrieved,
                llm_provider="openrouter",
                model="mistralai/mistral-7b-instruct:free"
            )
            
            # Format relevant chunks for response
            relevant_chunks = [
                {
                    "text": c.get("text", "")[:200],
                    "score": float(c.get("similarity_score", 0)),
                    "timestamp": c.get("start_time", 0)
                }
                for c in retrieved
            ]
            
            # Store in LLM conversation history
            conversations = session.get("conversations", {"baseline": [], "llm": []})
            conversations["llm"].append({"role": "user", "content": req.question})
            conversations["llm"].append({"role": "assistant", "content": result.get("answer", "")})
            session["conversations"] = conversations
            
            return {
                "success": True,
                "question": req.question,
                "answer": result.get("answer", ""),
                "mode": "llm",
                "relevant_chunks": relevant_chunks,
                "conversation_history": conversations["llm"]
            }
        
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use 'baseline' or 'llm'")
    
    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")