# YouTube Video Q&A System

## Overview
This project builds a system that allows users to input a YouTube video URL and ask natural-language questions about its content. The system retrieves the video transcript, preprocesses it, and answers queries using both a keyword-based baseline and an improved LLM-powered retrieval model.

## Features
- Automatic YouTube transcript extraction  
- Transcript cleaning and segmentation  
- Baseline keyword-based QA  
- Improved LLM + embeddings retrieval QA  
- Quantitative and qualitative evaluation  
- Error analysis and ethical considerations  
- **Modern web interface** with dual QA modes, dark mode, and user controls  
- Conversational Q&A with separate conversation histories per mode  
- Configurable retrieval parameters (1-20 chunks) for flexible results

![YouTube Video Q&A System Architecture](docs/system_architecture.png)

## Project Goals
- Compare baseline vs. LLM-based QA performance  
- Analyze system errors and limitations  
- Evaluate how well models handle long, noisy transcripts  
- Demonstrate usability through an optional interface

## Tech Stack
- Python  
- YouTube Transcript API  
- Embedding models  
- LLM API (open-source or hosted)  
- FastAPI / Flask (optional interface)

## Web Interface

A **modern FastAPI demo interface** is included in `src/interface` with the following features:

### Getting Started

After installing dependencies:

```bash
pip install -r requirements.txt
uvicorn src.interface.app:app --reload
```

Then open `http://127.0.0.1:8000/` in your browser.

### Features

**Setup View:**
- Enter a YouTube video URL to get started
- Load button triggers transcript extraction and preprocessing

**Chat Interface:**
- **Dual QA Modes**: Switch between Baseline (TF-IDF) and LLM (embeddings + model) seamlessly
- **Separate Conversations**: Each mode maintains its own conversation history
- **Configurable Retrieval**: Use the chunks slider (1–20) to control how many transcript segments are retrieved for each question
- **Video Thumbnail**: Displays the video's thumbnail with YouTube link, making it easy to switch between the Q&A and the video
- **Message History**: View your questions and the system's responses in a clean chat interface
- **Dark Mode**: Toggle between light and dark themes (preference saved to your browser)

**Controls:**
- **Ask Button**: Send your question to the current QA mode
- **Clear Button**: Clear conversation history for the current mode
- **🎬 New Video Button**: Load a different YouTube video
- **Dark Mode Toggle**: Switch between light and dark themes (🌙/☀️ icon in header)

### API Endpoints

- `GET /` - Main web interface
- `POST /api/transcribe` - Fetch and process YouTube transcript
  - Request: `{ "video_url": "..." }`
  - Response: Video metadata and processing status
- `POST /api/ask` - Get QA response
  - Request: `{ "video_id": "...", "question": "...", "mode": "baseline|llm", "chunks_k": 5 }`
  - Response: `{ "answer": "...", "chunks": [...], "mode": "..." }`

### Technical Details

**Frontend Stack:**
- HTML5 with Jinja2 templating
- Vanilla JavaScript with state management (appState object)
- CSS3 with CSS variables for light/dark theme support
- Responsive design (mobile-first, supports 320px–1920px widths)

**Backend:**
- FastAPI with async/await support
- Session-based video storage
- Real-time transcript processing with progress feedback
- Integrated embedding and LLM inference

**Styling:**
- Modern gradient buttons with hover effects
- Smooth dark mode transitions with localStorage persistence
- Animated message appearances
- Mobile-optimized layout with proper spacing and typography

Responsive UI Design
-------------------

The web interface is fully responsive and tested across device sizes:

- **Desktop (1024px+)**: Full chat layout with optimal spacing
- **Tablet (768px–1024px)**: Adjusted padding and message bubbles
- **Mobile (320px–768px)**: Stacked layout, thumb-friendly button sizes, full-width input

To test responsiveness:
- Open `http://127.0.0.1:8000/` in any modern browser
- Use browser DevTools (F12) → Device Toolbar to emulate phones/tablets
- Try different orientations to verify layout stability

Dark Mode
---------

The interface includes a professional dark mode that:
- Toggles via the 🌙/☀️ button in the header
- Persists across browser sessions using localStorage
- Uses CSS variables for seamless theme switching
- Applies to all UI elements (buttons, text, backgrounds, borders)

Click the dark mode toggle to instantly switch themes—your preference is saved automatically.

Conversation Management
-----------------------

The interface supports rich conversation management:

**Per-Mode Conversations:**
- Baseline and LLM modes maintain completely separate conversation histories
- Switch between modes without losing your conversation context
- Each mode tracks its own question-answer pairs independently

**User Controls:**
- **Chunks Slider (1–20)**: Adjust how many transcript segments are retrieved for better or broader context
  - Lower values (1–5): Focus on most relevant answers
  - Higher values (10–20): Include more context for comprehensive understanding
- **Clear Conversation**: Reset conversation for the current mode
- **New Video**: Load a different video and start fresh (clears both conversation histories)

**Conversation Storage:**
- Currently stored in your browser only (localStorage and session memory)
- Conversations persist during your session; refresh the page to maintain history
- Close the browser to clear all conversation data (by design, for privacy)

## Troubleshooting

### YouTube Transcript API

If you see warnings about missing `list_transcripts` API during startup:

```powershell
pip install -U youtube-transcript-api
```

This ensures you have the latest version with full API support.

### Embedding Model Download

The embedding model (sentence-transformers/all-MiniLM-L6-v2) downloads automatically on first use (~50MB). This may take a minute on first run.

### FastAPI Server Issues

If the server fails to start:
1. Ensure port 8000 is available: `netstat -ano | findstr :8000` (PowerShell on Windows)
2. Try a different port: `uvicorn src.interface.app:app --port 8001 --reload`
3. Check that all dependencies in `requirements.txt` are installed

## Planned Repository Structure

```css
youtube-video-qa/
  ├── src/
  │    ├── data/
  │    ├── preprocessing/
  │    ├── retrieval/
  │    ├── qa/
  │    └── interface/
  ├── notebooks/
  ├── evaluation/
  ├── docs/
  ├── tests/
  ├── requirements.txt
  ├── README.md
  └── LICENSE
```

## Team
- [Kaiwei Hsu](https://github.com/hsu-github)

- [Leonidas Fafoutis](https://github.com/LeoFafoutis)

- [Noah Shaw](https://github.com/NoahShaw99)

- [Rohan Dawkhar](https://github.com/sachelsout)

## How to Run
Detailed setup instructions will be added as implementation progresses.