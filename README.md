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
- Optional web interface or Chrome extension

![YouTube Video Q&A System Architecture](system_architecture.png)

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