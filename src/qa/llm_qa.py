"""
LLM-based Q&A Module for YouTube Video QA System
Generates natural language answers using retrieved context.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from dotenv import load_dotenv

load_dotenv() 

def load_prompt_template(template_path: str = "src/qa/prompts/llm_prompt.txt") -> str:
    """
    Load the prompt template from file.
    
    Args:
        template_path: Path to the prompt template file
        
    Returns:
        Prompt template as string
    """
    template_file = Path(template_path)
    
    if not template_file.exists():
        # Return default template if file doesn't exist
        return """You are a helpful assistant answering questions about a YouTube video based on transcript excerpts.

Context from video transcript:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the provided context
- If the context doesn't contain enough information, say so clearly
- Be concise but comprehensive
- Quote relevant parts when helpful
- If asked about timestamps, reference them from the context

Answer:"""
    
    with open(template_file, 'r', encoding='utf-8') as f:
        return f.read()


def format_context(chunks: List[Dict[str, Any]], max_chunks: int = 15) -> str:
    """
    Format retrieved chunks into context string for LLM.
    
    Args:
        chunks: List of retrieved chunk dictionaries
        max_chunks: Maximum number of chunks to include
        
    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant context found."
    
    context_parts = []
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        text = chunk.get('text', '')
        start_time = chunk.get('start_time', 0)
        similarity = chunk.get('similarity_score', 0)
        
        context_parts.append(
            f"[Excerpt {i}] (Timestamp: {start_time:.1f}s, Relevance: {similarity:.2f})\n{text}"
        )
    
    return "\n\n---\n\n".join(context_parts)


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    llm_provider: str = "openrouter",
    model: str = "google/gemini-2.0-flash-exp:free",
    temperature: float = 0.7,
    max_tokens: int = 500,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate an answer using an LLM based on retrieved context.
    
    Args:
        question: The user's question
        retrieved_chunks: List of relevant chunks from retrieval
        llm_provider: LLM provider ('openrouter', 'openai', 'anthropic', 'ollama')
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        api_key: API key (if not in environment)
        
    Returns:
        Dictionary with answer, metadata, and sources
    """
    # Format context
    context = format_context(retrieved_chunks)
    
    # Load and format prompt
    template = load_prompt_template()
    prompt = template.format(context=context, question=question)
    
    # Generate answer based on provider
    if llm_provider == "openrouter":
        answer = _generate_openrouter(prompt, model, temperature, max_tokens, api_key)
    elif llm_provider == "openai":
        answer = _generate_openai(prompt, model, temperature, max_tokens, api_key)
    elif llm_provider == "anthropic":
        answer = _generate_anthropic(prompt, model, temperature, max_tokens, api_key)
    elif llm_provider == "ollama":
        answer = _generate_ollama(prompt, model, temperature, max_tokens)
    else:
        answer = f"Error: Unsupported LLM provider '{llm_provider}'"
    
    return {
        'question': question,
        'answer': answer,
        'context_chunks': len(retrieved_chunks),
        'sources': retrieved_chunks,
        'model': model,
        'provider': llm_provider
    }


def _generate_openrouter(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: Optional[str]
) -> str:
    """Generate answer using OpenRouter API."""
    try:
        import requests
        
        if not api_key and not os.getenv("OPENROUTER_API_KEY"):
            return "Error: OPENROUTER_API_KEY not found. Get free key at https://openrouter.ai/keys"
        
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-username/youtube-video-qa",
                "X-Title": "YouTube Video QA"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", str(error_data))
            return f"Error from OpenRouter: {error_msg}"
    
    except ImportError:
        return "Error: requests package not installed. Run: pip install requests"
    except Exception as e:
        return f"Error calling OpenRouter: {str(e)}"


def _generate_openai(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: Optional[str]
) -> str:
    """Generate answer using OpenAI API."""
    try:
        import openai
        
        if api_key:
            openai.api_key = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            return "Error: OPENAI_API_KEY not found. Set it with: export OPENAI_API_KEY='your-key'"
        
        client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    except ImportError:
        return "Error: openai package not installed. Run: pip install openai"
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"


def _generate_anthropic(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: Optional[str]
) -> str:
    """Generate answer using Anthropic API."""
    try:
        import anthropic
        
        if not api_key and not os.getenv("ANTHROPIC_API_KEY"):
            return "Error: ANTHROPIC_API_KEY not found. Set it with: export ANTHROPIC_API_KEY='your-key'"
        
        client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text.strip()
    
    except ImportError:
        return "Error: anthropic package not installed. Run: pip install anthropic"
    except Exception as e:
        return f"Error calling Anthropic API: {str(e)}"


def _generate_ollama(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> str:
    """Generate answer using Ollama (local)."""
    try:
        import requests
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"Error: Ollama returned status {response.status_code}"
    
    except ImportError:
        return "Error: requests package not installed. Run: pip install requests"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure it's running: ollama serve"
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"


def main():
    """Command-line interface for LLM Q&A."""
    parser = argparse.ArgumentParser(
        description='Generate answers using LLM with retrieved context',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with OpenRouter (FREE!)
  python src/qa/llm_qa.py --video_id dQw4w9WgXcQ --question "What is this about?"
  
  # Use different free model
  python src/qa/llm_qa.py --video_id dQw4w9WgXcQ --question "Summarize" --model meta-llama/llama-3.1-8b-instruct:free
  
  # Use Anthropic Claude
  python src/qa/llm_qa.py --video_id dQw4w9WgXcQ --question "Summarize" --provider anthropic --model claude-3-5-sonnet-20241022
  
  # Use local Ollama
  python src/qa/llm_qa.py --video_id dQw4w9WgXcQ --question "What happens?" --provider ollama --model llama2
  
  # Custom retrieval settings
  python src/qa/llm_qa.py --video_id dQw4w9WgXcQ --question "Question?" --k 10 --threshold 0.3
        """
    )
    
    parser.add_argument('--video_id', required=True, help='YouTube video ID')
    parser.add_argument('--question', required=True, help='Question to ask')
    parser.add_argument('--k', type=int, default=5, help='Number of chunks to retrieve (default: 5)')
    parser.add_argument('--threshold', type=float, default=None, help='Minimum similarity threshold')
    parser.add_argument('--provider', default='openrouter',
                       choices=['openrouter', 'openai', 'anthropic', 'ollama'],
                       help='LLM provider (default: openrouter)')
    parser.add_argument('--model', default=None, help='Model name (default: google/gemini-2.0-flash-exp:free for OpenRouter)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature (default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=500, help='Max tokens in response (default: 500)')
    parser.add_argument('--show-context', action='store_true', help='Show retrieved context')
    
    args = parser.parse_args()
    
    # Set default model based on provider
    if args.model is None:
        if args.provider == 'openrouter':
            args.model = 'google/gemini-2.0-flash-exp:free'  # Free tier
        elif args.provider == 'openai':
            args.model = 'gpt-4o-mini'
        elif args.provider == 'anthropic':
            args.model = 'claude-3-5-sonnet-20241022'
        elif args.provider == 'ollama':
            args.model = 'llama2'
    
    try:
        # Import retrieval and embedding modules
        from src.retrieval.retrieval import retrieve_top_k_from_video, filter_by_threshold
        from src.retrieval.embedding_model import load_embedding_model
        
        print(f"\n{'='*60}")
        print(f"LLM Q&A System")
        print(f"{'='*60}\n")
        
        # Step 1: Load embedding model
        print("📊 Loading embedding model...")
        model_embed = load_embedding_model()
        embed_fn = lambda text: model_embed.encode([text])[0]
        
        # Step 2: Retrieve relevant chunks
        print(f"🔍 Retrieving context for: '{args.question}'")
        chunks = retrieve_top_k_from_video(
            question=args.question,
            video_id=args.video_id,
            k=args.k,
            embed_function=embed_fn
        )
        
        # Apply threshold if specified
        if args.threshold is not None:
            chunks = filter_by_threshold(chunks, args.threshold)
        
        print(f"✓ Found {len(chunks)} relevant chunks")
        
        if not chunks:
            print("\n⚠️  No relevant context found. Cannot generate answer.")
            sys.exit(1)
        
        # Show context if requested
        if args.show_context:
            print(f"\n{'='*60}")
            print("Retrieved Context:")
            print(f"{'='*60}")
            context = format_context(chunks)
            print(context)
        
        # Step 3: Generate answer with LLM
        print(f"\n🤖 Generating answer with {args.provider} ({args.model})...")
        result = generate_answer(
            question=args.question,
            retrieved_chunks=chunks,
            llm_provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Display results
        print(f"\n{'='*60}")
        print(f"Question: {result['question']}")
        print(f"{'='*60}")
        print(f"\n{result['answer']}")
        print(f"\n{'='*60}")
        print(f"Model: {result['provider']}/{result['model']}")
        print(f"Context chunks: {result['context_chunks']}")
        print(f"{'='*60}\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"Make sure you've processed the video and generated embeddings:")
        print(f"  python app.py process <video_url> --embed")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()