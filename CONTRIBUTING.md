# Contributing to YouTube Video Q&A System

Thank you for your interest in contributing to this project! We welcome contributions of all kinds, including bug fixes, feature additions, documentation improvements, and more.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Please be respectful, inclusive, and collaborative in all interactions. We aim to create a welcoming environment for everyone.

## Getting Started

### Prerequisites

Before setting up the development environment, ensure you have:
- **Python 3.8+** installed
- **pip** package manager
- **git** for version control
- A code editor (VS Code, PyCharm, etc.)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/youtube-video-qa.git
   cd youtube-video-qa
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/sachelsout/youtube-video-qa.git
   ```

## Development Setup

### 1. Create a Virtual Environment

We recommend using a virtual environment to isolate project dependencies:

```bash
# Using venv (built-in)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Core dependencies: YouTube Transcript API, sentence-transformers, torch
- Development tools: pytest, black, flake8, mypy
- Web framework: FastAPI, uvicorn

### 3. Environment Configuration

Create a `.env` file in the project root if needed for any API keys or configuration:

```bash
# Example .env (adjust based on your setup)
YOUTUBE_API_KEY=your_key_here
```

### 4. Verify Installation

Run the tests to ensure everything is set up correctly:

```bash
pytest tests/ -v
```

## Project Structure

Understanding the project layout will help you navigate and contribute effectively:

```
youtube-video-qa/
├── src/                    # Main application code
│   ├── data/              # YouTube transcript fetching and processing
│   │   ├── get_transcript.py
│   │   ├── embeddings/    # Pre-computed embeddings
│   │   ├── processed/     # Cleaned, chunked transcripts
│   │   └── raw/           # Raw transcript JSON files
│   ├── preprocessing/     # Transcript cleaning and chunking
│   │   ├── clean_transcript.py
│   │   ├── chunk_transcript.py
│   │   └── preprocess.py
│   ├── retrieval/         # Embedding-based retrieval
│   │   ├── embedding_model.py
│   │   └── retrieval.py
│   ├── qa/                # QA models and prompts
│   │   ├── baseline_qa.py
│   │   ├── llm_qa.py
│   │   └── prompts/
│   └── interface/         # Web UI (FastAPI)
│       ├── app.py
│       ├── templates/     # HTML templates
│       └── static/        # CSS and JavaScript
├── tests/                 # Unit and integration tests
├── evaluation/            # Evaluation scripts and metrics
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks for exploration
├── requirements.txt       # Python dependencies
├── README.md             # Project overview
├── CONTRIBUTING.md       # This file
└── .gitignore            # Git ignore rules
```

### Key Modules

- **`src/data/get_transcript.py`**: Fetches YouTube transcripts using the YouTube Transcript API
- **`src/preprocessing/`**: Cleans transcripts and chunks them into manageable segments
- **`src/retrieval/embedding_model.py`**: Uses sentence-transformers for semantic embeddings
- **`src/qa/baseline_qa.py`**: TF-IDF keyword-based question answering
- **`src/qa/llm_qa.py`**: LLM-powered retrieval-augmented QA
- **`src/interface/app.py`**: FastAPI web application with modern UI
- **`evaluation/`**: Performance metrics, error analysis, and evaluation scripts

## Making Changes

### 1. Create a Feature Branch

Always work on a separate branch for your changes:

```bash
# Update your local repository
git fetch upstream
git rebase upstream/main

# Create a new branch
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/issue-description
```

Use descriptive branch names:
- `feature/sentiment-analysis` for new features
- `fix/transcript-encoding-issue` for bug fixes
- `docs/update-readme` for documentation

### 2. Make Your Changes

- Keep changes focused and related to a single feature or bug fix
- Write clear, descriptive commit messages:
  ```bash
  git commit -m "Add support for multiple languages in preprocessing"
  ```
- Reference issue numbers in commits when applicable:
  ```bash
  git commit -m "Fix transcript parsing error (#42)"
  ```

### 3. Keep Your Branch Updated

As you work, keep your branch in sync with the upstream main branch:

```bash
git fetch upstream
git rebase upstream/main
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific module
pytest tests/test_preprocessing.py -v

# Run tests with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Writing Tests

When adding new features or fixing bugs, include tests:

```python
# tests/test_my_feature.py
import pytest
from src.my_module import my_function

def test_my_function_basic():
    result = my_function("input")
    assert result == "expected_output"

def test_my_function_edge_case():
    with pytest.raises(ValueError):
        my_function(None)
```

### Test File Organization

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test components working together
- Place tests in the `tests/` directory with names like `test_*.py`

## Code Quality

### Code Style

This project uses **black** for code formatting and **flake8** for linting:

```bash
# Format your code with black
black src/ tests/

# Check code style with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### Best Practices

1. **Write meaningful variable and function names**
   ```python
   # Good
   def extract_video_id_from_url(url: str) -> str:
       pass
   
   # Avoid
   def extract(u: str) -> str:
       pass
   ```

2. **Add docstrings to functions and classes**
   ```python
   def chunk_transcript(transcript: str, chunk_size: int) -> list[str]:
       """Split transcript into chunks of specified size.
       
       Args:
           transcript: Full transcript text
           chunk_size: Maximum chunk length in characters
           
       Returns:
           List of transcript chunks
       """
   ```

3. **Use type hints**
   ```python
   def get_embeddings(text: str) -> list[float]:
       pass
   ```

4. **Keep functions focused and testable**
   - Each function should have a single responsibility
   - Avoid side effects when possible
   - Extract testable units from complex logic

5. **Add comments for non-obvious logic**
   ```python
   # Skip chunks that are only whitespace or too short
   if len(chunk.strip()) < MIN_CHUNK_LENGTH:
       continue
   ```

## Submitting Changes

### 1. Push Your Changes

```bash
git push origin feature/your-feature-name
```

### 2. Create a Pull Request

1. Go to your fork on GitHub
2. Click "Compare & pull request"
3. Ensure the base repository is `sachelsout/youtube-video-qa` and base branch is `main`
4. Fill in the PR description following the template:
   ```markdown
   ## Description
   Brief description of what this PR does.
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Related Issues
   Closes #(issue number)
   
   ## Testing
   Description of testing performed.
   
   ## Checklist
   - [ ] My code follows the project's style guidelines
   - [ ] I have added tests for my changes
   - [ ] All tests pass (`pytest tests/`)
   - [ ] I have updated relevant documentation
   - [ ] My commit messages are clear and descriptive
   ```

### 3. Code Review

- Respond to feedback constructively
- Make requested changes in new commits (don't force-push, maintain history)
- Request re-review once changes are addressed

### 4. Merge

Once approved, a maintainer will merge your PR. Congratulations! 🎉

## Reporting Issues

### Before Creating an Issue

1. Check existing issues to see if it's already reported
2. Search closed issues in case there's a resolution
3. Verify you're using the latest version: `pip install -U -r requirements.txt`

### Creating an Issue

Use a clear, descriptive title and include:

```markdown
## Description
A clear description of what the issue is.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen?

## Actual Behavior
What actually happens?

## Environment
- OS: Windows/macOS/Linux
- Python version: 3.9/3.10/3.11
- Relevant packages: (list versions if applicable)

## Additional Context
Any other context, screenshots, or error messages.
```

## Getting Help

- **Questions?** Check the [README.md](README.md) and existing issues
- **Need guidance?** Open a discussion or comment on an issue
- **Found a security issue?** Please email the maintainers privately instead of using the issue tracker

## Development Workflow Summary

1. Fork → Clone → Create branch
2. Make changes → Test locally
3. Format code with `black` and check with `flake8`
4. Run `pytest` to ensure all tests pass
5. Commit with clear messages
6. Push and create a pull request
7. Respond to feedback and iterate
8. Celebrate your contribution! 🚀

Thank you for contributing to make this project better!
