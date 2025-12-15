"""
Evaluation metrics for QA system.

Implements:
- Exact Match (EM): Checks if prediction and gold answer are identical (case-insensitive)
- F1 Score: Token-level overlap between prediction and gold answer
- ROUGE-L: Longest common subsequence-based metric
"""

import re
from typing import Tuple


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison: lowercase, remove punctuation, extra spaces.
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def exact_match(pred: str, gold: str) -> int:
    """
    Compute exact match score.
    
    Returns 1 if normalized prediction equals normalized gold answer, 0 otherwise.
    
    Args:
        pred: Predicted answer
        gold: Gold/reference answer
        
    Returns:
        1 if exact match, 0 otherwise
    """
    return int(normalize_text(pred) == normalize_text(gold))


def f1_score(pred: str, gold: str) -> float:
    """
    Compute F1 score based on token overlap.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    where:
    - precision = common_tokens / pred_tokens
    - recall = common_tokens / gold_tokens
    
    Args:
        pred: Predicted answer
        gold: Gold/reference answer
        
    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = set(normalize_text(pred).split())
    gold_tokens = set(normalize_text(gold).split())
    
    common_tokens = pred_tokens & gold_tokens
    
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def _lcs_length(s1: str, s2: str) -> int:
    """
    Compute longest common subsequence (LCS) length using dynamic programming.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Length of longest common subsequence
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def rouge_l(pred: str, gold: str) -> float:
    """
    Compute ROUGE-L (Longest Common Subsequence) score.
    
    ROUGE-L = 2 * (lcs_recall * lcs_precision) / (lcs_recall + lcs_precision)
    
    where:
    - lcs_recall = LCS(pred, gold) / len(gold)
    - lcs_precision = LCS(pred, gold) / len(pred)
    
    Args:
        pred: Predicted answer (as sequence of tokens)
        gold: Gold/reference answer (as sequence of tokens)
        
    Returns:
        ROUGE-L F-score between 0 and 1
    """
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    
    lcs_len = _lcs_length(pred_tokens, gold_tokens)
    
    lcs_recall = lcs_len / len(gold_tokens)
    lcs_precision = lcs_len / len(pred_tokens)
    
    if lcs_recall + lcs_precision == 0:
        return 0.0
    
    rouge_l_score = 2 * (lcs_recall * lcs_precision) / (lcs_recall + lcs_precision)
    return rouge_l_score


def compute_metrics(pred: str, gold: str) -> Tuple[int, float, float]:
    """
    Compute all metrics for a single prediction-gold pair.
    
    Args:
        pred: Predicted answer
        gold: Gold/reference answer
        
    Returns:
        Tuple of (EM, F1, ROUGE-L) scores
    """
    em = exact_match(pred, gold)
    f1 = f1_score(pred, gold)
    rouge = rouge_l(pred, gold)
    
    return em, f1, rouge
