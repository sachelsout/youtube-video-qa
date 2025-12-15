"""
Unit tests for evaluation metrics.
"""

import os, sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from evaluation.metrics import (
    exact_match,
    f1_score,
    rouge_l,
    compute_metrics,
    normalize_text
)


class TestNormalizeText:
    """Tests for text normalization."""
    
    def test_lowercase(self):
        """Test that text is lowercased."""
        assert normalize_text("Hello World") == "hello world"
    
    def test_punctuation_removal(self):
        """Test that punctuation is removed."""
        assert normalize_text("hello, world!") == "hello world"
    
    def test_article_removal(self):
        """Test that articles are removed."""
        text = normalize_text("a cat and the dog")
        assert "cat" in text and "dog" in text
    
    def test_extra_spaces(self):
        """Test that extra spaces are normalized."""
        assert normalize_text("hello   world") == "hello world"


class TestExactMatch:
    """Tests for exact match metric."""
    
    def test_exact_match_identical(self):
        """Test that identical strings match."""
        assert exact_match("hello world", "hello world") == 1
    
    def test_exact_match_case_insensitive(self):
        """Test that matching is case-insensitive."""
        assert exact_match("Hello World", "hello world") == 1
    
    def test_exact_match_punctuation_insensitive(self):
        """Test that matching ignores punctuation."""
        assert exact_match("hello, world!", "hello world") == 1
    
    def test_exact_match_different(self):
        """Test that different strings don't match."""
        assert exact_match("hello world", "goodbye world") == 0
    
    def test_exact_match_empty_strings(self):
        """Test that two empty strings match."""
        assert exact_match("", "") == 1


class TestF1Score:
    """Tests for F1 score metric."""
    
    def test_f1_identical(self):
        """Test that identical strings have F1=1.0."""
        assert f1_score("hello world", "hello world") == 1.0
    
    def test_f1_no_overlap(self):
        """Test that completely different strings have F1=0.0."""
        assert f1_score("hello", "goodbye") == 0.0
    
    def test_f1_partial_overlap(self):
        """Test partial token overlap."""
        score = f1_score("hello there", "hello world")
        assert 0.45 < score < 0.55
    
    def test_f1_case_insensitive(self):
        """Test that F1 is case-insensitive."""
        assert f1_score("Hello World", "hello world") == 1.0
    
    def test_f1_empty_strings(self):
        """Test that two empty strings have F1=1.0."""
        assert f1_score("", "") == 1.0
    
    def test_f1_one_empty(self):
        """Test that one empty string with non-empty has F1=0.0."""
        assert f1_score("hello", "") == 0.0
        assert f1_score("", "hello") == 0.0


class TestRougeL:
    """Tests for ROUGE-L metric."""
    
    def test_rouge_l_identical(self):
        """Test that identical strings have ROUGE-L=1.0."""
        assert rouge_l("hello world", "hello world") == 1.0
    
    def test_rouge_l_no_overlap(self):
        """Test that completely different strings have ROUGE-L=0.0."""
        assert rouge_l("abc", "xyz") == 0.0
    
    def test_rouge_l_partial_overlap(self):
        """Test that partial overlap has 0 < ROUGE-L < 1."""
        score = rouge_l("hello there", "hello world")
        assert 0 < score < 1
    
    def test_rouge_l_case_insensitive(self):
        """Test that ROUGE-L is case-insensitive."""
        assert rouge_l("Hello World", "hello world") == 1.0
    
    def test_rouge_l_empty_strings(self):
        """Test that two empty strings have ROUGE-L=1.0."""
        assert rouge_l("", "") == 1.0
    
    def test_rouge_l_one_empty(self):
        """Test that one empty string with non-empty has ROUGE-L=0.0."""
        assert rouge_l("hello", "") == 0.0


class TestComputeMetrics:
    """Tests for compute_metrics function."""
    
    def test_compute_metrics_identical(self):
        """Test metrics for identical strings."""
        em, f1, rouge = compute_metrics("hello world", "hello world")
        assert em == 1
        assert f1 == 1.0
        assert rouge == 1.0
    
    def test_compute_metrics_different(self):
        """Test metrics for different strings."""
        em, f1, rouge = compute_metrics("hello", "goodbye")
        assert em == 0
        assert f1 == 0.0
        assert rouge == 0.0
    
    def test_compute_metrics_returns_tuple(self):
        """Test that function returns tuple of 3 elements."""
        result = compute_metrics("test", "test")
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_compute_metrics_realistic(self):
        """Test with realistic Q&A answers."""
        gold = "You should run in the rain to minimize time"
        pred = "Running minimizes the time you spend in the rain"
        
        em, f1, rouge = compute_metrics(pred, gold)
        assert em == 0
        assert 0 < f1 < 1
        assert 0 < rouge < 1


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_character(self):
        """Test with single character strings."""
        assert exact_match("a", "a") == 1
        assert f1_score("a", "a") == 1.0
    
    def test_special_characters(self):
        """Test with special characters."""
        pred = "hello@world"
        gold = "hello world"
        em = exact_match(pred, gold)
        assert em == 1
    
    def test_unicode_characters(self):
        """Test with unicode characters."""
        f1 = f1_score("café", "cafe")
        assert f1 >= 0


class TestMetricConsistency:
    """Tests for consistency between metrics."""
    
    def test_identical_all_perfect(self):
        """Test that identical strings score perfectly."""
        s = "the quick brown fox"
        assert exact_match(s, s) == 1
        assert f1_score(s, s) == 1.0
        assert rouge_l(s, s) == 1.0
    
    def test_different_all_zero(self):
        """Test that different strings score zero."""
        assert exact_match("abc", "xyz") == 0
        assert f1_score("abc", "xyz") == 0.0
        assert rouge_l("abc", "xyz") == 0.0
    
    def test_f1_greater_than_em(self):
        """Test that F1 > EM for partial matches."""
        f1 = f1_score("hello there", "hello world")
        em = exact_match("hello there", "hello world")
        assert f1 > em


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

