"""
Unit Tests for Confidence Scoring
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')

from confidence_scoring import AdvancedConfidenceScoring


def test_model_score():
    """Test model probability scoring"""
    scorer = AdvancedConfidenceScoring()

    # High confidence case
    probs_high = np.array([0.1, 0.1, 0.8])
    score = scorer._calculate_model_score(probs_high)
    assert 60 < score < 100

    # Low confidence case
    probs_low = np.array([0.33, 0.33, 0.34])
    score = scorer._calculate_model_score(probs_low)
    assert 0 < score < 50


def test_feature_quality():
    """Test feature quality scoring"""
    scorer = AdvancedConfidenceScoring()

    # Good quality features
    features_good = np.random.randn(50)
    score = scorer._calculate_feature_quality(features_good)
    assert score > 70

    # Poor quality (lots of NaN)
    features_bad = np.array([np.nan] * 25 + list(np.random.randn(25)))
    score = scorer._calculate_feature_quality(features_bad)
    assert score < 60


def test_composite_confidence():
    """Test composite confidence calculation"""
    scorer = AdvancedConfidenceScoring()

    probabilities = np.array([0.1, 0.2, 0.7])
    features = np.random.randn(50)
    market_state = {
        'volatility': 0.015,
        'volume': 5000,
        'trend_strength': 0.5
    }

    composite, components = scorer.calculate_composite_confidence(
        probabilities=probabilities,
        features=features,
        market_state=market_state
    )

    assert 0 <= composite <= 100
    assert 'model_probability' in components
    assert 'feature_quality' in components


def test_dynamic_threshold():
    """Test dynamic threshold by regime"""
    scorer = AdvancedConfidenceScoring()

    assert scorer.get_dynamic_threshold('trending') == 60.0
    assert scorer.get_dynamic_threshold('ranging') == 75.0
    assert scorer.get_dynamic_threshold('volatile') == 70.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
