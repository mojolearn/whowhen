import pytest
from sklearn.ensemble import HistGradientBoostingClassifier


@pytest.fixture
def fast_scorer():
    """A small CPU scorer so the invariant tests run in seconds."""
    return HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1)
