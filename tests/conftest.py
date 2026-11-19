import pytest
from sklearn.ensemble import HistGradientBoostingClassifier


@pytest.fixture
def fast_scorer():
    """A small CPU scorer so the invariant tests run in seconds."""
    return HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1)


PROBE = """
import numpy as np
from whowhen.scorers import fit_score, resolve
rng = np.random.default_rng(0)
X = rng.normal(size=(400, 4)).astype(np.float32)
fit_score(resolve(None), 0, 1, X[:300], (X[:300, 0] > 0).astype(int), X[300:])
"""


@pytest.fixture(scope="session")
def default_scorer_runs():
    """Skip when the default scorer cannot fit on this machine at all.

    The probe runs in a child process because a scorer built for
    hardware the machine lacks can end the interpreter (an illegal
    instruction, an unsupported GPU), which no except clause catches.
    A skip here is about the machine, not about whowhen.
    """
    import subprocess
    import sys
    p = subprocess.run([sys.executable, "-W", "ignore", "-c", PROBE],
                       capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        tail = (p.stderr.strip().splitlines() or ["no output"])[-1]
        pytest.skip(f"default scorer cannot run here (exit "
                    f"{p.returncode}): {tail[:200]}")
