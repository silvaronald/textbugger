"""
TextBugger Attack Implementations
"""

from .blackbox import BlackBoxTextBugger
from .batch_blackbox import BatchBlackBoxTextBugger, BatchLevel

# Optional whitebox import (requires sklearn)
try:
    from .whitebox import AdversarialAttack
    __all__ = ["BlackBoxTextBugger", "BatchBlackBoxTextBugger", "BatchLevel", "AdversarialAttack"]
except ImportError:
    __all__ = ["BlackBoxTextBugger", "BatchBlackBoxTextBugger", "BatchLevel"]