"""
TextBugger Attack Implementations
"""

from .blackbox import BlackBoxTextBugger

# Optional whitebox import (requires sklearn)
try:
    from .whitebox import AdversarialAttack
    __all__ = ["BlackBoxTextBugger", "AdversarialAttack"]
except ImportError:
    __all__ = ["BlackBoxTextBugger"]