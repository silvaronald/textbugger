"""
TextBugger - Adversarial Text Attack Library

This package contains implementations of TextBugger attacks on text classification models.
"""

__version__ = "1.0.0"
__author__ = "TextBugger Research Team"

from .attacks import BlackBoxTextBugger
from .models import APIModelWrapper

# Handle optional imports
try:
    from .models import LocalModelWrapper
    __all__ = ["BlackBoxTextBugger", "LocalModelWrapper", "APIModelWrapper"]
except ImportError:
    __all__ = ["BlackBoxTextBugger", "APIModelWrapper"]