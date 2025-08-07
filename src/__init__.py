"""
TextBugger - Adversarial Text Attack Library

This package contains implementations of TextBugger attacks on text classification models.
"""

__version__ = "1.0.0"
__author__ = "TextBugger Research Team"

from .attacks import BlackBoxTextBugger
from .models import LocalModelWrapper, APIModelWrapper

__all__ = ["BlackBoxTextBugger", "LocalModelWrapper", "APIModelWrapper"]