"""
Model Wrappers and Interfaces
"""

from .api_models import APIModelWrapper

# Optional local models import (requires fasttext, transformers, etc.)
try:
    from .local_models import LocalModelWrapper
    __all__ = ["APIModelWrapper", "LocalModelWrapper"]
except ImportError:
    __all__ = ["APIModelWrapper"]