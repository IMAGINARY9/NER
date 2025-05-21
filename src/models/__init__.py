"""
Create an __init__.py file for models package.
"""

from src.models.ner_model import build_model

__all__ = ['build_model']
