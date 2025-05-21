"""
Create an __init__.py file for services package.
"""

from src.services.api import NERService, create_app

__all__ = ['NERService', 'create_app']
