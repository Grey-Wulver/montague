# app/interfaces/__init__.py
"""
Interface adapters for Universal Pipeline
Connects different interfaces (Chat, Web, etc.) to Universal Pipeline
"""

from .chat_adapter import ChatAdapter, chat_adapter

__all__ = ["ChatAdapter", "chat_adapter"]
