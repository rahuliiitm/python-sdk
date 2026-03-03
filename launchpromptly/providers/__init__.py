"""LLM provider adapters for LaunchPromptly."""

from .anthropic import (
    wrap_anthropic_client,
    extract_anthropic_message_texts,
    extract_anthropic_response_text,
    extract_anthropic_tool_calls,
    extract_anthropic_stream_chunk,
    extract_content_block_text,
)
from .gemini import (
    wrap_gemini_client,
    extract_gemini_message_texts,
    extract_gemini_response_text,
    extract_gemini_function_calls,
    extract_gemini_stream_chunk,
    extract_gemini_content_text,
)

__all__ = [
    # Anthropic
    "wrap_anthropic_client",
    "extract_anthropic_message_texts",
    "extract_anthropic_response_text",
    "extract_anthropic_tool_calls",
    "extract_anthropic_stream_chunk",
    "extract_content_block_text",
    # Gemini
    "wrap_gemini_client",
    "extract_gemini_message_texts",
    "extract_gemini_response_text",
    "extract_gemini_function_calls",
    "extract_gemini_stream_chunk",
    "extract_gemini_content_text",
]
