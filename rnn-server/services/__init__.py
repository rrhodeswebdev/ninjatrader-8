"""Service layer for composing pure functions with the existing model."""

from .request_handler import (
    handle_historical_request,
    handle_realtime_request,
    create_response_builder,
)

__all__ = [
    "handle_historical_request",
    "handle_realtime_request",
    "create_response_builder",
]
