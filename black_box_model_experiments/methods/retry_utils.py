from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Optional, TypeVar


T = TypeVar("T")

# Hardcoded point-level retry policy.
POINT_RETRY_ATTEMPTS = 3
POINT_RETRY_DELAY_SECONDS = 1.0


@dataclass
class RetryResult(Generic[T]):
    success: bool
    value: Optional[T]
    error: Optional[BaseException]
    attempts: int


def retry_sync(
    operation: Callable[[], T],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> RetryResult[T]:
    last_error: Optional[BaseException] = None
    for attempt in range(1, POINT_RETRY_ATTEMPTS + 1):
        try:
            return RetryResult(success=True, value=operation(), error=None, attempts=attempt)
        except Exception as exc:  # pragma: no cover - covered in tests with custom ops
            last_error = exc
            if attempt < POINT_RETRY_ATTEMPTS:
                sleep_fn(POINT_RETRY_DELAY_SECONDS)
    return RetryResult(
        success=False,
        value=None,
        error=last_error,
        attempts=POINT_RETRY_ATTEMPTS,
    )


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    sleep_fn: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> RetryResult[T]:
    last_error: Optional[BaseException] = None
    for attempt in range(1, POINT_RETRY_ATTEMPTS + 1):
        try:
            return RetryResult(success=True, value=await operation(), error=None, attempts=attempt)
        except Exception as exc:  # pragma: no cover - covered in tests with custom ops
            last_error = exc
            if attempt < POINT_RETRY_ATTEMPTS:
                await sleep_fn(POINT_RETRY_DELAY_SECONDS)
    return RetryResult(
        success=False,
        value=None,
        error=last_error,
        attempts=POINT_RETRY_ATTEMPTS,
    )
