"""Input contract for one model- and adapter-aware request."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    model_id: str
    text: str
    token_ids: tuple[int, ...]
    share_group: str = "controlled-public-context"
    adapter_id: str | None = None
