from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from .receiver_lens import ReceiverLens


class ReceiverLensBank(nn.Module):
    """Explicitly select a local lens by trusted receiver identity.

    Missing identities deliberately return ``None`` so the caller executes the
    unchanged base route.  This module is not a learned router.
    """

    def __init__(self, lenses: Mapping[str, ReceiverLens]) -> None:
        super().__init__()
        if not lenses:
            raise ValueError("lens bank must contain at least one receiver")
        for receiver_id, lens in lenses.items():
            if not isinstance(receiver_id, str) or not receiver_id.strip():
                raise ValueError("receiver IDs must be non-empty strings")
            if "." in receiver_id:
                raise ValueError("receiver IDs cannot contain module separators")
            if not isinstance(lens, ReceiverLens):
                raise TypeError("lens bank values must be ReceiverLens modules")
        self.lenses = nn.ModuleDict(dict(lenses))

    def select(self, receiver_id: str) -> ReceiverLens | None:
        if not isinstance(receiver_id, str) or not receiver_id:
            raise ValueError("receiver_id must be non-empty text")
        return self.lenses[receiver_id] if receiver_id in self.lenses else None

    @property
    def receiver_ids(self) -> tuple[str, ...]:
        return tuple(self.lenses.keys())
