"""Shared utility classes used across kaiwacoach modules."""

from collections import OrderedDict


class BoundedDict(OrderedDict):
    """OrderedDict that evicts the oldest entry once *maxsize* is reached."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key, value) -> None:  # type: ignore[override]
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self._maxsize:
            self.popitem(last=False)
