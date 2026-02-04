"""KaiwaCoach application entrypoint."""

from __future__ import annotations

from kaiwacoach.config import load_config


def main() -> None:
    """Load and validate configuration, then start the application."""
    _ = load_config()
    # TODO: initialize UI and orchestrator.


if __name__ == "__main__":
    main()
