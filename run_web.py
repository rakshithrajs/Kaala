"""Web UI entry point for Kaala."""

import uvicorn
from config.config import get_settings


def main():
    settings = get_settings()
    uvicorn.run(
        "web.app:app",
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()