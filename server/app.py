import uvicorn

from environment import app


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)
