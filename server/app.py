"""
Server entry point for model-card-auditor OpenEnv environment.
Used by 'uv run server' and 'openenv serve'.
"""
import uvicorn
from model_card_auditor.server.app import app


def main():
    """Start the model-card-auditor server."""
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
