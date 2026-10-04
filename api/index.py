# Vercel serverless entry point.
# Vercel's Python runtime expects the ASGI/WSGI app object to be importable
# from a file inside the `api/` directory.  We simply re-export the FastAPI
# `app` object that lives in backend/app.py so no application code is
# duplicated or modified here.

import sys
import os
from pathlib import Path

# Make the repo root importable so `import backend.app` resolves correctly
# regardless of Vercel's working directory at function invocation time.
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.app import app  # noqa: F401  – re-exported for Vercel
