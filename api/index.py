"""
Vercel Serverless Function entrypoint for Emotion Finder FastHTML application.
"""
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from root modules
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import app

# Explicit top-level aliases for Vercel ASGI serverless function AST detection
application = app
handler = app
app = app

