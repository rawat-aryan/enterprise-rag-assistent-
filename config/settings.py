import os
from pathlib import Path
from dotenv import load_dotenv

# Go up to project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from project root
load_dotenv(BASE_DIR / ".env")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# print("OPENAI KEY LOADED:", OPENAI_API_KEY is not None)
