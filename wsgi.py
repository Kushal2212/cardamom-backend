import sys
import os

# Add backend folder to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, backend_path)

os.chdir(backend_path)

from app import create_app

app = create_app()