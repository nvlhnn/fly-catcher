import os, sys

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(__file__))

from app.main import app as application

