# conftest.py
import sys
import os

# Insert the absolute path to ./src at the front of sys.path
ROOT = os.path.dirname(__file__)
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, os.path.abspath(SRC))
