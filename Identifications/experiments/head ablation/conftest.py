"""Root conftest.py — ensures the project root is on sys.path for all tests.

pytest automatically loads this file before importing any test modules,
so tests can do `from config import AblationConfig` etc. without fragile
sys.path hacks.
"""

import sys
import os

# Add the project root (directory containing this file) to sys.path
sys.path.insert(0, os.path.dirname(__file__))
