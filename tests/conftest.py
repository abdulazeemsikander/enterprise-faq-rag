import sys
import os
from pathlib import Path

# Add project root (so src/ is discoverable)
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
