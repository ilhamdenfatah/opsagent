import sys
from pathlib import Path

# Add project root to Python path so 'src' is importable during tests
sys.path.insert(0, str(Path(__file__).parent))