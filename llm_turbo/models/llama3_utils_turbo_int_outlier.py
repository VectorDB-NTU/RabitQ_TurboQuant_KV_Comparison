import sys
from pathlib import Path

_llm_dir = str(Path(__file__).resolve().parent.parent)
if _llm_dir not in sys.path:
    sys.path.insert(0, _llm_dir)

from utils_turbo_outlier import TurboSketch as RQJLSketch
from utils_turbo_outlier import TurboKeyQuantizer as RQJLKeyQuantizer
