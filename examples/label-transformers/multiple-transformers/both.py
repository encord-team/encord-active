import sys
from pathlib import Path

p = Path(__file__).absolute().parent
sys.path.append(p.as_posix())

from bbox_transformer import BBoxTransformer
from polygon_transformer import PolyTransformer
