from enum import Enum
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import streamlit as st
from natsort import natsorted
from pandas import Series

from encord_active.app.common.utils import get_geometries, load_json, load_or_fill_image
