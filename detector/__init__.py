"""
Face Mask Detection Package
Core modules for face detection and mask classification.
"""

from .mask_detector import FaceMaskDetector
from .utils import (
    FaceDetector,
    Visualizer,
    calculate_iou
)

__version__ = "2.2.3"
__author__ = "Face Mask Detection Team"

__all__ = [
    'FaceMaskDetector',
    'FaceDetector',
    'Visualizer',
    'calculate_iou'
]