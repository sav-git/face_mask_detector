# 🎭 Face Mask Detector

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.6%2B-green)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A practical, production-ready computer vision system for real-time face mask detection. This project demonstrates a clean implementation of a two-stage detection pipeline using OpenCV's face detector and a custom MobileNetV2 classifier, focused on inference efficiency and modular design.

## ✨ Features

### Core Capabilities
- **🎭 Real-time Detection** - Live webcam processing with interactive controls
- **📸 Image Processing** - Batch processing of images and directories
- **🧠 Modular Architecture** - Clean separation of face detection, mask classification, and visualization
- **⚡ Performance Optimized** - Configurable confidence thresholds and GPU support
- **📊 Live Statistics** - Real-time FPS, face counts, and mask percentage tracking

### Processing Modes
- **Real-time Video** - Process webcam feed with interactive keyboard controls
- **Static Images** - Process single images or entire directories
- **Batch Processing** - Automated processing with results export

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time mode)
- 4GB+ RAM (8GB+ recommended for optimal performance)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sartemv-of/Face_Mask_Detector.git
cd Face_Mask_Detector
```

2. **Create and activate virtual environment (recommended):**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download face detection models:**
```bash
python -c "from detector.utils import download_face_detector_models; download_face_detector_models()"
```

5. **Download mask classification model:**
Place the pre-trained `mask_detector.model` file in the `models/` directory.
```bash
# Example using wget:
wget -P models/ https://github.com/sartemv-of/Face_Mask_Detector/releases/download/v1.0.0/mask_detector.model
```

## 📁 Project Structure

```
face-mask-detector/
├── detector/                 # Core detection module
│   ├── __init__.py           # Package exports and version
│   ├── mask_detector.py      # Main detection class with batch processing
│   └── utils.py              # FaceDetector, Visualizer, utilities
├── face_detector/            # Pre-trained face detection models (downloaded)
│   ├── deploy.prototxt       # Network configuration
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── models/                   # Trained mask detection models (user-provided)
├── examples/                 # Sample images and configurations
├── recordings/               # Video recordings and screenshots
├── test_image.py             # Image testing script with CLI
├── test_camera.py            # Real-time camera testing
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This documentation
```

## 🛠️ Usage Examples

### Real-time Webcam Detection

```bash
# Basic usage
python test_camera.py

# With custom settings
python test_camera.py \
  --camera 0 \
  --resolution 1280x720 \
  --model models/mask_detector.model \
  --mask-threshold 0.7 \
  --record-dir recordings/
```

**Interactive Controls:**
- `SPACE` - Capture screenshot
- `R` - Start/stop video recording
- `F` - Toggle face detection
- `M` - Toggle mask detection
- `+`/`-` - Adjust confidence threshold
- `Q` - Quit application

### Image Processing

```bash
# Single image
python test_image.py -i examples/example.jpg -o results/

# Directory processing
python test_image.py -i dataset/test/ -o results/ --save-stats

# Batch processing with custom thresholds
python test_image.py \
  -i img1.jpg img2.jpg img3.jpg \
  -o results/ \
  --mask-threshold 0.6 \
  --face-threshold 0.5 \
  --verbose
```

## ⚙️ Configuration

### Environment Variables (Optional)
Create a `.env` file in project root:

```bash
# Application Settings
DEBUG=False
LOG_LEVEL=INFO
MODEL_PATH=models/mask_detector.model

# Camera Settings
CAMERA_ID=0
RESOLUTION_WIDTH=640
RESOLUTION_HEIGHT=480
FPS_LIMIT=30

# Detection Thresholds
FACE_CONFIDENCE=0.5
MASK_CONFIDENCE=0.5
```

### Programmatic Configuration

```python
from detector import FaceMaskDetector, FaceDetector, Visualizer

# Initialize with custom parameters
face_detector = FaceDetector(
    prototxt_path="face_detector/deploy.prototxt",
    model_path="face_detector/res10_300x300_ssd_iter_140000.caffemodel",
    confidence_threshold=0.5
)

mask_detector = FaceMaskDetector(
    model_path="models/mask_detector.model",
    confidence_thresh=0.7,
    input_size=(224, 224),
    gpu_mode=True  # Enable GPU acceleration
)

visualizer = Visualizer()

# Update settings dynamically
mask_detector.update_threshold(0.8)

# Get model information
info = mask_detector.get_model_info()
print(f"Model: {info['input_shape']} -> {info['output_shape']}")
```

## 📊 Performance

### Model Accuracy (Typical Results)
| Metric | Value | Description |
|--------|-------|-------------|
| **Overall Accuracy** | 95-98% | Total classification accuracy |
| **Precision (Mask)** | 94-97% | Accuracy when predicting "with_mask" |
| **Recall (Mask)** | 93-96% | Ability to find all mask instances |
| **F1 Score** | 94-97% | Balance between precision and recall |
| **Inference Time** | 15-25ms | Per-face processing time (GPU) |

### System Performance
| Mode | CPU (i5) | GPU (GTX 1060) |
|------|----------|----------------|
| **Single Image** | 80-150ms | 15-25ms |
| **Real-time** | 8-15 FPS | 25-40 FPS |

### Resource Utilization
- **Model Size**: ~15 MB (MobileNetV2 + custom head)
- **Memory Usage**: ~400-600 MB (with TensorFlow and OpenCV)
- **CPU Utilization**: 40-70% in real-time mode
- **GPU Utilization**: 20-40% with CUDA acceleration

## 🔧 Technical Details

### Architecture Overview
```
┌─────────────────────────────────────────────┐
│              Input Sources                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────────┐  │
│  │ Webcam  │ │ Images  │ │ Video Files │  │
│  └─────────┘ └─────────┘ └─────────────┘  │
└───────────────────┬────────────────────────┘
                    │
           ┌────────▼────────┐
           │  Face Detection │
           │   (OpenCV DNN)  │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │ Mask Classifier │
           │  (MobileNetV2)  │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │  Visualization  │
           │   & Results     │
           └─────────────────┘
```

### Model Architecture
Based on MobileNetV2 with custom classification head:
```python
Input(224, 224, 3)
    ↓
MobileNetV2 (pretrained, frozen)
    ↓
Global Average Pooling
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(128) → BatchNorm → Dropout(0.3)
    ↓
Dense(2, softmax)  # with_mask / without_mask
```

## 🐛 Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| **Camera not opening** | Wrong camera ID or permissions | Try `--camera 1`, on Linux: `sudo usermod -a -G video $USER` |
| **Model not found** | Mask detector model missing | Download and place `mask_detector.model` in `models/` directory |
| **Low accuracy** | Incorrect preprocessing or model | Verify input image normalization (0-1 range) and model compatibility |
| **Slow performance** | Running on CPU instead of GPU | Install GPU-enabled TensorFlow, verify CUDA drivers |
| **Memory errors** | High resolution or batch size | Reduce resolution, close other applications |

### Debug Mode
```bash
# Enable verbose output
python test_camera.py -v

# Log to file
python test_camera.py 2> debug.log

# Profile performance
python -m cProfile -o profile.stats test_camera.py
```

## 🔄 Extending Functionality

### Adding New Classes
1. **Extend dataset structure:**
```bash
mkdir dataset/improper_mask
# Add images of incorrectly worn masks
```

2. **Retrain model with modified architecture:**
```python
# In training script
num_classes = 3  # Update from 2 to 3
```

### Custom Model Architecture
Replace MobileNetV2 in your training script:

```python
from tensorflow.keras.applications import EfficientNetB0, ResNet50

# Example with EfficientNet
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=input_shape)
)
```

### Integration Examples
```python
import cv2
from detector import FaceMaskDetector, FaceDetector

class SecurityMonitoringSystem:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.mask_detector = FaceMaskDetector("models/mask_detector.model")
        self.violations = []
    
    def process_camera_feed(self, frame):
        faces = self.face_detector.detect(frame)
        results = self.mask_detector.detect_multiple_faces(frame, faces)
        
        for result in results:
            if result['label'] == 'without_mask':
                self.log_violation(result, frame)
        
        return results
```

## 📈 Roadmap

### Planned Features
- [ ] Multi-class detection (proper/improper/no mask)
- [ ] Edge deployment (Raspberry Pi, Jetson Nano optimization)
- [ ] Mobile application integration
- [ ] Cloud deployment templates
- [ ] Advanced analytics dashboard

### Version History
- **v1.0.0** (Current): Initial release with core detection functionality
- **v1.1.0** (Planned): Performance optimizations and bug fixes
- **v2.0.0** (Planned): Multi-class detection and edge deployment

## 🤝 Contributing

### Development Setup
```bash
# Fork and clone
git clone https://github.com/sartemv-of/Face_Mask_Detector.git
cd Face_Mask_Detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt  # Create this file with pytest, black, etc.

# Run tests
pytest tests/
```

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards
- Follow PEP 8 for Python code
- Add type hints for function signatures
- Include docstrings for all public functions
- Update documentation when changing APIs
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### ⭐️ If you find this project useful, please give it a star on GitHub!

Stay safe and wear your mask properly! 😷

*Made with ❤️ for the developer community*

</div>

---