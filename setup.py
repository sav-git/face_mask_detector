from setuptools import setup, find_packages

setup(
    name="face_mask_detector",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.8.0",
        "opencv-python>=4.6.0",
        "numpy>=1.21.0",
        "flask>=2.0.0",
        "flask-socketio>=5.0.0",
        "flask-cors>=3.0.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "imutils>=0.5.4",
        "pillow>=9.0.0",
        "python-dotenv>=0.19.0",
        "tqdm>=4.64.0",
        "joblib>=1.1.0"
    ],
    python_requires=">=3.7",
)
