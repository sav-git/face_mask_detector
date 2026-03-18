import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import List, Dict, Tuple, Optional
import time


class FaceMaskDetector:

    def __init__(
        self, 
        model_path: str,
        confidence_thresh: float = 0.5,
        input_size: Tuple[int, int] = (224, 224),
        gpu_mode: bool = False
    ):
        self.confidence_thresh = confidence_thresh
        self.input_size = input_size
        
        if gpu_mode:
            self._setup_gpu()
        
        self.model = self._load_model(model_path)
        
        self._cache = {}
        
        print(f"[INFO] FaceMaskDetector инициализирован")
        print(f"  Model: {model_path}")
        print(f"  Input size: {input_size}")
        print(f"  Confidence threshold: {confidence_thresh}")
    
    def _setup_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"[INFO] Используется GPU: {len(gpus)} устройств")
            except RuntimeError as e:
                print(f"[WARNING] Не удалось настроить GPU: {e}")
    
    def _load_model(self, model_path: str):
        try:
            print(f"[INFO] Загрузка модели из {model_path}")
            
            import os
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Файл модели не найден: {model_path}")
            
            model = load_model(model_path, compile=False)
            
            test_input = np.random.randn(1, *self.input_size, 3).astype(np.float32)
            _ = model.predict(test_input, verbose=0)
            
            print(f"[INFO] Модель успешно загружена")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
            
            return model
            
        except Exception as e:
            raise ValueError(f"Ошибка при загрузке модели: {e}")
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, self.input_size)
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def detect_single_face(self, face_image: np.ndarray) -> Dict:
        if face_image is None or face_image.size == 0:
            return {
                'label': 'unknown',
                'confidence': 0.0,
                'mask_probability': 0.0,
                'no_mask_probability': 0.0,
                'error': 'Empty image'
            }
        
        try:
            processed_face = self.preprocess_face(face_image)
            
            start_time = time.time()
            predictions = self.model.predict(processed_face, verbose=0)[0]
            inference_time = time.time() - start_time
            
            no_mask_prob = float(predictions[0])
            mask_prob = float(predictions[1])
            
            if mask_prob >= self.confidence_thresh:
                label = 'with_mask'
                confidence = mask_prob
            else:
                label = 'without_mask'
                confidence = no_mask_prob
            
            return {
                'label': label,
                'confidence': confidence,
                'mask_probability': mask_prob,
                'no_mask_probability': no_mask_prob,
                'inference_time_ms': inference_time * 1000
            }
            
        except Exception as e:
            print(f"[ERROR] Ошибка при детекции: {e}")
            return {
                'label': 'error',
                'confidence': 0.0,
                'mask_probability': 0.0,
                'no_mask_probability': 0.0,
                'error': str(e)
            }
    
    def detect_multiple_faces(
        self, 
        frame: np.ndarray, 
        face_locations: List[Tuple[int, int, int, int]]
    ) -> List[Dict]:
        results = []
        
        for i, (x, y, w, h) in enumerate(face_locations):
            face = frame[y:y+h, x:x+w]
            
            if face.size == 0:
                continue
            
            detection_result = self.detect_single_face(face)
            
            detection_result.update({
                'bbox': (x, y, w, h),
                'face_id': i,
                'face_area': w * h
            })
            
            results.append(detection_result)
        
        return results
    
    def batch_detect(self, face_images: List[np.ndarray]) -> List[Dict]:
        if not face_images:
            return []
        
        batch_data = []
        valid_indices = []
        
        for i, face_img in enumerate(face_images):
            if face_img is not None and face_img.size > 0:
                processed = self.preprocess_face(face_img)
                batch_data.append(processed[0])
                valid_indices.append(i)
        
        if not batch_data:
            return []
        
        batch_array = np.array(batch_data)
        
        start_time = time.time()
        batch_predictions = self.model.predict(batch_array, verbose=0)
        batch_time = time.time() - start_time
        
        results = []
        for idx, pred_idx in enumerate(valid_indices):
            predictions = batch_predictions[idx]
            mask_prob = float(predictions[1])
            no_mask_prob = float(predictions[0])
            
            if mask_prob >= self.confidence_thresh:
                label = 'with_mask'
                confidence = mask_prob
            else:
                label = 'without_mask'
                confidence = no_mask_prob
            
            results.append({
                'face_id': pred_idx,
                'label': label,
                'confidence': confidence,
                'mask_probability': mask_prob,
                'no_mask_probability': no_mask_prob,
                'avg_inference_time_ms': (batch_time / len(batch_data)) * 1000
            })
        
        return results
    
    def update_threshold(self, new_threshold: float):
        if 0 <= new_threshold <= 1:
            old_threshold = self.confidence_thresh
            self.confidence_thresh = new_threshold
            print(f"[INFO] Порог уверенности обновлен: {old_threshold} -> {new_threshold}")
        else:
            print(f"[WARNING] Некорректный порог: {new_threshold}. Должен быть в диапазоне [0, 1]")
    
    def get_model_info(self) -> Dict:
        if self.model is None:
            return {}
        
        return {
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_layers': len(self.model.layers),
            'trainable_params': self.model.count_params(),
            'confidence_threshold': self.confidence_thresh
        }
    
    def warmup(self, num_iterations: int = 10):
        print("[INFO] Прогрев модели...")
        
        test_input = np.random.randn(
            num_iterations, 
            self.input_size[0], 
            self.input_size[1], 
            3
        ).astype(np.float32)
        
        times = []
        for i in range(num_iterations):
            start_time = time.time()
            _ = self.model.predict(test_input[i:i+1], verbose=0)
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times) * 1000
        print(f"[INFO] Прогрев завершен. Среднее время инференса: {avg_time:.2f} мс")

_detector_instance = None

def get_detector(
    model_path: str = "../models/mask_detector.keras",
    **kwargs
) -> FaceMaskDetector:
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = FaceMaskDetector(model_path, **kwargs)
    
    return _detector_instance