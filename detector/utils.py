import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import os
import urllib.request


class FaceDetector:
    
    def __init__(self, prototxt_path: str, model_path: str, confidence_threshold: float = 0.5):
        self.prototxt_path = prototxt_path
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.input_size = (300, 300)
        self.mean_values = (104.0, 177.0, 123.0)
        
        self._load_model()
    
    def _load_model(self):
        if not os.path.exists(self.prototxt_path):
            raise FileNotFoundError(f"Prototxt файл не найден: {self.prototxt_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
        
        print(f"[INFO] Загрузка детектора лиц...")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
        
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[INFO] Используется CUDA для ускорения")
        except:
            print("[INFO] Используется CPU")
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.net is None:
            self._load_model()
        
        (h, w) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, self.input_size),
            1.0,
            self.input_size,
            self.mean_values
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        face_locations = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < self.confidence_threshold:
                continue
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            width = endX - startX
            height = endY - startY
            
            face_locations.append((startX, startY, width, height))
        
        return face_locations
    
    def detect_with_landmarks(self, frame: np.ndarray):
        face_locations = self.detect(frame)
        results = []
        
        for (x, y, w, h) in face_locations:
            face_info = {
                'bbox': (x, y, w, h),
                'landmarks': self._estimate_landmarks(frame, x, y, w, h),
                'center': (x + w // 2, y + h // 2),
                'area': w * h
            }
            results.append(face_info)
        
        return results
    
    def _estimate_landmarks(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> Dict:
        return {
            'left_eye': (x + w // 3, y + h // 3),
            'right_eye': (x + 2 * w // 3, y + h // 3),
            'nose': (x + w // 2, y + h // 2),
            'mouth_left': (x + w // 3, y + 2 * h // 3),
            'mouth_right': (x + 2 * w // 3, y + 2 * h // 3)
        }


class Visualizer:    
    def __init__(self):
        self.color_map = {
            'with_mask': (0, 255, 0),
            'without_mask': (0, 0, 255),
            'unknown': (255, 255, 0),
            'error': (255, 0, 255)
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 2
        self.box_thickness = 2
    
    def draw_detection(self, 
                      frame: np.ndarray, 
                      detection: Dict,
                      show_confidence: bool = True,
                      show_landmarks: bool = False) -> np.ndarray:

        output_frame = frame.copy()
        
        bbox = detection.get('bbox', (0, 0, 0, 0))
        label = detection.get('label', 'unknown')
        confidence = detection.get('confidence', 0.0)
        
        (x, y, w, h) = bbox
        
        color = self.color_map.get(label, (255, 255, 255))
        
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, self.box_thickness)
        
        if show_confidence:
            text = f"{label}: {confidence:.2f}"
        else:
            text = label
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )
        
        cv2.rectangle(
            output_frame,
            (x, y - text_height - 10),
            (x + text_width, y),
            color,
            -1
        )
        
        cv2.putText(
            output_frame,
            text,
            (x, y - 5),
            self.font,
            self.font_scale,
            (0, 0, 0),
            self.font_thickness
        )
        
        if show_landmarks and 'landmarks' in detection:
            landmarks = detection['landmarks']
            for point_name, (px, py) in landmarks.items():
                cv2.circle(output_frame, (px, py), 3, (0, 255, 255), -1)
        
        return output_frame
    
    def draw_multiple_detections(self,
                                frame: np.ndarray,
                                detections: List[Dict],
                                show_stats: bool = True,
                                show_confidence: bool = True) -> np.ndarray:

        output_frame = frame.copy()
        
        for detection in detections:
            output_frame = self.draw_detection(output_frame, detection, show_confidence)
        
        if show_stats and detections:
            stats = self._calculate_stats(detections)
            output_frame = self._draw_stats(output_frame, stats)
        
        return output_frame
    
    def _calculate_stats(self, detections: List[Dict]) -> Dict:
        total = len(detections)
        with_mask = sum(1 for d in detections if d.get('label') == 'with_mask')
        without_mask = total - with_mask
        
        return {
            'total': total,
            'with_mask': with_mask,
            'without_mask': without_mask,
            'mask_percentage': (with_mask / total * 100) if total > 0 else 0
        }
    
    def _draw_stats(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        y_offset = 30
        line_height = 25
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        texts = [
            f"Всего лиц: {stats['total']}",
            f"С маской: {stats['with_mask']}",
            f"Без маски: {stats['without_mask']}",
            f"Маски: {stats['mask_percentage']:.1f}%"
        ]
        
        for i, text in enumerate(texts):
            y = y_offset + i * line_height
            cv2.putText(
                frame, text, (20, y),
                self.font, 0.6, (255, 255, 255), 1
            )
        
        return frame
    
    def create_legend_image(self, width: int = 300, height: int = 200) -> np.ndarray:
        legend = np.zeros((height, width, 3), dtype=np.uint8)
        legend.fill(255)
        
        y_offset = 30
        line_height = 40
        
        for i, (label, color) in enumerate(self.color_map.items()):
            y = y_offset + i * line_height
            
            cv2.rectangle(legend, (20, y - 20), (50, y), color, -1)
            cv2.rectangle(legend, (20, y - 20), (50, y), (0, 0, 0), 2)
            
            cv2.putText(
                legend, label, (70, y - 5),
                self.font, 0.6, (0, 0, 0), 1
            )
        
        cv2.putText(
            legend, "ЛЕГЕНДА", (width // 2 - 50, 20),
            self.font, 0.7, (0, 0, 0), 2
        )
        
        return legend


def download_face_detector_models(output_dir: str = "face_detector"):
    os.makedirs(output_dir, exist_ok=True)
    
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    prototxt_path = os.path.join(output_dir, "deploy.prototxt")
    model_path = os.path.join(output_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    if not os.path.exists(prototxt_path):
        print(f"[INFO] Скачивание prototxt файла...")
        try:
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
            print(f"[INFO] Prototxt сохранен: {prototxt_path}")
        except Exception as e:
            print(f"[ERROR] Не удалось скачать prototxt: {e}")
    
    if not os.path.exists(model_path):
        print(f"[INFO] Скачивание модели детектора лиц...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print(f"[INFO] Модель сохранена: {model_path}")
        except Exception as e:
            print(f"[ERROR] Не удалось скачать модель: {e}")


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    resized_image = cv2.resize(rgb_image, target_size)
    
    normalized_image = resized_image.astype(np.float32) / 255.0
    
    return normalized_image


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]
    
    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

def load_face_detector(prototxt_path: str = None, model_path: str = None) -> FaceDetector:
    if prototxt_path is None:
        prototxt_path = "face_detector/deploy.prototxt"
    
    if model_path is None:
        model_path = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    
    return FaceDetector(prototxt_path, model_path)


def create_visualizer() -> Visualizer:
    return Visualizer()