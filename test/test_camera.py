#!/usr/bin/env python3
import argparse
import cv2
import time
import sys
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict

from detector import FaceMaskDetector, FaceDetector, Visualizer
from detector.utils import download_face_detector_models

BASE_DIR = Path(__file__).parent.absolute()

class CameraDetector:
    def __init__(self, model_path: str, face_detector_config: Dict = None):
        print("[INFO] Инициализация CameraDetector...")
        
        print(f"[INFO] Загрузка модели детектора масок: {model_path}")
        self.mask_detector = FaceMaskDetector(model_path)
        
        if face_detector_config is None:
            face_detector_config = {
                'prototxt_path': "face_detector/deploy.prototxt",
                'model_path': "face_detector/res10_300x300_ssd_iter_140000.caffemodel",
                'confidence_threshold': 0.5
            }
        
        if not os.path.exists(face_detector_config['prototxt_path']):
            print("[WARNING] Модели детектора лиц не найдены. Скачивание...")
            download_face_detector_models()
        
        self.face_detector = FaceDetector(**face_detector_config)
        self.visualizer = Visualizer()
        
        self.camera = None
        self.is_recording = False
        self.video_writer = None
        self.output_dir = "recordings"
        
        self.stats = {
            'total_frames': 0,
            'total_faces': 0,
            'with_mask': 0,
            'without_mask': 0,
            'fps_history': [],
            'start_time': time.time()
        }
        
        self.settings = {
            'show_fps': True,
            'show_stats': True,
            'show_confidence': True,
            'mirror_camera': True,
            'face_detection_enabled': True,
            'mask_detection_enabled': True,
            'resolution': (640, 480),
            'face_confidence': 0.5,
            'mask_confidence': 0.5
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def initialize_camera(self, camera_id: int = 0):
        print(f"[INFO] Инициализация камеры {camera_id}...")
        
        self.camera = cv2.VideoCapture(camera_id)
        
        if not self.camera.isOpened():
            print(f"[ERROR] Не удалось открыть камеру {camera_id}")
            return False
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings['resolution'][0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings['resolution'][1])
        
        time.sleep(2.0)
        
        ret, frame = self.camera.read()
        if not ret:
            print("[ERROR] Не удалось получить кадр с камеры")
            return False
        
        print(f"[INFO] Камера инициализирована. Разрешение: {frame.shape[1]}x{frame.shape[0]}")
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        start_time = time.time()
        
        if self.settings['mirror_camera']:
            frame = cv2.flip(frame, 1)
        
        detections = []
        if self.settings['face_detection_enabled']:
            face_locations = self.face_detector.detect(frame)
            
            if face_locations and self.settings['mask_detection_enabled']:
                detections = self.mask_detector.detect_multiple_faces(frame, face_locations)
                
                self.stats['total_faces'] += len(detections)
                mask_count = sum(1 for d in detections if d['label'] == 'with_mask')
                self.stats['with_mask'] += mask_count
                self.stats['without_mask'] += len(detections) - mask_count
        
        if detections:
            output_frame = self.visualizer.draw_multiple_detections(
                frame, detections, 
                show_stats=self.settings['show_stats'],
                show_confidence=self.settings['show_confidence']
            )
        else:
            output_frame = frame.copy()
        
        if self.settings['show_fps']:
            self._add_fps_display(output_frame, start_time)
        
        self._add_settings_display(output_frame)
        
        self.stats['total_frames'] += 1
        
        return output_frame
    
    def _add_fps_display(self, frame: np.ndarray, start_time: float):
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        self.stats['fps_history'].append(fps)
        if len(self.stats['fps_history']) > 30:
            self.stats['fps_history'].pop(0)
        
        avg_fps = np.mean(self.stats['fps_history']) if self.stats['fps_history'] else 0
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _add_settings_display(self, frame: np.ndarray):
        settings_text = [
            f"Face Det: {'ON' if self.settings['face_detection_enabled'] else 'OFF'}",
            f"Mask Det: {'ON' if self.settings['mask_detection_enabled'] else 'OFF'}",
            f"Thresh: {self.settings['mask_confidence']:.2f}"
        ]
        
        y_offset = 60
        for i, text in enumerate(settings_text):
            y = y_offset + i * 25
            cv2.putText(frame, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def start_recording(self):
        if self.is_recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.output_dir, f"recording_{timestamp}.avi")
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20.0
        
        self.video_writer = cv2.VideoWriter(
            video_path, fourcc, fps, self.settings['resolution']
        )
        
        self.is_recording = True
        print(f"[INFO] Начата запись: {video_path}")
    
    def stop_recording(self):
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print("[INFO] Запись остановлена")
    
    def capture_screenshot(self, frame: np.ndarray):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(self.output_dir, f"screenshot_{timestamp}.jpg")
        
        cv2.imwrite(screenshot_path, frame)
        print(f"[INFO] Скриншот сохранен: {screenshot_path}")
    
    def print_statistics(self):
        elapsed_time = time.time() - self.stats['start_time']
        
        print("\n" + "="*50)
        print("СТАТИСТИКА РАБОТЫ")
        print("="*50)
        print(f"Время работы: {elapsed_time:.1f} секунд")
        print(f"Обработано кадров: {self.stats['total_frames']}")
        
        if self.stats['total_frames'] > 0:
            avg_fps = self.stats['total_frames'] / elapsed_time
            print(f"Средний FPS: {avg_fps:.1f}")
        
        print(f"Обнаружено лиц: {self.stats['total_faces']}")
        
        if self.stats['total_faces'] > 0:
            mask_percentage = (self.stats['with_mask'] / self.stats['total_faces'] * 100)
            print(f"  • С маской: {self.stats['with_mask']} ({mask_percentage:.1f}%)")
            print(f"  • Без маски: {self.stats['without_mask']} ({100 - mask_percentage:.1f}%)")
        
        print("="*50)
    
    def cleanup(self):
        if self.camera is not None:
            self.camera.release()
        
        self.stop_recording()
        cv2.destroyAllWindows()
        
        print("[INFO] Ресурсы освобождены")

def main():
    parser = argparse.ArgumentParser(
        description='Детекция масок в реальном времени с веб-камеры',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Управление:
  SPACE - сделать скриншот
  R     - начать/остановить запись
  F     - вкл/выкл детекцию лиц
  M     - вкл/выкл детекцию масок
  +     - увеличить порог уверенности
  -     - уменьшить порог уверенности
  Q     - выход
        """
    )

    parser.add_argument('-m', '--model', default='models/mask_detector.keras',
                       help='Путь к модели детектора масок')
    parser.add_argument('-c', '--camera', type=int, default=0,
                       help='ID камеры (по умолчанию: 0)')
    
    parser.add_argument('-r', '--resolution', default='640x480',
                       help='Разрешение камеры (по умолчанию: 640x480)')
    parser.add_argument('--record-dir', default='recordings',
                       help='Папка для сохранения записей')
    
    args = parser.parse_args()

    if not os.path.isabs(args.model):
        args.model = os.path.join(BASE_DIR, args.model)

    if not os.path.exists(args.model):
        print(f"[ERROR] Модель детектора масок не найдена: {args.model}")
        print("[INFO] Сначала обучите модель: python training/train.py")
        sys.exit(1)
    
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"[WARNING] Некорректное разрешение: {args.resolution}. Используется 640x480")
        resolution = (640, 480)
    
    detector = CameraDetector(args.model)
    detector.settings['resolution'] = resolution
    detector.output_dir = args.record_dir
    
    if not detector.initialize_camera(args.camera):
        sys.exit(1)
    
    print("\n[INFO] Запуск основного цикла. Нажмите 'q' для выхода.")
    print("[INFO] См. справку по управлению выше.\n")
    
    try:
        while True:
            ret, frame = detector.camera.read()
            if not ret:
                print("[ERROR] Не удалось получить кадр с камеры")
                break
            
            processed_frame = detector.process_frame(frame)
            
            if detector.is_recording and detector.video_writer is not None:
                detector.video_writer.write(processed_frame)
            
            cv2.imshow("Face Mask Detector", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                detector.capture_screenshot(processed_frame)
            elif key == ord('r'): 
                if detector.is_recording:
                    detector.stop_recording()
                else:
                    detector.start_recording()
            elif key == ord('f'):
                detector.settings['face_detection_enabled'] = not detector.settings['face_detection_enabled']
                print(f"[INFO] Детекция лиц: {'ВКЛ' if detector.settings['face_detection_enabled'] else 'ВЫКЛ'}")
            elif key == ord('m'):  
                detector.settings['mask_detection_enabled'] = not detector.settings['mask_detection_enabled']
                print(f"[INFO] Детекция масок: {'ВКЛ' if detector.settings['mask_detection_enabled'] else 'ВЫКЛ'}")
            elif key == ord('+'):
                new_threshold = min(detector.settings['mask_confidence'] + 0.05, 1.0)
                detector.mask_detector.update_threshold(new_threshold)
                detector.settings['mask_confidence'] = new_threshold
                print(f"[INFO] Порог уверенности: {new_threshold:.2f}")
            elif key == ord('-'):
                new_threshold = max(detector.settings['mask_confidence'] - 0.05, 0.0)
                detector.mask_detector.update_threshold(new_threshold)
                detector.settings['mask_confidence'] = new_threshold
                print(f"[INFO] Порог уверенности: {new_threshold:.2f}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Прервано пользователем")
    
    finally:
        detector.cleanup()
        detector.print_statistics()


if __name__ == "__main__":
    main()