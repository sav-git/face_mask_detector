#!/usr/bin/env python3
import argparse
import cv2
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import FaceMaskDetector, FaceDetector, Visualizer
from detector.utils import download_face_detector_models


class ImageTester:
    
    def __init__(self, model_path: str, face_detector_config: Dict = None):
        print("[INFO] Инициализация ImageTester...")
        
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
        
        self.stats = {
            'total_images': 0,
            'total_faces': 0,
            'with_mask': 0,
            'without_mask': 0,
            'processing_times': []
        }
    
    def process_single_image(self, image_path: str, output_dir: str = None) -> Dict:
        print(f"[INFO] Обработка изображения: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Не удалось загрузить изображение: {image_path}")
            return None
        
        start_time = time.time()
        
        face_locations = self.face_detector.detect(image)
        
        if not face_locations:
            print(f"[INFO] На изображении не обнаружено лиц")
            result = {
                'image_path': image_path,
                'faces_detected': 0,
                'detections': [],
                'processing_time': time.time() - start_time
            }
        else:
            detections = self.mask_detector.detect_multiple_faces(image, face_locations)
            
            self.stats['total_faces'] += len(detections)
            self.stats['with_mask'] += sum(1 for d in detections if d['label'] == 'with_mask')
            self.stats['without_mask'] += len(detections) - sum(1 for d in detections if d['label'] == 'with_mask')
            
            output_image = self.visualizer.draw_multiple_detections(
                image, detections, show_stats=True, show_confidence=True
            )
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
                cv2.imwrite(output_path, output_image)
                print(f"[INFO] Результат сохранен: {output_path}")
            
            result = {
                'image_path': image_path,
                'faces_detected': len(detections),
                'detections': detections,
                'processing_time': time.time() - start_time
            }
        
        self.stats['total_images'] += 1
        self.stats['processing_times'].append(time.time() - start_time)
        
        return result
    
    def process_directory(self, input_dir: str, output_dir: str = None) -> List[Dict]:
        print(f"[INFO] Обработка папки: {input_dir}")
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        image_paths = []
        for ext in valid_extensions:
            image_paths.extend(Path(input_dir).glob(f'*{ext}'))
            image_paths.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        if not image_paths:
            print(f"[WARNING] В папке не найдены изображения: {input_dir}")
            return []
        
        print(f"[INFO] Найдено {len(image_paths)} изображений")
        
        results = []
        for i, image_path in enumerate(image_paths):
            print(f"[INFO] Обработка {i+1}/{len(image_paths)}: {image_path.name}")
            
            result = self.process_single_image(str(image_path), output_dir)
            if result:
                results.append(result)
        
        return results
    
    def process_batch(self, image_paths: List[str], output_dir: str = None) -> List[Dict]:
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"[INFO] Обработка {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.process_single_image(image_path, output_dir)
            if result:
                results.append(result)
        
        return results
    
    def print_statistics(self):
        print("\n" + "="*50)
        print("СТАТИСТИКА ОБРАБОТКИ")
        print("="*50)
        
        if self.stats['total_images'] == 0:
            print("Нет обработанных изображений")
            return
        
        print(f"Обработано изображений: {self.stats['total_images']}")
        print(f"Обнаружено лиц: {self.stats['total_faces']}")
        print(f"  • С маской: {self.stats['with_mask']} ({self._percentage(self.stats['with_mask'], self.stats['total_faces'])}%)")
        print(f"  • Без маски: {self.stats['without_mask']} ({self._percentage(self.stats['without_mask'], self.stats['total_faces'])}%)")
        
        if self.stats['processing_times']:
            avg_time = np.mean(self.stats['processing_times'])
            print(f"Среднее время обработки: {avg_time:.3f} секунд")
            print(f"Общее время: {sum(self.stats['processing_times']):.3f} секунд")
        
        print("="*50)
    
    def _percentage(self, part: int, total: int) -> float:
        return (part / total * 100) if total > 0 else 0
    
    def save_statistics(self, output_path: str = "statistics.json"):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=4, ensure_ascii=False)
        
        print(f"[INFO] Статистика сохранена: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Тестирование детектора масок на изображениях',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s -i image.jpg -o results/
  %(prog)s -i folder/ -o results/ -c 0.7
  %(prog)s -i img1.jpg img2.jpg -o results/
        """
    )
    
    parser.add_argument('-i', '--input', required=True, nargs='+',
                       help='Путь к изображению или папке с изображениями')
    parser.add_argument('-o', '--output', default='output',
                       help='Папка для сохранения результатов (по умолчанию: output)')
    parser.add_argument('-m', '--model', default='models/mask_detector.model',
                       help='Путь к модели детектора масок (по умолчанию: models/mask_detector.model)')
    
    parser.add_argument('-p', '--prototxt', default='face_detector/deploy.prototxt',
                       help='Путь к prototxt файлу детектора лиц')
    parser.add_argument('-c', '--caffemodel', default='face_detector/res10_300x300_ssd_iter_140000.caffemodel',
                       help='Путь к модели детектора лиц (caffemodel)')
    parser.add_argument('-ft', '--face-threshold', type=float, default=0.5,
                       help='Порог уверенности для детекции лиц (0-1)')
    
    parser.add_argument('-mt', '--mask-threshold', type=float, default=0.5,
                       help='Порог уверенности для детекции масок (0-1)')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Подробный вывод')
    parser.add_argument('-s', '--save-stats', action='store_true',
                       help='Сохранять статистику в JSON файл')
    parser.add_argument('--no-display', action='store_true',
                       help='Не показывать изображения (только сохранять)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"[ERROR] Модель детектора масок не найдена: {args.model}")
        print("[INFO] Сначала обучите модель: python training/02_train_model.py")
        sys.exit(1)
    
    face_detector_config = {
        'prototxt_path': args.prototxt,
        'model_path': args.caffemodel,
        'confidence_threshold': args.face_threshold
    }
    
    tester = ImageTester(args.model, face_detector_config)
    
    tester.mask_detector.update_threshold(args.mask_threshold)
    
    input_paths = args.input
    
    if len(input_paths) == 1:
        input_path = input_paths[0]
        
        if os.path.isfile(input_path):
            results = [tester.process_single_image(input_path, args.output)]
        elif os.path.isdir(input_path):
            results = tester.process_directory(input_path, args.output)
        else:
            print(f"[ERROR] Путь не существует: {input_path}")
            sys.exit(1)
    else:
        results = tester.process_batch(input_paths, args.output)
    
    tester.print_statistics()
    
    if args.save_stats:
        tester.save_statistics(os.path.join(args.output, "statistics.json"))
    
    if results and not args.no_display:
        try:
            first_result = results[0]
            if first_result and first_result.get('faces_detected', 0) > 0:
                result_image_path = os.path.join(args.output, f"result_{os.path.basename(first_result['image_path'])}")
                if os.path.exists(result_image_path):
                    image = cv2.imread(result_image_path)
                    cv2.imshow("Пример результата", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        except Exception as e:
            print(f"[INFO] Не удалось отобразить результат: {e}")
    
    print("[INFO] Обработка завершена!")


if __name__ == "__main__":
    main()