#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path


def print_header():
    header = """
╔══════════════════════════════════════════════════════════╗
║           Face Mask Detection System Setup               ║
║                  Версия 1.4.9                            ║
╚══════════════════════════════════════════════════════════╝
    """
    print(header)


def check_python_version():
    print("[1/8] Проверка версии Python...")
    
    required_version = (3, 7)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"  ❌ Требуется Python {required_version[0]}.{required_version[1]}+")
        print(f"  Текущая версия: {sys.version}")
        sys.exit(1)
    
    print(f"  ✅ Python {sys.version}")


def create_project_structure():
    print("[2/8] Создание структуры папок...")
    
    directories = [
        'dataset',
        'dataset/with_mask',
        'dataset/without_mask',
        'models',
        'logs',
        'examples',
        'face_detector',
        'webapp/static',
        'webapp/static/screenshots',
        'webapp/templates',
        'recordings'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  📁 Создана папка: {directory}")


def install_dependencies():
    print("[3/8] Установка зависимостей...")
    
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"  ❌ Файл {requirements_file} не найден")
        sys.exit(1)
    
    try:
        print("  🔄 Обновление pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        print("  📦 Установка пакетов...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        
        system = platform.system()
        if system == "Windows":
            print("  🪟 Установка дополнительных пакетов для Windows...")
            pass
        elif system == "Linux":
            print("  🐧 Установка дополнительных пакетов для Linux...")
            subprocess.check_call(["apt-get", "update"])
            subprocess.check_call(["apt-get", "install", "-y", "libgl1-mesa-glx", "libglib2.0-0"])
        
        print("  ✅ Зависимости установлены")
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Ошибка установки зависимостей: {e}")
        sys.exit(1)


def download_models():
    print("[4/8] Скачивание моделей...")
    
    try:
        from detector.utils import download_face_detector_models
        
        download_face_detector_models("face_detector")
        print("  ✅ Модели детектора лиц загружены")
        
    except Exception as e:
        print(f"  ⚠️  Не удалось скачать модели автоматически: {e}")
        print("  ℹ️  Вы можете скачать их вручную:")
        print("     - deploy.prototxt")
        print("     - res10_300x300_ssd_iter_140000.caffemodel")
        print("  📁 Поместите их в папку face_detector/")


def create_example_files():
    print("[5/8] Создание примеров файлов...")
    
    example_image = os.path.join("examples", "example.jpg")
    if not os.path.exists(example_image):
        try:
            import cv2
            import numpy as np
            
            img = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(img, "Face Mask Detector", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(example_image, img)
            print(f"  📷 Создан пример изображения: {example_image}")
        except:
            print("  ⚠️  Не удалось создать пример изображения")
    
    config_example = os.path.join("examples", "config.json")
    if not os.path.exists(config_example):
        config = {
            "camera": {
                "id": 0,
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "detection": {
                "face_confidence": 0.5,
                "mask_confidence": 0.5,
                "enable_face_detection": True,
                "enable_mask_detection": True
            },
            "display": {
                "show_fps": True,
                "show_stats": True,
                "show_confidence": True
            }
        }
        
        import json
        with open(config_example, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"  ⚙️  Создан пример конфигурации: {config_example}")


def check_gpu_support():
    print("[6/8] Проверка поддержки GPU...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"  ✅ Обнаружен GPU: {len(gpus)} устройств")
            for gpu in gpus:
                print(f"     - {gpu.name}")
            
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("  ℹ️  GPU не обнаружен, будет использоваться CPU")
            
    except ImportError:
        print("  ⚠️  TensorFlow не установлен, проверка GPU пропущена")
    except Exception as e:
        print(f"  ⚠️  Ошибка проверки GPU: {e}")


def create_launch_scripts():
    print("[7/8] Создание скриптов запуска...")
    
    scripts = {
        "run_training.bat": """
@echo off
echo Запуск обучения модели...
python training/02_train_model.py
pause
""",
        "run_training.sh": """#!/bin/bash
echo "Запуск обучения модели..."
python training/02_train_model.py
read -p "Нажмите Enter для продолжения..."
""",
        "run_webcam.bat": """
@echo off
echo Запуск детекции с веб-камеры...
python test_camera.py
pause
""",
        "run_webcam.sh": """#!/bin/bash
echo "Запуск детекции с веб-камеры..."
python test_camera.py
read -p "Нажмите Enter для продолжения..."
""",
        "run_webapp.bat": """
@echo off
echo Запуск веб-приложения...
cd webapp
python app.py
pause
""",
        "run_webapp.sh": """#!/bin/bash
echo "Запуск веб-приложения..."
cd webapp
python app.py
read -p "Нажмите Enter для продолжения..."
"""
    }
    
    for script_name, script_content in scripts.items():
        script_path = os.path.join(".", script_name)
        
        system = platform.system()
        
        if system == "Windows" and script_name.endswith(".bat"):
            with open(script_path, "w", encoding="cp866") as f:
                f.write(script_content)
            print(f"  📜 Создан скрипт: {script_name}")
        
        elif system != "Windows" and script_name.endswith(".sh"):
            with open(script_path, "w") as f:
                f.write(script_content)
            # Делаем исполняемым
            os.chmod(script_path, 0o755)
            print(f"  📜 Создан скрипт: {script_name}")


def print_final_instructions():
    print("[8/8] Финальные инструкции...")
    
    instructions = """
╔══════════════════════════════════════════════════════════╗
║              УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!               ║
╚══════════════════════════════════════════════════════════╝

🎉 Поздравляем! Face Mask Detection System успешно установлен.

📋 СЛЕДУЮЩИЕ ШАГИ:

1. 📊 ПОДГОТОВКА ДАННЫХ
   • Поместите изображения с масками в: dataset/with_mask/
   • Поместите изображения без масок в: dataset/without_mask/

2. 🧠 ОБУЧЕНИЕ МОДЕЛИ
   • Запустите обучение: python training/02_train_model.py
   • Или используйте скрипт: run_training.bat (Windows)
                             run_training.sh (Linux/Mac)

3. 🎬 ТЕСТИРОВАНИЕ
   • С веб-камеры: python test_camera.py
   • На изображении: python test_image.py -i examples/example.jpg
   • Веб-приложение: python webapp/app.py

4. 🌐 ВЕБ-ИНТЕРФЕЙС
   • Откройте браузер и перейдите по адресу: http://localhost:5000

📁 СТРУКТУРА ПРОЕКТА:
   dataset/          - данные для обучения
   models/           - обученные модели
   detector/         - модуль детекции
   training/         - скрипты обучения
   webapp/           - веб-приложение
   examples/         - примеры файлов
   recordings/       - записи с камеры

📞 ПОДДЕРЖКА:
   • Проблемы с установкой: проверьте версию Python (>=3.7)
   • Проблемы с камерой: проверьте подключение веб-камеры
   • Проблемы с моделями: запустите setup.py еще раз

🚀 Удачи в использовании Face Mask Detection System!
    """
    
    print(instructions)


def main():
    parser = argparse.ArgumentParser(description='Установка Face Mask Detection System')
    parser.add_argument('--skip-download', action='store_true',
                       help='Пропустить скачивание моделей')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Пропустить установку зависимостей')
    
    args = parser.parse_args()
    
    print_header()
    check_python_version()
    create_project_structure()
    
    if not args.skip_deps:
        install_dependencies()
    else:
        print("[3/8] Пропуск установки зависимостей...")
    
    if not args.skip_download:
        download_models()
    else:
        print("[4/8] Пропуск скачивания моделей...")
    
    create_example_files()
    check_gpu_support()
    create_launch_scripts()
    print_final_instructions()


if __name__ == "__main__":
    main()
