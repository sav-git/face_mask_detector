import os
import sys
import base64
import json
import threading
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from detector import FaceMaskDetector, FaceDetector, Visualizer
from detector.utils import download_face_detector_models

app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

detector = None
face_detector = None
visualizer = None
camera = None
camera_thread = None
camera_running = False
current_frame = None
frame_lock = threading.Lock()

stats = {
    'total_faces': 0,
    'with_mask': 0,
    'without_mask': 0,
    'fps': 0,
    'is_detecting': False,
    'last_update': time.time()
}

settings = {
    'face_confidence': 0.5,
    'mask_confidence': 0.5,
    'show_fps': True,
    'show_stats': True,
    'show_confidence': True,
    'face_detection_enabled': True,
    'mask_detection_enabled': True,
    'resolution': (640, 480)
}


def initialize_models():
    global detector, face_detector, visualizer
    
    print("[INFO] Инициализация моделей...")
    
    try:
        model_path = "models/mask_detector.model"
        if not os.path.exists(model_path):
            model_path = "../models/mask_detector.model"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        detector = FaceMaskDetector(model_path, confidence_thresh=settings['mask_confidence'])
        
        prototxt_path = "face_detector/deploy.prototxt"
        caffemodel_path = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        
        if not os.path.exists(prototxt_path):
            print("[WARNING] Модели детектора лиц не найдены. Скачивание...")
            download_face_detector_models()
        
        face_detector = FaceDetector(prototxt_path, caffemodel_path, 
                                   confidence_threshold=settings['face_confidence'])
        
        visualizer = Visualizer()
        
        print("[INFO] Модели инициализированы успешно")
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации моделей: {e}")
        return False


def start_camera(camera_id=0):
    global camera, camera_running, camera_thread
    
    if camera_running:
        return
    
    print(f"[INFO] Запуск камеры {camera_id}...")
    
    camera = cv2.VideoCapture(camera_id)
    if not camera.isOpened():
        print(f"[ERROR] Не удалось открыть камеру {camera_id}")
        return False
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings['resolution'][0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['resolution'][1])
    
    time.sleep(2.0)
    
    camera_running = True
    
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    
    return True


def camera_loop():
    global camera_running, current_frame, stats
    
    fps_start_time = time.time()
    frame_count = 0
    
    while camera_running and camera is not None:
        ret, frame = camera.read()
        if not ret:
            print("[ERROR] Не удалось получить кадр с камеры")
            time.sleep(0.1)
            continue
        
        frame = cv2.flip(frame, 1)
        
        processed_frame, detections = process_frame(frame)
        
        with frame_lock:
            current_frame = processed_frame
        
        frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            stats['fps'] = frame_count
            frame_count = 0
            fps_start_time = time.time()
            
            socketio.emit('stats_update', stats)
        
        time.sleep(0.03)


def process_frame(frame):
    global stats
    
    start_time = time.time()
    
    detections = []
    if settings['face_detection_enabled'] and face_detector is not None:
        face_locations = face_detector.detect(frame)
        
        if face_locations and settings['mask_detection_enabled'] and detector is not None:
            detections = detector.detect_multiple_faces(frame, face_locations)
            
            with frame_lock:
                stats['total_faces'] += len(detections)
                mask_count = sum(1 for d in detections if d['label'] == 'with_mask')
                stats['with_mask'] += mask_count
                stats['without_mask'] += len(detections) - mask_count
                stats['last_update'] = time.time()
    
    if visualizer is not None:
        output_frame = visualizer.draw_multiple_detections(
            frame, detections,
            show_stats=settings['show_stats'],
            show_confidence=settings['show_confidence']
        )
    else:
        output_frame = frame.copy()
    
    if settings['show_fps']:
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return output_frame, detections


def generate_frames():
    while True:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.1)
                continue
            
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if not ret:
                continue
            
            frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def get_stats():
    return jsonify(stats)


@app.route('/api/settings')
def get_settings():
    return jsonify(settings)


@app.route('/api/settings/update', methods=['POST'])
def update_settings():
    global settings
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    for key, value in data.items():
        if key in settings:
            settings[key] = value
            
            if key == 'mask_confidence' and detector is not None:
                detector.update_threshold(value)
            elif key == 'face_confidence' and face_detector is not None:
                face_detector.confidence_threshold = value
    
    return jsonify({'success': True, 'settings': settings})


@app.route('/api/detect', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        processed_image, detections = process_frame(image)
        
        original_b64 = image_to_base64(image)
        processed_b64 = image_to_base64(processed_image)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'original_image': original_b64,
            'processed_image': processed_b64,
            'count': len(detections)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/camera/start', methods=['POST'])
def start_camera_api():
    data = request.get_json()
    camera_id = data.get('camera_id', 0) if data else 0
    
    if start_camera(camera_id):
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Failed to start camera'}), 500


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera_api():
    global camera_running, camera
    
    camera_running = False
    if camera is not None:
        camera.release()
        camera = None
    
    return jsonify({'success': True})


@app.route('/api/screenshot', methods=['GET'])
def take_screenshot():
    with frame_lock:
        if current_frame is None:
            return jsonify({'error': 'No frame available'}), 404
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_dir = "static/screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        
        screenshot_path = os.path.join(screenshot_dir, f"screenshot_{timestamp}.jpg")
        cv2.imwrite(screenshot_path, current_frame)
        
        image_b64 = image_to_base64(current_frame)
        
        return jsonify({
            'success': True,
            'image': image_b64,
            'path': f"/screenshots/screenshot_{timestamp}.jpg"
        })


@app.route('/screenshots/<filename>')
def serve_screenshot(filename):
    return send_from_directory('static/screenshots', filename)


@socketio.on('connect')
def handle_connect():
    """Обработчик подключения WebSocket"""
    print(f"[INFO] WebSocket client connected: {request.sid}")
    emit('connected', {'message': 'Connected to Face Mask Detector'})


@socketio.on('disconnect')
def handle_disconnect():
    print(f"[INFO] WebSocket client disconnected: {request.sid}")


@socketio.on('get_stats')
def handle_get_stats():
    emit('stats', stats)


@socketio.on('update_settings')
def handle_update_settings(data):
    global settings
    
    for key, value in data.items():
        if key in settings:
            settings[key] = value
    
    emit('settings_updated', settings)


if __name__ == '__main__':
    if not initialize_models():
        print("[ERROR] Не удалось инициализировать модели. Завершение...")
        sys.exit(1)
    
    start_camera()
    
    print("[INFO] Запуск веб-приложения...")
    print("[INFO] Откройте браузер и перейдите по адресу: http://localhost:5000")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)