"""
GestiCalc - Complete Flask Application
Hand Gesture Calculator with Live Camera Recognition
"""

from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from collections import defaultdict
import traceback
import signal
import sys

# Import calculator modules
sys.path.append('src')
from hand_calculator.hands import HandTracker
from hand_calculator.evaluator import ExpressionEvaluator


# ==================== FLASK APP SETUP ====================

app = Flask(__name__)
app.secret_key = 'gesticalc_secret_key_2025'


# ✅ StableGestureRecognizer Class
class StableGestureRecognizer:
    def __init__(self):
        self.last_time = 0
        self.cooldown = 0.8
        self.last_gesture = None

        # Stability tracking
        self.gesture_buffer = []
        self.stability_frames = 3
        self.max_buffer_size = 8

    def recognize_gesture(self, landmarks_list):
        if not landmarks_list:
            return None

        try:
            fingers = hand_tracker.fingers_up(landmarks_list[0])
            print(f"🔍 Checking fingers: {fingers}")

            gestures = {
                (0, 0, 0, 0, 0): "0",  # Closed fist
                (0, 1, 0, 0, 0): "1",  # Index finger only
                (0, 1, 1, 0, 0): "2",  # Index + Middle
                (0, 1, 1, 1, 0): "3",  # Index + Middle + Ring
                (0, 1, 1, 1, 1): "4",  # All fingers except thumb
                (1, 1, 1, 1, 1): "5",  # All fingers
                (1, 0, 0, 0, 0): "+",  # Thumb only
                (1, 1, 0, 0, 0): "-",  # Thumb + Index
                (0, 0, 0, 0, 1): "=",  # Pinky only
                (1, 0, 0, 0, 1): "C",  # Thumb + Pinky
            }

            pattern = tuple(fingers)
            gesture = gestures.get(pattern)

            if gesture:
                print(f"✅ Gesture detected: {pattern} = {gesture}")

            return gesture

        except Exception as e:
            print(f"⚠️ Gesture recognition error: {e}")
            return None

    def update_gesture_state(self, gesture):
        if not gesture:
            self.gesture_buffer = []
            return None

        try:
            now = time.time()

            self.gesture_buffer.append(gesture)
            if len(self.gesture_buffer) > self.max_buffer_size:
                self.gesture_buffer = self.gesture_buffer[-self.max_buffer_size:]

            recent_gestures = self.gesture_buffer[-self.stability_frames:]

            if (len(recent_gestures) >= self.stability_frames and
                    all(g == gesture for g in recent_gestures)):

                if (gesture != self.last_gesture and
                        (now - self.last_time) > self.cooldown):

                    self.last_time = now
                    self.last_gesture = gesture
                    self.gesture_buffer = []

                    print(f"🎯 STABLE GESTURE ACCEPTED: {gesture}")
                    print(f"⏰ Next gesture available in {self.cooldown} seconds")
                    return gesture

                elif gesture == self.last_gesture:
                    print(f"🔄 Same gesture ignored: {gesture}")
                else:
                    remaining_time = self.cooldown - (now - self.last_time)
                    print(f"⏳ Cooldown active: {remaining_time:.1f}s remaining")
            else:
                stable_count = len([g for g in recent_gestures if g == gesture])
                print(f"📊 Stability: {stable_count}/{self.stability_frames} - Need {self.stability_frames - stable_count} more")

        except Exception as e:
            print(f"⚠️ Gesture state error: {e}")

        return None

    def get_stability_info(self):
        recent_gestures = self.gesture_buffer[-self.stability_frames:] if self.gesture_buffer else []
        current_count = len(recent_gestures)
        return {'count': current_count, 'required': self.stability_frames}


# Initialize components
hand_tracker = HandTracker(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
gesture_recognizer = StableGestureRecognizer()
evaluator = ExpressionEvaluator()

# Global variables
camera = None
camera_active = False
processing_lock = threading.Lock()
current_state = {
    'gesture': None,
    'expression': '',
    'result': '',
    'landmarks_detected': False,
    'gesture_count': 0,
    'processing_time': 0
}


# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/current_state', methods=['GET'])
def get_current_state():
    try:
        return jsonify({
            'success': True,
            'gesture': current_state.get('gesture'),
            'expression': evaluator.get_current_expression(),
            'result': evaluator.get_last_result(),
            'landmarks_detected': current_state.get('landmarks_detected', False),
            'gesture_count': current_state.get('gesture_count', 0),
            'processing_time': current_state.get('processing_time', 0)
        })
    except Exception as e:
        print(f"❌ Current state error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'gesture': None,
            'expression': '',
            'result': ''
        })


@app.route('/api/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        if not data or 'expression' not in data:
            return jsonify({'success': False, 'error': 'No expression provided'})

        expression = data['expression'].strip()
        if not expression:
            return jsonify({'success': False, 'error': 'Empty expression'})

        print(f"🧮 Manual calculation request: {expression}")

        evaluator.current_expression = expression
        result = evaluator.evaluate_expression()

        print(f"📊 Manual calculation result: {expression} = {result}")

        return jsonify({
            'success': True,
            'result': result,
            'expression': evaluator.get_current_expression(),
            'history_count': len(evaluator.get_history())
        })

    except Exception as e:
        print(f"❌ Manual calculation error: {e}")
        return jsonify({'success': False, 'error': f'Calculation failed: {str(e)}'})


@app.route('/api/clear', methods=['POST'])
def clear():
    try:
        evaluator.clear_expression()
        current_state['gesture'] = None
        print("🗑️ Calculator cleared")
        return jsonify({'success': True, 'message': 'Calculator cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    try:
        evaluator.clear_history()
        print("🗑️ History cleared")
        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        history = evaluator.get_history()

        enhanced_history = []
        for entry in history:
            enhanced_entry = entry.copy()
            enhanced_entry['success'] = not str(entry['result']).startswith('Error')
            enhanced_entry['expression_length'] = len(entry['expression'])
            enhanced_history.append(enhanced_entry)

        return jsonify({
            'success': True,
            'history': enhanced_history,
            'total_count': len(enhanced_history)
        })
    except Exception as e:
        print(f"❌ History retrieval error: {e}")
        return jsonify({'success': False, 'error': str(e)})
        


@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    global camera, camera_active

    try:
        if camera_active:
            return jsonify({'success': False, 'error': 'Camera already active'})

        print("🎥 Initializing camera...")

        if camera:
            camera.release()

        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                camera = cv2.VideoCapture(0, backend)
                if camera.isOpened():
                    print(f"✅ Camera opened with backend: {backend}")
                    break
                camera.release()
            except:
                continue

        if not camera or not camera.isOpened():
            return jsonify({'success': False, 'error': 'Cannot access camera. Check permissions.'})

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 25)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("🔧 Warming up camera...")
        for i in range(5):
            ret, _ = camera.read()
            if ret:
                break

        camera_active = True
        current_state['gesture_count'] = 0

        print("✅ Camera started successfully!")
        return jsonify({'success': True, 'message': 'Camera started successfully'})

    except Exception as e:
        print(f"❌ Camera start error: {e}")
        if camera:
            camera.release()
            camera = None
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    global camera, camera_active

    try:
        camera_active = False

        if camera:
            camera.release()
            camera = None

        current_state.update({
            'gesture': None,
            'landmarks_detected': False,
            'processing_time': 0
        })

        print("🛑 Camera stopped successfully")
        return jsonify({'success': True, 'message': 'Camera stopped successfully'})

    except Exception as e:
        print(f"❌ Camera stop error: {e}")
        return jsonify({'success': False, 'error': str(e)})


def generate_frames():
    global camera, camera_active, current_state

    print("🎥 Starting video with state updates...")
    frame_count = 0

    while camera_active and camera is not None:
        try:
            ret, frame = camera.read()
            if not ret:
                print("❌ Failed to read frame")
                time.sleep(0.1)
                continue

            frame_count += 1
            frame = cv2.flip(frame, 1)

            if frame_count % 2 == 0:
                try:
                    landmarks = hand_tracker.process_frame(frame)

                    if landmarks:
                        for i, hand_landmarks in enumerate(landmarks):
                            hand_tracker.draw_landmarks(frame, hand_landmarks, i)

                        current_gesture = gesture_recognizer.recognize_gesture(landmarks)

                        current_state['gesture'] = current_gesture
                        current_state['landmarks_detected'] = True

                        if current_gesture:
                            stability_info = gesture_recognizer.get_stability_info()
                            progress_text = f"Stability: {stability_info['count']}/{stability_info['required']}"

                            cv2.putText(frame, f"Gesture: {current_gesture}", (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(frame, progress_text, (10, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                            progress_width = int((stability_info['count'] / stability_info['required']) * 300)
                            cv2.rectangle(frame, (10, 100), (310, 120), (0, 0, 0), -1)
                            cv2.rectangle(frame, (10, 100), (10 + progress_width, 120), (0, 255, 0), -1)
                            cv2.rectangle(frame, (10, 100), (310, 120), (255, 255, 255), 2)

                        else:
                            cv2.putText(frame, "Hold gesture steady...", (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        token = gesture_recognizer.update_gesture_state(current_gesture)
                        if token:
                            print(f"🚀 Processing token: {token}")
                            result = process_token(token)
                            current_state['gesture_count'] += 1

                            current_state['expression'] = evaluator.get_current_expression()
                            current_state['result'] = evaluator.get_last_result()
                            print(f"📊 UPDATE - Expression: {current_state['expression']}, Result: {current_state['result']}")

                            cv2.putText(frame, f"Added: {token}", (10, 110),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    else:
                        current_state['landmarks_detected'] = False
                        current_state['gesture'] = None
                        cv2.putText(frame, "Show your hand to camera...", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                except Exception as e:
                    print(f"⚠️ Gesture processing error: {e}")
                    cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.025)

        except Exception as e:
            print(f"❌ Frame error: {e}")
            break

    print("🛑 Video stream ended")


@app.route('/video_feed')
def video_feed():
    try:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"❌ Video feed error: {e}")
        return Response("Video feed error", status=500)


def process_token(token):
    """Process gesture token."""
    try:
        if token == "=":
            result = evaluator.evaluate_expression()
            print(f"🟰 Equals: {result}")
            return result

        elif token == "C":
            evaluator.clear_expression()
            print("🗑️ Clear")
            return "Cleared"

        elif token == "⌫":
            evaluator.backspace()
            print("⌫ Backspace")
            return "Deleted"

        else:
            evaluator.add_token(token)
            print(f"➕ Added: {token}")
            return f"Added: {token}"

    except Exception as e:
        print(f"❌ Token processing error {token}: {e}")
        return f"Error: {str(e)}"


# Graceful shutdown handling
def signal_handler(sig, frame):
    global camera, camera_active
    print("\n🛑 Shutting down gracefully...")
    camera_active = False
    if camera:
        camera.release()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


if __name__ == '__main__':
    print("🚀 Starting GestiCalc with Waitress Server...")
    print("📝 Features: Live camera gestures, manual calculator")
    print("🌐 Access at: http://localhost:5000")

    from waitress import serve

    try:
        serve(app, host='127.0.0.1', port=5000, threads=4)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        try:
            if 'camera' in globals() and camera:
                camera_active = False
                camera.release()
        except:
            pass
        print("🧹 Cleanup completed")
