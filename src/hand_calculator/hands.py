"""
Enhanced hand tracking with improved performance, accuracy, and gesture recognition.
Complete implementation with all required methods for the GestiCalc project.
"""

import cv2
import mediapipe as mp
import numpy as np
import time

class HandTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Initialize hand tracker with optimized settings.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands with optimized settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=0  # Use lighter model for better performance
        )
        
        # Hand landmark indices for finger detection
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        self.pip_ids = [3, 6, 10, 14, 18]  # PIP joints for comparison
        self.mcp_ids = [2, 5, 9, 13, 17]   # MCP joints
        
        # Performance tracking
        self.processing_times = []
        self.max_processing_samples = 30
        
        # Gesture stability tracking
        self.last_gestures = []
        self.gesture_buffer_size = 5
        
        print("🤖 HandTracker initialized successfully")
    
    def process_frame(self, frame):
        """
        Process video frame and extract hand landmarks with performance optimization.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of hand landmarks (one list per detected hand)
        """
        start_time = time.time()
        
        try:
            # Convert BGR to RGB as required by MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False  # Performance improvement
            
            # Process the frame
            results = self.hands.process(rgb_frame)
            
            # Convert back to BGR for OpenCV
            rgb_frame.flags.writeable = True
            
            landmarks_list = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks with improved precision
                    landmarks = self._extract_landmarks(hand_landmarks, frame.shape)
                    if len(landmarks) == 21:  # Ensure we have all landmarks
                        landmarks_list.append(landmarks)
            
            # Track processing performance
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            return landmarks_list
            
        except Exception as e:
            print(f"❌ Hand processing error: {e}")
            return []
    
    def _extract_landmarks(self, hand_landmarks, frame_shape):
        """
        Extract hand landmarks with pixel coordinates and validation.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            frame_shape: Shape of the video frame (height, width, channels)
            
        Returns:
            List of [x, y] pixel coordinates for each landmark
        """
        landmarks = []
        h, w = frame_shape[:2]
        
        for lm in hand_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            x = max(0, min(int(lm.x * w), w-1))  # Clamp to frame bounds
            y = max(0, min(int(lm.y * h), h-1))  # Clamp to frame bounds
            landmarks.append([x, y])
        
        return landmarks
    
    def fingers_up(self, landmarks):
        """
        MAIN METHOD: Determine which fingers are up based on landmarks.
        This is the method called by flask_app.py for gesture recognition.
        
        Args:
            landmarks: List of hand landmarks [[x, y], [x, y], ...]
            
        Returns:
            List of [thumb, index, middle, ring, pinky] where 1=up, 0=down
        """
        if len(landmarks) < 21:
            return [0, 0, 0, 0, 0]
        
        fingers = []
        
        # 👍 THUMB DETECTION (Most Important for Gestures)
        thumb_tip = landmarks[4]    # Thumb tip
        thumb_ip = landmarks[3]     # Thumb IP joint
        thumb_mcp = landmarks[2]    # Thumb MCP joint
        
        # Determine hand orientation (left vs right hand)
        wrist_x = landmarks[0][0]
        middle_mcp_x = landmarks[9][0]
        
        # Enhanced thumb detection algorithm
        if middle_mcp_x > wrist_x:  # Right hand
            # For right hand, thumb is up if tip is to the right of IP joint
            thumb_up = thumb_tip[0] > thumb_ip[0] + 15  # Added threshold for stability
        else:  # Left hand
            # For left hand, thumb is up if tip is to the left of IP joint
            thumb_up = thumb_tip[0] < thumb_ip[0] - 15  # Added threshold for stability
        
        fingers.append(1 if thumb_up else 0)
        
        # 🖐️ OTHER FINGERS DETECTION (Index, Middle, Ring, Pinky)
        finger_tips = [8, 12, 16, 20]   # Tip landmarks
        finger_pips = [6, 10, 14, 18]   # PIP joint landmarks
        finger_mcps = [5, 9, 13, 17]    # MCP joint landmarks
        
        for i, (tip_id, pip_id, mcp_id) in enumerate(zip(finger_tips, finger_pips, finger_mcps)):
            tip_y = landmarks[tip_id][1]
            pip_y = landmarks[pip_id][1]
            mcp_y = landmarks[mcp_id][1]
            
            # Enhanced finger detection with multiple checks
            finger_extended = False
            
            # Primary check: tip above PIP joint
            if tip_y < pip_y - 10:  # Added 10 pixel threshold for stability
                # Secondary check: ensure finger is actually extended (not just bent weird)
                if pip_y < mcp_y + 5:  # PIP should be above or near MCP level
                    finger_extended = True
            
            fingers.append(1 if finger_extended else 0)
        
        return fingers
    
    def get_finger_positions(self, landmarks):
        """
        Get detailed finger positions for advanced gesture recognition.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Dictionary with detailed finger positions
        """
        if len(landmarks) < 21:
            return None
            
        finger_positions = {
            'thumb': {
                'tip': landmarks[4],
                'ip': landmarks[3],
                'mcp': landmarks[2]
            },
            'index': {
                'tip': landmarks[8],
                'pip': landmarks[6],
                'mcp': landmarks[5]
            },
            'middle': {
                'tip': landmarks[12],
                'pip': landmarks[10],
                'mcp': landmarks[9]
            },
            'ring': {
                'tip': landmarks[16],
                'pip': landmarks[14],
                'mcp': landmarks[13]
            },
            'pinky': {
                'tip': landmarks[20],
                'pip': landmarks[18],
                'mcp': landmarks[17]
            },
            'wrist': landmarks[0]
        }
        
        return finger_positions
    
    def draw_landmarks(self, frame, landmarks, hand_index=0):
        """
        Draw enhanced hand landmarks with better visualization.
        
        Args:
            frame: Video frame to draw on
            landmarks: List of landmark coordinates
            hand_index: Index of the hand (0 or 1) for color coding
        """
        if len(landmarks) < 21:
            return
        
        # Enhanced color scheme for different hands
        colors = [
            (0, 255, 0),    # Green for first hand
            (255, 0, 0),    # Red for second hand
            (0, 255, 255),  # Yellow for third hand
            (255, 0, 255)   # Magenta for fourth hand
        ]
        
        color = colors[hand_index % len(colors)]
        
        # Enhanced hand connections with different line styles
        connections = [
            # Thumb (thicker lines)
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky finger
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17)
        ]
        
        # Draw connections (bones)
        for connection in connections:
            start_idx, end_idx = connection
            
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = tuple(landmarks[start_idx])
                end_point = tuple(landmarks[end_idx])
                
                # Different thickness for different connection types
                thickness = 3 if start_idx == 0 else 2  # Thicker lines from wrist
                cv2.line(frame, start_point, end_point, color, thickness, cv2.LINE_AA)
        
        # Draw landmarks (joints) with enhanced styling
        for i, landmark in enumerate(landmarks):
            x, y = landmark
            
            # Different styles for different landmark types
            if i in self.tip_ids:  # Fingertips
                # Large yellow circles for fingertips
                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 8, (0, 0, 0), 2, cv2.LINE_AA)
                
                # Add finger labels
                finger_names = ['T', 'I', 'M', 'R', 'P']
                finger_idx = self.tip_ids.index(i)
                cv2.putText(frame, finger_names[finger_idx], (x-8, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
            elif i in self.pip_ids or i in self.mcp_ids:  # Joint landmarks
                # Medium circles for joints
                cv2.circle(frame, (x, y), 6, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 6, (0, 0, 0), 1, cv2.LINE_AA)
                
            elif i == 0:  # Wrist
                # Special marking for wrist
                cv2.circle(frame, (x, y), 10, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 10, color, 3, cv2.LINE_AA)
                cv2.putText(frame, f"H{hand_index+1}", (x-15, y+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                
            else:  # Other landmarks
                # Small circles for other points
                cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)
    
    def debug_fingers(self, frame, landmarks, hand_index=0):
        """
        Enhanced debug function with detailed finger state information.
        Shows finger states and gesture patterns on the video frame.
        
        Args:
            frame: Video frame to draw on
            landmarks: List of landmark coordinates  
            hand_index: Index of the hand for positioning
        """
        if len(landmarks) < 21:
            return
        
        fingers = self.fingers_up(landmarks)
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        
        # Enhanced debug display
        debug_y = 50 + (hand_index * 200)
        
        # Background for debug info
        cv2.rectangle(frame, (10, debug_y - 20), (350, debug_y + 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, debug_y - 20), (350, debug_y + 150), (255, 255, 255), 2)
        
        # Hand title
        cv2.putText(frame, f"Hand {hand_index + 1} Debug:", (15, debug_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Individual finger states
        for i, (name, state) in enumerate(zip(finger_names, fingers)):
            y_pos = debug_y + 25 + i * 20
            
            # Color coding: Green for up, Red for down
            color = (0, 255, 0) if state else (0, 0, 255)
            status = 'UP  ✓' if state else 'DOWN ✗'
            
            cv2.putText(frame, f"{name:>6}: {status}", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # Gesture pattern display
        gesture_pattern = str(tuple(fingers))
        cv2.putText(frame, f"Pattern: {gesture_pattern}", (15, debug_y + 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Performance info
        avg_time = self.get_average_processing_time()
        cv2.putText(frame, f"Avg Time: {avg_time:.1f}ms", (15, debug_y + 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1, cv2.LINE_AA)
    
    def is_hand_stable(self, landmarks, stability_threshold=10):
        """
        Check if hand position is stable (useful for gesture recognition).
        
        Args:
            landmarks: Current hand landmarks
            stability_threshold: Maximum movement threshold in pixels
            
        Returns:
            Boolean indicating if hand is stable
        """
        if len(landmarks) < 21:
            return False
        
        # Use wrist position as reference point
        current_wrist = landmarks[0]
        
        # Add to buffer
        self.last_gestures.append(current_wrist)
        if len(self.last_gestures) > self.gesture_buffer_size:
            self.last_gestures.pop(0)
        
        # Check stability
        if len(self.last_gestures) < self.gesture_buffer_size:
            return False
        
        # Calculate variance in position
        positions = np.array(self.last_gestures)
        variance = np.var(positions, axis=0)
        
        return np.max(variance) < stability_threshold ** 2
    
    def get_hand_center(self, landmarks):
        """
        Calculate the center point of the hand.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Tuple (x, y) representing hand center
        """
        if len(landmarks) < 21:
            return None
        
        # Calculate center as average of all landmark positions
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        
        center_x = sum(x_coords) // len(x_coords)
        center_y = sum(y_coords) // len(y_coords)
        
        return (center_x, center_y)
    
    def get_hand_bounding_box(self, landmarks):
        """
        Get bounding box of the hand.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Tuple (x, y, width, height) of bounding box
        """
        if len(landmarks) < 21:
            return None
        
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def calculate_finger_angles(self, landmarks):
        """
        Calculate angles between finger joints for advanced gesture recognition.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Dictionary with finger angles in degrees
        """
        if len(landmarks) < 21:
            return None
        
        def calculate_angle(p1, p2, p3):
            """Calculate angle between three points."""
            # Vectors
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)
        
        finger_angles = {}
        
        # Thumb angle
        finger_angles['thumb'] = calculate_angle(landmarks[2], landmarks[3], landmarks[4])
        
        # Other fingers
        finger_names = ['index', 'middle', 'ring', 'pinky']
        for i, name in enumerate(finger_names):
            mcp = landmarks[5 + i*4]
            pip = landmarks[6 + i*4]
            tip = landmarks[8 + i*4]
            finger_angles[name] = calculate_angle(mcp, pip, tip)
        
        return finger_angles
    
    def _update_performance_stats(self, processing_time):
        """Update performance statistics."""
        self.processing_times.append(processing_time)
        
        # Keep only recent samples
        if len(self.processing_times) > self.max_processing_samples:
            self.processing_times = self.processing_times[-self.max_processing_samples:]
    
    def get_average_processing_time(self):
        """Get average processing time in milliseconds."""
        if not self.processing_times:
            return 0.0
        return (sum(self.processing_times) / len(self.processing_times)) * 1000
    
    def get_performance_stats(self):
        """Get detailed performance statistics."""
        if not self.processing_times:
            return {
                'average_ms': 0.0,
                'min_ms': 0.0,
                'max_ms': 0.0,
                'samples': 0
            }
        
        times_ms = [t * 1000 for t in self.processing_times]
        return {
            'average_ms': sum(times_ms) / len(times_ms),
            'min_ms': min(times_ms),
            'max_ms': max(times_ms),
            'samples': len(times_ms)
        }
    
    def cleanup(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
            print("🧹 HandTracker resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.cleanup()

# Test function for standalone testing
def test_hand_tracker():
    """Test the HandTracker with a simple camera feed."""
    import cv2
    
    print("🧪 Testing HandTracker...")
    
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("📹 Camera opened. Press 'q' to quit, 'd' to toggle debug mode")
    
    debug_mode = False
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)
        
        # Process hands
        landmarks = tracker.process_frame(frame)
        
        # Draw landmarks
        for i, hand_landmarks in enumerate(landmarks):
            tracker.draw_landmarks(frame, hand_landmarks, i)
            
            if debug_mode:
                tracker.debug_fingers(frame, hand_landmarks, i)
                
                # Show finger positions
                fingers = tracker.fingers_up(hand_landmarks)
                print(f"Frame {frame_count}: Hand {i+1} fingers: {fingers}")
        
        # Show performance
        if landmarks:
            stats = tracker.get_performance_stats()
            cv2.putText(frame, f"Avg: {stats['average_ms']:.1f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('HandTracker Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    tracker.cleanup()
    print("✅ HandTracker test completed")

if __name__ == "__main__":
    # Run test if this file is executed directly
    test_hand_tracker()
