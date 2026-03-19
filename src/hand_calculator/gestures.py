"""
Enhanced gesture recognition with improved stability and performance.
"""

import time
import numpy as np
from typing import List, Tuple, Optional, Dict

class GestureRecognizer:
    """Advanced gesture recognition with stability tracking and cooldown management."""
    
    def __init__(self, stability_frames=3, cooldown_time=0.8):
        """
        Initialize gesture recognizer with optimized settings.
        
        Args:
            stability_frames: Number of consecutive frames needed for stable recognition
            cooldown_time: Minimum time between gesture recognitions (seconds)
        """
        self.stability_frames = stability_frames
        self.cooldown_time = cooldown_time
        
        # ✅ ADD MISSING ATTRIBUTES
        self.last_token_time = 0  # This was missing!
        self.last_gesture = None
        self.gesture_history = []
        self.stable_gesture_count = 0
        self.current_stable_gesture = None
        
        # Performance tracking
        self.recognition_count = 0
        self.successful_recognitions = 0
        
        # Enhanced gesture mapping
        self.gesture_patterns = {
            # Numbers (0-5)
            (0, 0, 0, 0, 0): "0",  # Closed fist
            (0, 1, 0, 0, 0): "1",  # Index finger
            (0, 1, 1, 0, 0): "2",  # Peace sign
            (0, 1, 1, 1, 0): "3",  # Three fingers
            (0, 1, 1, 1, 1): "4",  # Four fingers
            (1, 1, 1, 1, 1): "5",  # Open hand
            
            # Operations
            (1, 0, 0, 0, 0): "+",  # Thumb up
            (1, 1, 0, 0, 0): "-",  # Thumb + index
            (0, 0, 0, 0, 1): "=",  # Pinky
            (1, 0, 0, 0, 1): "C",  # Thumb + pinky (clear)
            
            # Additional operations
            (1, 0, 1, 0, 0): "*",  # Thumb + middle
            (0, 1, 0, 1, 0): "/",  # Index + ring
        }
        
        print(f"🎯 GestureRecognizer initialized - Stability: {stability_frames} frames, Cooldown: {cooldown_time}s")
    
    def recognize_gesture(self, landmarks_list: List[List]) -> Optional[str]:
        """
        Recognize gesture from hand landmarks.
        
        Args:
            landmarks_list: List of hand landmark coordinates
            
        Returns:
            Recognized gesture string or None
        """
        if not landmarks_list or len(landmarks_list) == 0:
            return None
        
        try:
            self.recognition_count += 1
            
            # Use first hand for gesture recognition
            hand_landmarks = landmarks_list[0]
            
            # Get finger states (assuming fingers_up method exists in HandTracker)
            fingers = self._extract_finger_states(hand_landmarks)
            
            # Map finger pattern to gesture
            gesture = self.gesture_patterns.get(tuple(fingers))
            
            if gesture:
                self.successful_recognitions += 1
                print(f"🔍 Gesture detected: {fingers} = {gesture}")
            else:
                print(f"❓ Unknown pattern: {fingers}")
            
            return gesture
            
        except Exception as e:
            print(f"⚠️ Gesture recognition error: {e}")
            return None
    
    def _extract_finger_states(self, landmarks: List) -> List[int]:
        """
        Extract finger up/down states from landmarks.
        
        Args:
            landmarks: Hand landmark coordinates
            
        Returns:
            List of [thumb, index, middle, ring, pinky] states (1=up, 0=down)
        """
        if len(landmarks) < 21:
            return [0, 0, 0, 0, 0]
        
        fingers = []
        
        # Thumb detection (landmark 4 vs landmark 3)
        if landmarks[4][0] > landmarks[3][0]:  # Thumb tip right of IP joint
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers (tips vs PIP joints)
        tip_ids = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        pip_ids = [6, 10, 14, 18]  # PIP joints
        
        for tip, pip in zip(tip_ids, pip_ids):
            if landmarks[tip][1] < landmarks[pip][1]:  # Tip above PIP
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def update_gesture_state(self, current_gesture: Optional[str]) -> Optional[str]:
        """
        Update gesture state with stability tracking and cooldown.
        
        Args:
            current_gesture: Currently detected gesture
            
        Returns:
            Stable gesture token or None
        """
        current_time = time.time()
        
        # ✅ FIX: Check cooldown with proper attribute
        if current_time - self.last_token_time < self.cooldown_time:
            return None
        
        if not current_gesture:
            # Reset stability when no gesture
            self.gesture_history = []
            self.stable_gesture_count = 0
            self.current_stable_gesture = None
            return None
        
        # Add to gesture history
        self.gesture_history.append(current_gesture)
        
        # Keep only recent history
        if len(self.gesture_history) > 10:
            self.gesture_history = self.gesture_history[-10:]
        
        # Check for stability
        recent_gestures = self.gesture_history[-self.stability_frames:]
        
        if (len(recent_gestures) >= self.stability_frames and 
            all(g == current_gesture for g in recent_gestures)):
            
            # Check if this is a new stable gesture
            if (current_gesture != self.current_stable_gesture or 
                current_time - self.last_token_time >= self.cooldown_time):
                
                self.current_stable_gesture = current_gesture
                self.last_token_time = current_time
                self.gesture_history = []  # Reset after successful recognition
                
                print(f"✅ Stable gesture accepted: {current_gesture}")
                return current_gesture
        
        # Track stability progress
        same_gesture_count = sum(1 for g in recent_gestures if g == current_gesture)
        print(f"📊 Stability progress: {same_gesture_count}/{self.stability_frames}")
        
        return None
    
    def get_stability_info(self) -> Dict:
        """Get current stability information for display."""
        if not self.gesture_history:
            return {'count': 0, 'required': self.stability_frames}
        
        recent_gestures = self.gesture_history[-self.stability_frames:]
        if not recent_gestures:
            return {'count': 0, 'required': self.stability_frames}
        
        last_gesture = recent_gestures[-1]
        count = sum(1 for g in recent_gestures if g == last_gesture)
        
        return {'count': count, 'required': self.stability_frames}
    
    def get_performance_stats(self) -> Dict:
        """Get recognition performance statistics."""
        success_rate = (self.successful_recognitions / max(self.recognition_count, 1)) * 100
        
        return {
            'total_recognitions': self.recognition_count,
            'successful_recognitions': self.successful_recognitions,
            'success_rate': round(success_rate, 1),
            'cooldown_time': self.cooldown_time,
            'stability_frames': self.stability_frames
        }
    
    def reset(self):
        """Reset all gesture state."""
        self.gesture_history = []
        self.stable_gesture_count = 0
        self.current_stable_gesture = None
        self.last_token_time = 0
        print("🔄 Gesture recognizer reset")
    
    def set_sensitivity(self, stability_frames: int, cooldown_time: float):
        """
        Adjust recognition sensitivity.
        
        Args:
            stability_frames: Number of frames for stability
            cooldown_time: Cooldown between recognitions
        """
        self.stability_frames = stability_frames
        self.cooldown_time = cooldown_time
        print(f"⚙️ Sensitivity updated - Stability: {stability_frames}, Cooldown: {cooldown_time}s")

# ✅ ADD COMPATIBILITY CLASSES
class FastGestureRecognizer(GestureRecognizer):
    """Fast gesture recognizer for quick response."""
    
    def __init__(self):
        super().__init__(stability_frames=2, cooldown_time=0.5)

class StableGestureRecognizer(GestureRecognizer):
    """Stable gesture recognizer for accuracy."""
    
    def __init__(self):
        super().__init__(stability_frames=4, cooldown_time=1.0)

class UltraFastGestureRecognizer(GestureRecognizer):
    """Ultra-fast gesture recognizer for immediate response."""
    
    def __init__(self):
        super().__init__(stability_frames=1, cooldown_time=0.3)
