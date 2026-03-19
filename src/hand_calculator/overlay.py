"""
UI overlay for displaying calculator interface using OpenCV.
"""

import cv2
import numpy as np

class UIOverlay:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2
        self.line_type = cv2.LINE_AA
        
        # Colors (BGR)
        self.colors = {
            'background': (50, 50, 50),
            'text': (255, 255, 255),
            'expression': (0, 255, 255),  # Yellow
            'result': (0, 255, 0),        # Green  
            'gesture': (255, 0, 255),     # Magenta
            'progress': (0, 165, 255),    # Orange
            'error': (0, 0, 255)          # Red
        }
    
    def draw_overlay(self, frame, expression, result, current_gesture, stability_info):
        """Draw the complete UI overlay on the frame."""
        h, w = frame.shape[:2]
        
        # Draw semi-transparent background panel
        self._draw_panel(frame, 10, 10, w-20, 120)
        
        # Draw expression
        self._draw_text(frame, f"Expression: {expression or 'None'}", 
                       20, 40, self.colors['expression'])
        
        # Draw result
        result_color = self.colors['error'] if result and 'Error' in result else self.colors['result']
        self._draw_text(frame, f"Result: {result or 'None'}", 
                       20, 70, result_color)
        
        # Draw current gesture
        self._draw_text(frame, f"Gesture: {current_gesture or 'None'}", 
                       20, 100, self.colors['gesture'])
        
        # Draw stability progress bar
        self._draw_progress_bar(frame, stability_info, 20, 110, 200, 15)
        
        # Draw instructions
        self._draw_instructions(frame, w, h)
        
        # Draw gesture reference
        self._draw_gesture_reference(frame, w)
    
    def _draw_panel(self, frame, x, y, width, height):
        """Draw semi-transparent background panel."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), 
                     self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + width, y + height), 
                     self.colors['text'], 2)
    
    def _draw_text(self, frame, text, x, y, color):
        """Draw text with shadow for better visibility."""
        # Shadow
        cv2.putText(frame, text, (x+2, y+2), self.font, self.font_scale, 
                   (0, 0, 0), self.thickness, self.line_type)
        # Main text
        cv2.putText(frame, text, (x, y), self.font, self.font_scale, 
                   color, self.thickness, self.line_type)
    
    def _draw_progress_bar(self, frame, stability_info, x, y, width, height):
        """Draw gesture stability progress bar."""
        if not stability_info['current_gesture']:
            return
        
        progress = stability_info['progress']
        filled_width = int(width * progress)
        
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), 
                     (100, 100, 100), -1)
        
        # Progress fill
        if filled_width > 0:
            cv2.rectangle(frame, (x, y), (x + filled_width, y + height), 
                         self.colors['progress'], -1)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), 
                     self.colors['text'], 1)
        
        # Progress text
        progress_text = f"{stability_info['count']}/{stability_info['required']}"
        text_x = x + width + 10
        cv2.putText(frame, progress_text, (text_x, y + height), 
                   self.font, 0.5, self.colors['text'], 1, self.line_type)
    
    def _draw_instructions(self, frame, width, height):
        """Draw usage instructions."""
        instructions = [
            "Controls:",
            "1-5: Fingers up for numbers",
            "Thumb only: +",
            "Thumb + Index: -", 
            "Pinky only: =",
            "Press 'q' to quit"
        ]
        
        start_y = height - len(instructions) * 25 - 10
        
        for i, instruction in enumerate(instructions):
            y_pos = start_y + i * 25
            self._draw_text(frame, instruction, 20, y_pos, self.colors['text'])
    
    def _draw_gesture_reference(self, frame, width):
        """Draw gesture reference guide."""
        reference_x = width - 300
        reference_y = 150
        
        gestures = [
            "1: Index up",
            "2: Index + Middle",
            "3: Index + Middle + Ring", 
            "4: All except thumb",
            "5: All fingers",
            "+: Thumb only",
            "-: Thumb + Index",
            "=: Pinky only",
            "C: Thumb + Pinky"
        ]
        
        # Draw reference panel
        panel_height = len(gestures) * 25 + 20
        self._draw_panel(frame, reference_x - 10, reference_y - 10, 
                        280, panel_height)
        
        # Draw title
        self._draw_text(frame, "Gesture Reference:", reference_x, reference_y + 15, 
                       self.colors['text'])
        
        # Draw gestures
        for i, gesture in enumerate(gestures):
            y_pos = reference_y + 40 + i * 25
            self._draw_text(frame, gesture, reference_x, y_pos, 
                           self.colors['gesture'])
