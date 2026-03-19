"""
Enhanced expression evaluation with history, percentage, and unit conversions.
Handles mathematical expressions safely with comprehensive error handling.
"""

from sympy import sympify, SympifyError
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re
import json
from datetime import datetime
import math

class ExpressionEvaluator:
    def __init__(self):
        self.current_expression = ""
        self.last_result = ""
        self.history = []  # Calculation history
        self.max_history = 100  # Store last 100 calculations
        
        # Define allowed symbols and functions for safety
        self.allowed_names = {
            'pi': math.pi,
            'e': math.e,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'abs': abs,
            'round': round,
            'pow': pow,
            'max': max,
            'min': min
        }
        
        # SymPy transformations for parsing
        self.transformations = (
            standard_transformations + 
            (implicit_multiplication_application,)
        )
        
        # Unit conversion rates
        self.unit_conversions = {
            # Length conversions (to meters)
            'cm_to_m': 0.01,
            'mm_to_m': 0.001,
            'km_to_m': 1000,
            'ft_to_m': 0.3048,
            'in_to_m': 0.0254,
            
            # Weight conversions (to kg)
            'g_to_kg': 0.001,
            'lb_to_kg': 0.453592,
            'oz_to_kg': 0.0283495,
            
            # Temperature conversions
            'f_to_c': lambda f: (f - 32) * 5/9,
            'c_to_f': lambda c: c * 9/5 + 32,
            'k_to_c': lambda k: k - 273.15,
            'c_to_k': lambda c: c + 273.15
        }
    
    def add_token(self, token):
        """Add a token to the current expression with enhanced support."""
        if not token:
            return
        
        # Handle different token types
        if token in "0123456789":
            self.current_expression += token
        elif token == "%":
            # Convert percentage to decimal multiplication
            if self.current_expression and self.current_expression[-1].isdigit():
                self.current_expression += "*0.01"
            else:
                self.current_expression += "0.01"
        elif token in "+-*/":
            # Prevent consecutive operators
            if (self.current_expression and 
                self.current_expression[-1] not in "+-*/("):
                self.current_expression += token
            elif not self.current_expression and token in "+-":
                # Allow negative numbers at start
                self.current_expression += token
        elif token == ".":
            # Add decimal point if valid
            if (self.current_expression and 
                self.current_expression[-1].isdigit()):
                # Check if current number already has decimal
                last_number = re.split(r'[+\-*/()]', self.current_expression)[-1]
                if "." not in last_number:
                    self.current_expression += token
        elif token == "(":
            # Add opening parenthesis
            if (not self.current_expression or 
                self.current_expression[-1] in "+-*/("):
                self.current_expression += token
        elif token == ")":
            # Add closing parenthesis if there are unclosed opening ones
            open_count = self.current_expression.count("(")
            close_count = self.current_expression.count(")")
            if open_count > close_count:
                self.current_expression += token
        else:
            # Handle special functions or constants
            if token in self.allowed_names:
                if (not self.current_expression or 
                    self.current_expression[-1] in "+-*/("):
                    self.current_expression += token
    
    def backspace(self):
        """Remove the last character from expression."""
        if self.current_expression:
            self.current_expression = self.current_expression[:-1]
    
    def clear_expression(self):
        """Clear the current expression."""
        self.current_expression = ""
        self.last_result = ""
    
    def clear_history(self):
        """Clear calculation history."""
        self.history = []
        print("📊 History cleared - all calculation records removed")
    
    def evaluate_expression(self):
        """Safely evaluate the current expression with comprehensive error handling."""
        if not self.current_expression:
            return "Error: Empty expression"
        
        try:
            # Clean and validate the expression
            cleaned_expr = self._clean_expression(self.current_expression)
            
            if not cleaned_expr:
                return "Error: Invalid expression"
            
            # Handle simple arithmetic first (faster)
            if self._is_simple_arithmetic(cleaned_expr):
                result = self._evaluate_simple(cleaned_expr)
            else:
                # Use SymPy for complex expressions
                result = self._evaluate_complex(cleaned_expr)
            
            # Format result
            if isinstance(result, (int, float)):
                if abs(result) > 1e10:
                    formatted_result = f"{result:.2e}"  # Scientific notation for large numbers
                elif result == int(result):
                    formatted_result = str(int(result))
                else:
                    # Remove trailing zeros
                    formatted_result = f"{result:.10g}"
            else:
                formatted_result = str(result)
            
            # Add to history
            self._add_to_history(self.current_expression, formatted_result)
            
            # Update state
            self.last_result = formatted_result
            self.current_expression = formatted_result  # Allow chaining operations
            
            return formatted_result
            
        except ZeroDivisionError:
            error_msg = "Error: Division by zero"
            self.last_result = error_msg
            self._add_to_history(self.current_expression, error_msg)
            return error_msg
            
        except OverflowError:
            error_msg = "Error: Number too large"
            self.last_result = error_msg
            self._add_to_history(self.current_expression, error_msg)
            return error_msg
            
        except (SympifyError, ValueError, TypeError) as e:
            error_msg = f"Error: Invalid expression"
            self.last_result = error_msg
            self._add_to_history(self.current_expression, error_msg)
            return error_msg
            
        except Exception as e:
            error_msg = "Error: Calculation failed"
            self.last_result = error_msg
            self._add_to_history(self.current_expression, error_msg)
            print(f"Evaluation error: {e}")
            return error_msg
    
    def _is_simple_arithmetic(self, expr):
        """Check if expression contains only basic arithmetic."""
        # Only digits, basic operators, parentheses, and decimal points
        return re.match(r'^[0-9+\-*/.() ]+$', expr) is not None
    
    def _evaluate_simple(self, expr):
        """Evaluate simple arithmetic expressions safely."""
        try:
            # Replace any dangerous patterns
            expr = expr.replace(' ', '')
            
            # Basic validation
            if not expr or expr in "+-*/":
                raise ValueError("Invalid expression")
            
            # Use eval with restricted globals (only basic operations)
            allowed_names = {"__builtins__": {}}
            result = eval(expr, allowed_names, {})
            
            if not isinstance(result, (int, float)):
                raise ValueError("Invalid result type")
            
            return result
            
        except:
            # Fallback to SymPy if simple eval fails
            return self._evaluate_complex(expr)
    
    def _evaluate_complex(self, expr):
        """Evaluate complex expressions using SymPy."""
        parsed_expr = parse_expr(
            expr,
            transformations=self.transformations,
            local_dict=self.allowed_names,
            evaluate=True
        )
        
        result = float(parsed_expr)
        return result
    
    def _add_to_history(self, expression, result):
        """Add calculation to history with timestamp."""
        history_entry = {
            'expression': expression,
            'result': result,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'date': datetime.now().strftime("%Y-%m-%d"),
            'success': not result.startswith('Error') if isinstance(result, str) else True
        }
        
        self.history.append(history_entry)
        
        # Keep only last max_history calculations
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history(self):
        """Get calculation history."""
        return self.history.copy()
    
    def get_history_summary(self):
        """Get summary statistics of calculation history."""
        if not self.history:
            return {
                'total': 0,
                'successful': 0,
                'error_rate': 0,
                'avg_length': 0
            }
        
        total = len(self.history)
        successful = len([h for h in self.history if h.get('success', True)])
        error_rate = ((total - successful) / total) * 100
        avg_length = sum(len(h['expression']) for h in self.history) / total
        
        return {
            'total': total,
            'successful': successful,
            'error_rate': round(error_rate, 1),
            'avg_length': round(avg_length, 1),
            'success_rate': round((successful / total) * 100, 1)
        }
    
    def convert_units(self, value, conversion_type):
        """Convert units with error handling."""
        try:
            value = float(value)
            
            if conversion_type in self.unit_conversions:
                conversion_factor = self.unit_conversions[conversion_type]
                
                if callable(conversion_factor):
                    # For temperature conversions
                    result = conversion_factor(value)
                else:
                    # For linear conversions
                    result = value * conversion_factor
                
                return f"{result:.6g}"
            else:
                return f"Error: Unknown conversion '{conversion_type}'"
                
        except (ValueError, TypeError):
            return "Error: Invalid number for conversion"
        except Exception as e:
            return f"Error: Conversion failed"
    
    def _clean_expression(self, expr):
        """Clean and validate the expression."""
        if not expr:
            return ""
        
        # Remove extra whitespace
        expr = expr.strip()
        
        # Replace common symbols
        replacements = {
            '×': '*',
            '÷': '/',
            '−': '-',
            '√': 'sqrt'
        }
        
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        
        # Remove any non-mathematical characters (keeping parentheses and functions)
        cleaned = re.sub(r'[^0-9+\-*/.()abcdefghijklmnopqrstuvwxyz_]', '', expr)
        
        # Basic validation
        if not cleaned or cleaned in "+-*/":
            return ""
        
        # Check for balanced parentheses
        if cleaned.count('(') != cleaned.count(')'):
            # Try to balance parentheses
            open_count = cleaned.count('(')
            close_count = cleaned.count(')')
            if open_count > close_count:
                cleaned += ')' * (open_count - close_count)
        
        # Remove trailing operators
        cleaned = re.sub(r'[+\-*/]+$', '', cleaned)
        
        # Remove leading operators except minus
        cleaned = re.sub(r'^[+*/]+', '', cleaned)
        
        return cleaned.strip()
    
    def get_current_expression(self):
        """Get the current expression string."""
        return self.current_expression
    
    def get_last_result(self):
        """Get the last calculation result."""
        return self.last_result
    
    def export_history_json(self):
        """Export history as JSON string."""
        return json.dumps(self.history, indent=2)
    
    def import_history_json(self, json_string):
        """Import history from JSON string."""
        try:
            imported_history = json.loads(json_string)
            if isinstance(imported_history, list):
                self.history.extend(imported_history)
                # Keep only recent entries
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history:]
                return True
            return False
        except (json.JSONDecodeError, TypeError):
            return False
    
    def get_analytics_data(self):
        """Get comprehensive analytics data."""
        summary = self.get_history_summary()
        
        # Operation frequency
        operation_counts = {'+': 0, '-': 0, '*': 0, '/': 0}
        for entry in self.history:
            for op in operation_counts:
                operation_counts[op] += entry['expression'].count(op)
        
        # Recent activity (last 24 hours by hour)
        from collections import defaultdict
        hourly_activity = defaultdict(int)
        
        for entry in self.history:
            hour = entry.get('timestamp', '00:00:00').split(':')[0]
            hourly_activity[f"{hour}:00"] += 1
        
        return {
            **summary,
            'operation_counts': operation_counts,
            'hourly_activity': dict(hourly_activity),
            'recent_errors': [h for h in self.history[-10:] if not h.get('success', True)]
        }
