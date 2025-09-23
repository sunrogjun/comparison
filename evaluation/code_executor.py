"""
Safe code execution for candidate evaluation
"""
import os
import sys
import signal
import tempfile
import subprocess
import traceback
from typing import Dict, Any, List, Tuple, Optional
from contextlib import contextmanager
import multiprocessing as mp
from io import StringIO

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out")

class CodeExecutor:
    """Safe execution of candidate code with test cases"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    @contextmanager
    def time_limit(self, seconds):
        """Context manager for execution timeout"""
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    
    def extract_function_name(self, problem: Dict[str, Any]) -> str:
        """Extract the main function name from problem"""
        if 'entry_point' in problem:
            return problem['entry_point']
        
        # Try to extract from prompt
        prompt = problem.get('prompt', '')
        lines = prompt.strip().split('\n')
        for line in lines:
            if line.strip().startswith('def '):
                func_def = line.strip()
                func_name = func_def.split('(')[0].replace('def ', '').strip()
                return func_name
        
        return 'solution'  # Default fallback
    
    def prepare_test_code(self, problem: Dict[str, Any], candidate_code: str) -> str:
        """Prepare complete test code for execution"""
        
        # Get test cases
        test_code = problem.get('test', '')
        if not test_code.strip():
            return None
        
        function_name = self.extract_function_name(problem)
        
        # Check if candidate_code already has function definition
        if f'def {function_name}(' in candidate_code:
            # Complete function provided
            complete_candidate = candidate_code
        else:
            # Only function body provided, need to add function definition
            # Extract function signature from prompt
            prompt = problem.get('prompt', '')
            
            # Find the function signature in the prompt
            import re
            signature_match = re.search(rf'def {re.escape(function_name)}\([^)]*\)(?:\s*->\s*[^:]+)?:', prompt)
            if signature_match:
                signature = signature_match.group(0).rstrip(':')
            else:
                # Fallback signature
                signature = f"def {function_name}(*args, **kwargs)"
            
            # Clean and prepare candidate code
            lines = candidate_code.split('\n')
            indented_lines = []

            for line in lines:
                # Skip lines that are clearly not part of the function body
                stripped_line = line.strip()
                if (stripped_line.startswith('if __name__') or
                    stripped_line.startswith('import doctest') or
                    stripped_line.startswith('doctest.testmod')):
                    continue

                if line.strip():  # Non-empty line
                    # Ensure proper indentation (4 spaces)
                    if not line.startswith('    ') and not line.startswith('\t'):
                        indented_lines.append('    ' + line)
                    else:
                        indented_lines.append(line)
                else:
                    indented_lines.append(line)

            complete_candidate = f"{signature}:\n" + '\n'.join(indented_lines)
        
        # Extract any helper functions from the prompt
        prompt = problem.get('prompt', '')
        helper_functions = ""
        if prompt.strip():
            # Extract all function definitions from prompt except the main one
            lines = prompt.strip().split('\n')
            current_func = []
            in_function = False

            for line in lines:
                if line.strip().startswith('def ') and function_name not in line:
                    # Start of a helper function
                    in_function = True
                    current_func = [line]
                elif in_function:
                    if line.strip().startswith('def ') and function_name in line:
                        # Reached the main function, stop collecting helper
                        if current_func:
                            helper_functions += '\n'.join(current_func) + '\n\n'
                        break
                    elif line.strip() and not line.startswith(' ') and not line.startswith('\t') and not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                        # End of function (non-indented line that's not docstring)
                        if current_func:
                            helper_functions += '\n'.join(current_func) + '\n\n'
                        current_func = []
                        in_function = False
                    else:
                        current_func.append(line)

            # Don't forget the last function if file ends
            if in_function and current_func:
                helper_functions += '\n'.join(current_func) + '\n\n'

        # Prepare the complete executable code
        full_code = f"""
import sys
import traceback
import math
from typing import *

# Helper functions from prompt
{helper_functions}

# Candidate solution
{complete_candidate}

# Test code
{test_code}

# Execute test
try:
    check({function_name})
    print("PASSED")
except Exception as e:
    print(f"FAILED: {{str(e)}}")
    traceback.print_exc()
"""
        return full_code
    
    def execute_candidate_safe(self, problem: Dict[str, Any], candidate_code: str) -> Dict[str, Any]:
        """
        Safely execute a candidate solution with timeout and error handling
        
        Returns:
            Dict containing execution result with keys:
            - passed: bool indicating if tests passed
            - error: str containing error message if failed
            - timeout: bool indicating if execution timed out
            - execution_time: float execution time in seconds
        """
        
        result = {
            'passed': False,
            'error': None,
            'timeout': False,
            'execution_time': 0.0
        }
        
        # Skip empty candidates
        if not candidate_code.strip():
            result['error'] = "Empty candidate code"
            return result
        
        # Prepare test code
        test_code = self.prepare_test_code(problem, candidate_code)
        if test_code is None:
            result['error'] = "No test cases available"
            return result
        
        # Execute in a separate process for better isolation
        try:
            import time
            start_time = time.time()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_file = f.name
            
            try:
                # Run in subprocess with timeout
                process = subprocess.Popen(
                    [sys.executable, temp_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stdout, stderr = process.communicate(timeout=self.timeout)
                execution_time = time.time() - start_time
                result['execution_time'] = execution_time
                
                # Check result
                if process.returncode == 0 and "PASSED" in stdout:
                    result['passed'] = True
                else:
                    result['error'] = f"Exit code: {process.returncode}\\nSTDOUT: {stdout}\\nSTDERR: {stderr}"
                    
            except subprocess.TimeoutExpired:
                process.kill()
                result['timeout'] = True
                result['error'] = f"Execution timed out after {self.timeout} seconds"
                result['execution_time'] = self.timeout
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            result['error'] = f"Execution error: {str(e)}"
            result['execution_time'] = time.time() - start_time if 'start_time' in locals() else 0.0
        
        return result
    
    def evaluate_candidates(self, problem: Dict[str, Any], candidates: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate all candidates for a problem
        
        Returns:
            List of execution results for each candidate
        """
        results = []
        
        print(f"  Evaluating {len(candidates)} candidates...")
        
        for i, candidate in enumerate(candidates):
            if i > 0 and i % 20 == 0:
                print(f"    Progress: {i}/{len(candidates)}")
            
            result = self.execute_candidate_safe(problem, candidate)
            results.append(result)
        
        passed_count = sum(1 for r in results if r['passed'])
        print(f"  Evaluation completed: {passed_count}/{len(candidates)} passed")
        
        return results