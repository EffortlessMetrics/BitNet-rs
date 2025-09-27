"""
Cross-language validation framework for BitNet Python to Rust migration.
"""
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch
import time
import hashlib
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for cross-language validation."""
    numerical_tolerance: Dict[str, float]
    performance_tolerance: Dict[str, float]
    timeout_seconds: int = 300
    max_retries: int = 3
    temp_dir: Optional[Path] = None
    
    def __post_init__(self):
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="bitnet_validation_"))

@dataclass
class TestCase:
    """A single test case for cross-validation."""
    name: str
    inputs: Dict[str, Any]
    expected_outputs: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class ValidationResult:
    """Result of a cross-language validation."""
    test_name: str
    python_success: bool
    rust_success: bool
    outputs_match: bool
    numerical_errors: List[str]
    performance_comparison: Dict[str, float]
    execution_times: Dict[str, float]
    error_messages: List[str]
    
    @property
    def overall_success(self) -> bool:
        """Check if validation passed overall."""
        return (self.python_success and 
                self.rust_success and 
                self.outputs_match and 
                len(self.numerical_errors) == 0)

class ImplementationRunner(ABC):
    """Abstract base class for running implementations."""
    
    @abstractmethod
    def run_test_case(self, test_case: TestCase, config: ValidationConfig) -> Tuple[bool, Dict[str, Any], float, List[str]]:
        """
        Run a test case and return (success, outputs, execution_time, errors).
        """
        pass
    
    @abstractmethod
    def get_version_info(self) -> Dict[str, str]:
        """Get version information for this implementation."""
        pass

class PythonRunner(ImplementationRunner):
    """Runner for Python BitNet implementation."""
    
    def __init__(self, python_path: Optional[str] = None):
        self.python_path = python_path or "python"
        self.project_root = Path(__file__).parent.parent.parent
        self._validate_environment()
    
    def _validate_environment(self):
        """Validate that Python environment is set up correctly."""
        try:
            import torch
            import numpy as np
            # Try to import BitNet modules
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))
            sys.path.insert(0, str(project_root / "gpu"))
            sys.path.insert(0, str(project_root / "utils"))
            
            from gpu.model import ModelArgs, Transformer
            logger.info("Python environment validated successfully")
        except ImportError as e:
            raise RuntimeError(f"Python environment validation failed: {e}")
    
    def run_test_case(self, test_case: TestCase, config: ValidationConfig) -> Tuple[bool, Dict[str, Any], float, List[str]]:
        """Run test case in Python implementation."""
        try:
            start_time = time.time()
            
            # Import required modules
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))
            sys.path.insert(0, str(project_root / "gpu"))
            sys.path.insert(0, str(project_root / "utils"))
            
            from gpu.model import ModelArgs, Transformer, make_cache
            
            # Execute test case based on type
            if test_case.name.startswith("model_forward"):
                outputs = self._run_model_forward(test_case)
            elif test_case.name.startswith("quantization"):
                outputs = self._run_quantization(test_case)
            elif test_case.name.startswith("inference"):
                outputs = self._run_inference(test_case)
            else:
                raise ValueError(f"Unknown test case type: {test_case.name}")
            
            execution_time = time.time() - start_time
            return True, outputs, execution_time, []
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Python test case failed: {e}")
            return False, {}, execution_time, [str(e)]
    
    def _run_model_forward(self, test_case: TestCase) -> Dict[str, Any]:
        """Run model forward pass test."""
        from gpu.model import ModelArgs, Transformer, make_cache
        
        # Extract inputs
        config = test_case.inputs["config"]
        model_inputs = test_case.inputs["model_inputs"]
        
        # Create model
        torch.manual_seed(42)  # Ensure reproducibility
        args = ModelArgs(**config)
        model = Transformer(args)
        model.eval()
        
        # Prepare inputs
        token_values = torch.tensor(model_inputs["token_values"])
        token_lengths = torch.tensor(model_inputs["token_lengths"])
        start_pos = torch.tensor(model_inputs["start_pos"])
        cache = make_cache(args, model_inputs["cache_len"])
        
        # Forward pass
        with torch.no_grad():
            logits = model(token_values, token_lengths, start_pos, cache, model_inputs["cache_len"])
        
        return {
            "logits": logits.cpu().numpy().tolist(),
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
        }
    
    def _run_quantization(self, test_case: TestCase) -> Dict[str, Any]:
        """Run quantization test."""
        from gpu.pack_weight import convert_weight_int8_to_int2
        
        # Extract inputs
        weights = torch.tensor(test_case.inputs["weights"])
        
        # Apply quantization
        quantized = torch.sign(weights)
        int8_weights = quantized.to(torch.int8)
        
        outputs = {
            "quantized": quantized.cpu().numpy().tolist(),
            "int8_weights": int8_weights.cpu().numpy().tolist(),
        }
        
        # Convert to int2 if possible
        if weights.shape[1] % 4 == 0:
            int2_weights = convert_weight_int8_to_int2(int8_weights)
            outputs["int2_weights"] = int2_weights.cpu().numpy().tolist()
        
        return outputs
    
    def _run_inference(self, test_case: TestCase) -> Dict[str, Any]:
        """Run inference test using subprocess to call original BitNet.cpp."""
        try:
            # Extract inputs
            prompt = test_case.inputs.get("prompt", "Hello world")
            max_tokens = test_case.inputs.get("max_tokens", 10)
            temperature = test_case.inputs.get("temperature", 0.8)
            model_path = test_case.inputs.get("model_path", "models/bitnet_b1_58-3B/ggml-model-i2_s.gguf")
            
            # Use subprocess to run original BitNet inference
            cmd = [
                self.python_path,
                str(self.project_root / "run_inference.py"),
                "-m", model_path,
                "-p", prompt,
                "-n", str(max_tokens),
                "-temp", str(temperature),
                "-t", "1",  # Single thread for reproducibility
                "-c", "512"  # Small context for testing
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.project_root
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Inference failed: {result.stderr}")
            
            # Parse output to extract generated tokens
            output_text = result.stdout.strip()
            
            # For now, return the raw output - in a real implementation,
            # we would parse this to extract tokens and logits
            return {
                "generated_text": output_text,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "raw_output": output_text
            }
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Inference subprocess timed out")
        except Exception as e:
            raise RuntimeError(f"Inference subprocess failed: {e}")
    
    def get_version_info(self) -> Dict[str, str]:
        """Get Python implementation version info."""
        import torch
        import numpy as np
        import sys
        
        return {
            "implementation": "python",
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }

class RustRunner(ImplementationRunner):
    """Runner for Rust BitNet implementation."""
    
    def __init__(self, rust_binary_path: Optional[Path] = None):
        self.rust_binary_path = rust_binary_path or self._find_rust_binary()
        self._validate_environment()
    
    def _find_rust_binary(self) -> Path:
        """Find the Rust binary for BitNet."""
        # Look for common locations
        possible_paths = [
            Path("target/release/bitnet-cli"),
            Path("target/debug/bitnet-cli"),
            Path("./bitnet-cli"),
            Path("../target/release/bitnet-cli"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Try to find in PATH
        result = shutil.which("bitnet-cli")
        if result:
            return Path(result)
        
        raise RuntimeError("Could not find Rust BitNet binary. Please build the project first.")
    
    def _validate_environment(self):
        """Validate that Rust environment is set up correctly."""
        if not self.rust_binary_path.exists():
            raise RuntimeError(f"Rust binary not found: {self.rust_binary_path}")
        
        # Test that binary runs
        try:
            result = subprocess.run(
                [str(self.rust_binary_path), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"Rust binary failed to run: {result.stderr}")
            logger.info("Rust environment validated successfully")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Rust binary validation timed out")
    
    def run_test_case(self, test_case: TestCase, config: ValidationConfig) -> Tuple[bool, Dict[str, Any], float, List[str]]:
        """Run test case in Rust implementation."""
        try:
            start_time = time.time()
            
            # Create temporary files for input/output
            input_file = config.temp_dir / f"{test_case.name}_input.json"
            output_file = config.temp_dir / f"{test_case.name}_output.json"
            
            # Write input data
            with open(input_file, 'w') as f:
                json.dump(test_case.inputs, f)
            
            # Run Rust implementation
            cmd = [
                str(self.rust_binary_path),
                "validate",
                "--input", str(input_file),
                "--output", str(output_file),
                "--test-type", self._get_test_type(test_case.name)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode != 0:
                return False, {}, execution_time, [result.stderr]
            
            # Read output data
            if output_file.exists():
                with open(output_file, 'r') as f:
                    outputs = json.load(f)
            else:
                outputs = {}
            
            return True, outputs, execution_time, []
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return False, {}, execution_time, ["Rust execution timed out"]
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Rust test case failed: {e}")
            return False, {}, execution_time, [str(e)]
    
    def _get_test_type(self, test_name: str) -> str:
        """Get test type for Rust CLI."""
        if test_name.startswith("model_forward"):
            return "model-forward"
        elif test_name.startswith("quantization"):
            return "quantization"
        elif test_name.startswith("inference"):
            return "inference"
        else:
            return "unknown"
    
    def get_version_info(self) -> Dict[str, str]:
        """Get Rust implementation version info."""
        try:
            result = subprocess.run(
                [str(self.rust_binary_path), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                "implementation": "rust",
                "version_output": result.stdout.strip(),
                "binary_path": str(self.rust_binary_path),
            }
        except Exception as e:
            return {
                "implementation": "rust",
                "error": str(e),
                "binary_path": str(self.rust_binary_path),
            }

class TokenLevelComparator:
    """Utilities for token-level comparison with configurable tolerance."""
    
    def __init__(self, tolerance_config: Dict[str, float]):
        self.tolerance_config = tolerance_config
    
    def compare_token_sequences(self, seq1: List[int], seq2: List[int]) -> Tuple[bool, Dict[str, Any]]:
        """Compare two token sequences with detailed analysis."""
        if len(seq1) != len(seq2):
            return False, {
                "error": "sequence_length_mismatch",
                "seq1_length": len(seq1),
                "seq2_length": len(seq2),
                "length_diff": abs(len(seq1) - len(seq2))
            }
        
        # Exact match check
        exact_matches = sum(1 for t1, t2 in zip(seq1, seq2) if t1 == t2)
        total_tokens = len(seq1)
        exact_match_ratio = exact_matches / total_tokens if total_tokens > 0 else 0.0
        
        # Token-level differences
        differences = []
        for i, (t1, t2) in enumerate(zip(seq1, seq2)):
            if t1 != t2:
                differences.append({
                    "position": i,
                    "token1": t1,
                    "token2": t2,
                    "diff": abs(t1 - t2)
                })
        
        # Check if within tolerance
        min_match_ratio = self.tolerance_config.get("min_token_match_ratio", 0.95)
        max_differences = self.tolerance_config.get("max_token_differences", 5)
        
        within_tolerance = (
            exact_match_ratio >= min_match_ratio and
            len(differences) <= max_differences
        )
        
        return within_tolerance, {
            "exact_match_ratio": exact_match_ratio,
            "total_tokens": total_tokens,
            "exact_matches": exact_matches,
            "differences": differences,
            "within_tolerance": within_tolerance,
            "tolerance_config": self.tolerance_config
        }
    
    def compare_logits(self, logits1: List[List[float]], logits2: List[List[float]]) -> Tuple[bool, Dict[str, Any]]:
        """Compare logits with numerical tolerance."""
        try:
            arr1 = np.array(logits1)
            arr2 = np.array(logits2)
            
            if arr1.shape != arr2.shape:
                return False, {
                    "error": "shape_mismatch",
                    "shape1": arr1.shape,
                    "shape2": arr2.shape
                }
            
            # Numerical comparison
            rtol = self.tolerance_config.get("logits_rtol", 1e-3)
            atol = self.tolerance_config.get("logits_atol", 1e-4)
            
            close_mask = np.isclose(arr1, arr2, rtol=rtol, atol=atol)
            close_ratio = np.mean(close_mask)
            
            max_abs_diff = np.max(np.abs(arr1 - arr2))
            mean_abs_diff = np.mean(np.abs(arr1 - arr2))
            
            # Check tolerance
            min_close_ratio = self.tolerance_config.get("min_logits_close_ratio", 0.99)
            max_abs_diff_threshold = self.tolerance_config.get("max_logits_abs_diff", 0.1)
            
            within_tolerance = (
                close_ratio >= min_close_ratio and
                max_abs_diff <= max_abs_diff_threshold
            )
            
            return within_tolerance, {
                "close_ratio": close_ratio,
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "within_tolerance": within_tolerance,
                "rtol": rtol,
                "atol": atol
            }
            
        except Exception as e:
            return False, {"error": f"comparison_failed: {e}"}

class PerformanceAnalyzer:
    """Tools for performance comparison and regression detection."""
    
    def __init__(self, regression_threshold: float = 5.0):
        self.regression_threshold = regression_threshold
        self.baseline_times = {}
    
    def record_baseline(self, test_name: str, execution_time: float):
        """Record baseline execution time for a test."""
        self.baseline_times[test_name] = execution_time
    
    def analyze_performance(self, test_name: str, python_time: float, rust_time: float) -> Dict[str, Any]:
        """Analyze performance comparison between implementations."""
        analysis = {
            "python_time": python_time,
            "rust_time": rust_time,
            "speedup": 0.0,
            "regression_percent": 0.0,
            "performance_status": "unknown",
            "baseline_comparison": {}
        }
        
        # Calculate speedup
        if rust_time > 0:
            analysis["speedup"] = python_time / rust_time
            analysis["regression_percent"] = ((rust_time - python_time) / python_time) * 100
            
            # Determine performance status
            if analysis["speedup"] >= 1.0:
                analysis["performance_status"] = "improvement"
            elif abs(analysis["regression_percent"]) <= self.regression_threshold:
                analysis["performance_status"] = "acceptable"
            else:
                analysis["performance_status"] = "regression"
        
        # Compare against baseline if available
        if test_name in self.baseline_times:
            baseline_time = self.baseline_times[test_name]
            analysis["baseline_comparison"] = {
                "baseline_time": baseline_time,
                "python_vs_baseline": ((python_time - baseline_time) / baseline_time) * 100,
                "rust_vs_baseline": ((rust_time - baseline_time) / baseline_time) * 100 if rust_time > 0 else 0.0
            }
        
        return analysis
    
    def detect_regressions(self, performance_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect performance regressions exceeding threshold."""
        regressions = []
        
        for result in performance_results:
            if result.get("performance_status") == "regression":
                regressions.append({
                    "test_name": result.get("test_name", "unknown"),
                    "regression_percent": result.get("regression_percent", 0.0),
                    "python_time": result.get("python_time", 0.0),
                    "rust_time": result.get("rust_time", 0.0),
                    "speedup": result.get("speedup", 0.0)
                })
        
        return regressions
    
    def generate_performance_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive performance summary."""
        if not results:
            return {"error": "No performance results to analyze"}
        
        speedups = [r.get("speedup", 0) for r in results if r.get("speedup", 0) > 0]
        regression_percents = [r.get("regression_percent", 0) for r in results]
        
        summary = {
            "total_tests": len(results),
            "improvements": sum(1 for r in results if r.get("performance_status") == "improvement"),
            "acceptable": sum(1 for r in results if r.get("performance_status") == "acceptable"),
            "regressions": sum(1 for r in results if r.get("performance_status") == "regression"),
            "speedup_stats": {
                "mean": np.mean(speedups) if speedups else 0.0,
                "median": np.median(speedups) if speedups else 0.0,
                "min": min(speedups) if speedups else 0.0,
                "max": max(speedups) if speedups else 0.0,
                "std": np.std(speedups) if speedups else 0.0
            },
            "regression_stats": {
                "mean": np.mean(regression_percents),
                "median": np.median(regression_percents),
                "min": min(regression_percents),
                "max": max(regression_percents),
                "std": np.std(regression_percents)
            }
        }
        
        return summary

class CrossLanguageValidator:
    """Main validator for cross-language testing."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.python_runner = PythonRunner()
        self.rust_runner = None  # Will be initialized when Rust binary is available
        self.token_comparator = TokenLevelComparator(config.numerical_tolerance)
        self.performance_analyzer = PerformanceAnalyzer(
            config.performance_tolerance.get("max_regression_percent", 5.0)
        )
        
    def set_rust_runner(self, rust_binary_path: Optional[Path] = None):
        """Set up Rust runner when binary is available."""
        try:
            self.rust_runner = RustRunner(rust_binary_path)
            logger.info("Rust runner initialized successfully")
        except RuntimeError as e:
            logger.warning(f"Could not initialize Rust runner: {e}")
    
    def validate_test_case(self, test_case: TestCase) -> ValidationResult:
        """Validate a single test case across both implementations."""
        logger.info(f"Validating test case: {test_case.name}")
        
        # Run Python implementation
        python_success, python_outputs, python_time, python_errors = \
            self.python_runner.run_test_case(test_case, self.config)
        
        # Run Rust implementation if available
        if self.rust_runner is not None:
            rust_success, rust_outputs, rust_time, rust_errors = \
                self.rust_runner.run_test_case(test_case, self.config)
        else:
            logger.warning("Rust runner not available, skipping Rust validation")
            rust_success, rust_outputs, rust_time, rust_errors = False, {}, 0.0, ["Rust runner not available"]
        
        # Compare outputs
        outputs_match, numerical_errors = self._compare_outputs(python_outputs, rust_outputs)
        
        # Compare performance
        performance_comparison = self.performance_analyzer.analyze_performance(
            test_case.name, python_time, rust_time
        )
        
        return ValidationResult(
            test_name=test_case.name,
            python_success=python_success,
            rust_success=rust_success,
            outputs_match=outputs_match,
            numerical_errors=numerical_errors,
            performance_comparison=performance_comparison,
            execution_times={"python": python_time, "rust": rust_time},
            error_messages=python_errors + rust_errors
        )
    
    def validate_test_suite(self, test_cases: List[TestCase]) -> List[ValidationResult]:
        """Validate a suite of test cases."""
        results = []
        
        for test_case in test_cases:
            try:
                result = self.validate_test_case(test_case)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to validate test case {test_case.name}: {e}")
                # Create a failed result
                result = ValidationResult(
                    test_name=test_case.name,
                    python_success=False,
                    rust_success=False,
                    outputs_match=False,
                    numerical_errors=[f"Validation framework error: {e}"],
                    performance_comparison={},
                    execution_times={"python": 0.0, "rust": 0.0},
                    error_messages=[str(e)]
                )
                results.append(result)
        
        return results
    
    def _compare_outputs(self, python_outputs: Dict[str, Any], rust_outputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Compare outputs between implementations with enhanced token-level comparison."""
        if not python_outputs or not rust_outputs:
            return False, ["One or both implementations produced no output"]
        
        errors = []
        
        # Compare each output field
        for key in python_outputs:
            if key not in rust_outputs:
                errors.append(f"Missing key in Rust output: {key}")
                continue
            
            python_val = python_outputs[key]
            rust_val = rust_outputs[key]
            
            # Special handling for token sequences
            if key in ["tokens", "generated_tokens"] and isinstance(python_val, list) and isinstance(rust_val, list):
                # Use token-level comparison
                if all(isinstance(x, int) for x in python_val) and all(isinstance(x, int) for x in rust_val):
                    match, details = self.token_comparator.compare_token_sequences(python_val, rust_val)
                    if not match:
                        errors.append(f"Token sequence mismatch for {key}: {details}")
                    continue
            
            # Special handling for logits
            if key in ["logits"] and isinstance(python_val, list) and isinstance(rust_val, list):
                try:
                    match, details = self.token_comparator.compare_logits(python_val, rust_val)
                    if not match:
                        errors.append(f"Logits mismatch for {key}: {details}")
                    continue
                except Exception as e:
                    errors.append(f"Error comparing logits for {key}: {e}")
                    continue
            
            # General numerical comparison
            if isinstance(python_val, (list, np.ndarray)) and isinstance(rust_val, (list, np.ndarray)):
                try:
                    python_array = np.array(python_val)
                    rust_array = np.array(rust_val)
                    
                    if python_array.shape != rust_array.shape:
                        errors.append(f"Shape mismatch for {key}: Python {python_array.shape} vs Rust {rust_array.shape}")
                        continue
                    
                    # Check for numerical equality within tolerance
                    rtol = self.config.numerical_tolerance.get("rtol", 1e-4)
                    atol = self.config.numerical_tolerance.get("atol", 1e-5)
                    
                    if not np.allclose(python_array, rust_array, rtol=rtol, atol=atol):
                        max_diff = np.max(np.abs(python_array - rust_array))
                        errors.append(f"Numerical mismatch for {key}: max difference {max_diff}")
                
                except Exception as e:
                    errors.append(f"Error comparing {key}: {e}")
            
            else:
                # Direct comparison
                if python_val != rust_val:
                    errors.append(f"Value mismatch for {key}: Python {python_val} vs Rust {rust_val}")
        
        # Check for extra keys in Rust output
        for key in rust_outputs:
            if key not in python_outputs:
                errors.append(f"Extra key in Rust output: {key}")
        
        return len(errors) == 0, errors
    
    def _compare_performance(self, python_time: float, rust_time: float) -> Dict[str, float]:
        """Compare performance between implementations."""
        if python_time <= 0 or rust_time <= 0:
            return {"speedup": 0.0, "python_time": python_time, "rust_time": rust_time}
        
        speedup = python_time / rust_time
        regression_percent = ((rust_time - python_time) / python_time) * 100
        
        return {
            "speedup": speedup,
            "regression_percent": regression_percent,
            "python_time": python_time,
            "rust_time": rust_time,
        }
    
    def generate_report(self, results: List[ValidationResult], output_path: Path):
        """Generate a comprehensive validation report."""
        report = {
            "summary": {
                "total_tests": len(results),
                "passed_tests": sum(1 for r in results if r.overall_success),
                "failed_tests": sum(1 for r in results if not r.overall_success),
                "python_failures": sum(1 for r in results if not r.python_success),
                "rust_failures": sum(1 for r in results if not r.rust_success),
                "output_mismatches": sum(1 for r in results if not r.outputs_match),
            },
            "performance_summary": {
                "average_speedup": np.mean([r.performance_comparison.get("speedup", 0) for r in results if r.performance_comparison.get("speedup", 0) > 0]),
                "max_speedup": max([r.performance_comparison.get("speedup", 0) for r in results], default=0),
                "min_speedup": min([r.performance_comparison.get("speedup", 0) for r in results if r.performance_comparison.get("speedup", 0) > 0], default=0),
            },
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "overall_success": r.overall_success,
                    "python_success": r.python_success,
                    "rust_success": r.rust_success,
                    "outputs_match": r.outputs_match,
                    "numerical_errors": r.numerical_errors,
                    "performance_comparison": r.performance_comparison,
                    "execution_times": r.execution_times,
                    "error_messages": r.error_messages,
                }
                for r in results
            ],
            "environment_info": {
                "python": self.python_runner.get_version_info(),
                "rust": self.rust_runner.get_version_info() if self.rust_runner else {"error": "Not available"},
            },
            "config": asdict(self.config),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {output_path}")
        return report

class EdgeCaseGenerator:
    """Generate edge cases and stress test data."""
    
    @staticmethod
    def generate_edge_case_inputs() -> List[Dict[str, Any]]:
        """Generate edge case inputs for testing."""
        edge_cases = []
        
        # Empty inputs
        edge_cases.append({
            "name": "empty_tokens",
            "tokens": [],
            "description": "Empty token sequence"
        })
        
        # Single token
        edge_cases.append({
            "name": "single_token",
            "tokens": [1],
            "description": "Single token input"
        })
        
        # Very long sequence
        edge_cases.append({
            "name": "long_sequence",
            "tokens": list(range(1, 1025)),  # 1024 tokens
            "description": "Very long token sequence"
        })
        
        # Repeated tokens
        edge_cases.append({
            "name": "repeated_tokens",
            "tokens": [42] * 100,
            "description": "Repeated token sequence"
        })
        
        # Large token values
        edge_cases.append({
            "name": "large_token_values",
            "tokens": [65535, 65534, 65533, 65532],
            "description": "Large token values near vocabulary limit"
        })
        
        # Zero and negative values (if applicable)
        edge_cases.append({
            "name": "boundary_values",
            "tokens": [0, 1, 2, 3],
            "description": "Boundary token values"
        })
        
        return edge_cases
    
    @staticmethod
    def generate_stress_test_configs() -> List[Dict[str, Any]]:
        """Generate configurations for stress testing."""
        stress_configs = []
        
        # Minimal model
        stress_configs.append({
            "name": "minimal_model",
            "config": {
                "dim": 64,
                "n_layers": 1,
                "n_heads": 1,
                "n_kv_heads": 1,
                "vocab_size": 100,
                "ffn_dim": 128,
            },
            "description": "Minimal model configuration"
        })
        
        # Large model (within memory constraints)
        stress_configs.append({
            "name": "large_model",
            "config": {
                "dim": 512,
                "n_layers": 8,
                "n_heads": 16,
                "n_kv_heads": 8,
                "vocab_size": 10000,
                "ffn_dim": 1024,
            },
            "description": "Large model configuration"
        })
        
        # Unusual dimensions
        stress_configs.append({
            "name": "unusual_dimensions",
            "config": {
                "dim": 127,  # Prime number
                "n_layers": 3,
                "n_heads": 7,  # Prime number
                "n_kv_heads": 3,
                "vocab_size": 997,  # Prime number
                "ffn_dim": 251,  # Prime number
            },
            "description": "Model with unusual prime dimensions"
        })
        
        return stress_configs
    
    @staticmethod
    def generate_numerical_edge_cases() -> List[Dict[str, Any]]:
        """Generate numerical edge cases for quantization testing."""
        edge_cases = []
        
        # All zeros
        edge_cases.append({
            "name": "all_zeros",
            "weights": np.zeros((32, 64)).tolist(),
            "description": "All zero weights"
        })
        
        # All ones
        edge_cases.append({
            "name": "all_ones",
            "weights": np.ones((32, 64)).tolist(),
            "description": "All one weights"
        })
        
        # Very small values
        edge_cases.append({
            "name": "very_small_values",
            "weights": (np.random.randn(32, 64) * 1e-6).tolist(),
            "description": "Very small weight values"
        })
        
        # Very large values
        edge_cases.append({
            "name": "very_large_values",
            "weights": (np.random.randn(32, 64) * 1e6).tolist(),
            "description": "Very large weight values"
        })
        
        # Mixed positive/negative
        edge_cases.append({
            "name": "mixed_signs",
            "weights": np.where(
                np.random.randn(32, 64) > 0,
                np.random.randn(32, 64) * 100,
                np.random.randn(32, 64) * -100
            ).tolist(),
            "description": "Mixed positive and negative large values"
        })
        
        # Sparse weights (mostly zeros)
        sparse_weights = np.zeros((32, 64))
        sparse_indices = np.random.choice(32*64, size=32, replace=False)
        sparse_weights.flat[sparse_indices] = np.random.randn(32)
        edge_cases.append({
            "name": "sparse_weights",
            "weights": sparse_weights.tolist(),
            "description": "Sparse weight matrix"
        })
        
        return edge_cases

class TestCaseGenerator:
    """Generate test cases for cross-validation."""
    
    @staticmethod
    def generate_model_forward_tests() -> List[TestCase]:
        """Generate model forward pass test cases."""
        test_cases = []
        
        # Small model test
        small_config = {
            "dim": 128,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "vocab_size": 1000,
            "ffn_dim": 256,
        }
        
        model_inputs = {
            "token_values": [1, 2, 3, 4, 5],
            "token_lengths": [5],
            "start_pos": [0],
            "cache_len": 16,
        }
        
        test_cases.append(TestCase(
            name="model_forward_small",
            inputs={
                "config": small_config,
                "model_inputs": model_inputs,
            },
            metadata={"description": "Small model forward pass test"}
        ))
        
        # Medium model test
        medium_config = {
            "dim": 256,
            "n_layers": 4,
            "n_heads": 8,
            "n_kv_heads": 4,
            "vocab_size": 5000,
            "ffn_dim": 512,
        }
        
        test_cases.append(TestCase(
            name="model_forward_medium",
            inputs={
                "config": medium_config,
                "model_inputs": model_inputs,
            },
            metadata={"description": "Medium model forward pass test"}
        ))
        
        return test_cases
    
    @staticmethod
    def generate_quantization_tests() -> List[TestCase]:
        """Generate quantization test cases."""
        test_cases = []
        
        # Basic quantization test
        np.random.seed(42)
        weights = np.random.randn(32, 64).tolist()
        
        test_cases.append(TestCase(
            name="quantization_basic",
            inputs={"weights": weights},
            metadata={"description": "Basic quantization test"}
        ))
        
        # Large quantization test
        weights_large = np.random.randn(128, 256).tolist()
        
        test_cases.append(TestCase(
            name="quantization_large",
            inputs={"weights": weights_large},
            metadata={"description": "Large quantization test"}
        ))
        
        return test_cases
    
    @staticmethod
    def generate_inference_tests() -> List[TestCase]:
        """Generate inference test cases."""
        test_cases = []
        
        # Basic inference test
        test_cases.append(TestCase(
            name="inference_basic",
            inputs={
                "prompt": "Hello world",
                "max_tokens": 10,
                "temperature": 0.8,
            },
            metadata={"description": "Basic inference test"}
        ))
        
        return test_cases
    
    @staticmethod
    def generate_edge_case_tests() -> List[TestCase]:
        """Generate edge case test cases."""
        test_cases = []
        edge_generator = EdgeCaseGenerator()
        
        # Model forward edge cases
        edge_inputs = edge_generator.generate_edge_case_inputs()
        stress_configs = edge_generator.generate_stress_test_configs()
        
        for config_data in stress_configs:
            for input_data in edge_inputs:
                if len(input_data["tokens"]) > 0:  # Skip empty tokens for model tests
                    model_inputs = {
                        "token_values": input_data["tokens"][:min(len(input_data["tokens"]), 64)],  # Limit length
                        "token_lengths": [min(len(input_data["tokens"]), 64)],
                        "start_pos": [0],
                        "cache_len": 128,
                    }
                    
                    test_cases.append(TestCase(
                        name=f"model_forward_edge_{config_data['name']}_{input_data['name']}",
                        inputs={
                            "config": config_data["config"],
                            "model_inputs": model_inputs,
                        },
                        metadata={
                            "description": f"Edge case: {config_data['description']} with {input_data['description']}",
                            "category": "edge_case"
                        }
                    ))
        
        # Quantization edge cases
        numerical_edge_cases = edge_generator.generate_numerical_edge_cases()
        for edge_case in numerical_edge_cases:
            test_cases.append(TestCase(
                name=f"quantization_edge_{edge_case['name']}",
                inputs={"weights": edge_case["weights"]},
                metadata={
                    "description": f"Quantization edge case: {edge_case['description']}",
                    "category": "edge_case"
                }
            ))
        
        # Inference edge cases
        inference_edge_cases = [
            {"prompt": "", "description": "Empty prompt"},
            {"prompt": "A" * 1000, "description": "Very long prompt"},
            {"prompt": "Hello\n\nWorld\t\r", "description": "Prompt with special characters"},
            {"prompt": "🚀🌟💫", "description": "Prompt with Unicode emojis"},
        ]
        
        for edge_case in inference_edge_cases:
            test_cases.append(TestCase(
                name=f"inference_edge_{edge_case['description'].lower().replace(' ', '_')}",
                inputs={
                    "prompt": edge_case["prompt"],
                    "max_tokens": 5,
                    "temperature": 0.0,  # Deterministic for edge cases
                },
                metadata={
                    "description": f"Inference edge case: {edge_case['description']}",
                    "category": "edge_case"
                }
            ))
        
        return test_cases
    
    @staticmethod
    def generate_stress_tests() -> List[TestCase]:
        """Generate stress test cases."""
        test_cases = []
        
        # High-load inference tests
        stress_prompts = [
            "Write a detailed essay about artificial intelligence and its impact on society.",
            "Explain quantum computing in simple terms.",
            "Create a story about a robot learning to understand human emotions.",
        ]
        
        for i, prompt in enumerate(stress_prompts):
            test_cases.append(TestCase(
                name=f"stress_inference_long_{i}",
                inputs={
                    "prompt": prompt,
                    "max_tokens": 100,
                    "temperature": 0.8,
                },
                metadata={
                    "description": f"Stress test: Long inference with prompt {i}",
                    "category": "stress_test"
                }
            ))
        
        # Batch processing simulation
        batch_prompts = ["Hello", "World", "Test", "Batch", "Processing"]
        test_cases.append(TestCase(
            name="stress_batch_processing",
            inputs={
                "prompts": batch_prompts,
                "max_tokens": 10,
                "temperature": 0.5,
            },
            metadata={
                "description": "Stress test: Batch processing simulation",
                "category": "stress_test"
            }
        ))
        
        return test_cases
    
    @classmethod
    def generate_all_tests(cls) -> List[TestCase]:
        """Generate all test cases including edge cases and stress tests."""
        all_tests = []
        all_tests.extend(cls.generate_model_forward_tests())
        all_tests.extend(cls.generate_quantization_tests())
        all_tests.extend(cls.generate_inference_tests())
        all_tests.extend(cls.generate_edge_case_tests())
        all_tests.extend(cls.generate_stress_tests())
        return all_tests

def create_default_config() -> ValidationConfig:
    """Create default validation configuration."""
    return ValidationConfig(
        numerical_tolerance={
            "rtol": 1e-4,
            "atol": 1e-5,
            # Token-level comparison tolerances
            "min_token_match_ratio": 0.95,
            "max_token_differences": 5,
            # Logits comparison tolerances
            "logits_rtol": 1e-3,
            "logits_atol": 1e-4,
            "min_logits_close_ratio": 0.99,
            "max_logits_abs_diff": 0.1,
        },
        performance_tolerance={
            "max_regression_percent": 5.0,
        },
        timeout_seconds=300,
        max_retries=3,
    )

# Convenience functions for common use cases
def run_cross_validation(rust_binary_path: Optional[Path] = None, 
                        output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run complete cross-validation suite."""
    config = create_default_config()
    validator = CrossLanguageValidator(config)
    
    # Set up Rust runner if binary is available
    if rust_binary_path or Path("target/release/bitnet-cli").exists():
        validator.set_rust_runner(rust_binary_path)
    
    # Generate test cases
    test_cases = TestCaseGenerator.generate_all_tests()
    
    # Run validation
    results = validator.validate_test_suite(test_cases)
    
    # Generate report
    if output_path is None:
        output_path = Path("cross_validation_report.json")
    
    report = validator.generate_report(results, output_path)
    
    # Print summary
    summary = report["summary"]
    print(f"\nCross-Validation Summary:")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    
    if summary['failed_tests'] > 0:
        print(f"Python failures: {summary['python_failures']}")
        print(f"Rust failures: {summary['rust_failures']}")
        print(f"Output mismatches: {summary['output_mismatches']}")
    
    perf_summary = report["performance_summary"]
    if perf_summary["average_speedup"] > 0:
        print(f"Average speedup: {perf_summary['average_speedup']:.2f}x")
    
    return report

if __name__ == "__main__":
    # Run cross-validation when script is executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Run cross-language validation")
    parser.add_argument("--rust-binary", type=Path, help="Path to Rust binary")
    parser.add_argument("--output", type=Path, help="Output report path")
    
    args = parser.parse_args()
    
    report = run_cross_validation(args.rust_binary, args.output)
    
    # Exit with error code if tests failed
    if report["summary"]["failed_tests"] > 0:
        exit(1)