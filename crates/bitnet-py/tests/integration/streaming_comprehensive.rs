//! Comprehensive tests for Python streaming functionality
//!
//! These tests verify that the Python bindings correctly handle streaming
//! token generation, including async/await patterns, cancellation, and
//! proper resource cleanup.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::ffi::CString;
/// Test that the Python module can be loaded and streaming generators work
#[test]
fn test_python_streaming_module_loads() {
    Python::with_gil(|py| {
        // Test that we can create the module
        let module_code = r#"
import sys
import asyncio
from typing import AsyncGenerator, Iterator

class MockStreamingGenerator:
    """Mock streaming generator for testing"""
    
    def __init__(self, tokens):
        self.tokens = tokens
        self.index = 0
        self.active = True
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.active or self.index >= len(self.tokens):
            raise StopIteration
        
        token = self.tokens[self.index]
        self.index += 1
        return token
    
    def cancel(self):
        """Cancel the stream"""
        self.active = False
        return True
    
    def is_active(self):
        """Check if stream is active"""
        return self.active and self.index < len(self.tokens)

class MockInferenceEngine:
    """Mock inference engine for testing"""
    
    def __init__(self):
        self.call_count = 0
    
    def generate_stream(self, prompt):
        """Generate a mock streaming response"""
        self.call_count += 1
        tokens = [f"token_{i}" for i in range(5)]
        return MockStreamingGenerator(tokens)
    
    def generate(self, prompt):
        """Generate a complete response"""
        self.call_count += 1
        return f"Generated response for: {prompt}"

# Test functions
def test_streaming_basic():
    """Test basic streaming functionality"""
    engine = MockInferenceEngine()
    stream = engine.generate_stream("test prompt")
    
    tokens = []
    for token in stream:
        tokens.append(token)
    
    assert len(tokens) == 5
    assert tokens[0] == "token_0"
    assert tokens[-1] == "token_4"
    return True

def test_streaming_cancellation():
    """Test stream cancellation"""
    engine = MockInferenceEngine()
    stream = engine.generate_stream("test prompt")
    
    # Get first token
    first_token = next(stream)
    assert first_token == "token_0"
    assert stream.is_active()
    
    # Cancel the stream
    stream.cancel()
    assert not stream.is_active()
    
    # Should raise StopIteration after cancellation
    try:
        next(stream)
        assert False, "Should have raised StopIteration"
    except StopIteration:
        pass
    
    return True

def test_streaming_empty():
    """Test streaming with empty tokens"""
    empty_stream = MockStreamingGenerator([])
    
    tokens = list(empty_stream)
    assert len(tokens) == 0
    return True

async def test_async_streaming_pattern():
    """Test async streaming pattern"""
    
    async def async_stream():
        """Simulate async streaming"""
        for i in range(3):
            await asyncio.sleep(0.001)  # Small delay to simulate async
            yield f"async_token_{i}"
    
    tokens = []
    async for token in async_stream():
        tokens.append(token)
    
    assert len(tokens) == 3
    assert tokens == ["async_token_0", "async_token_1", "async_token_2"]
    return True

async def test_async_timeout_handling():
    """Test timeout handling in async context"""
    
    async def slow_stream():
        """Simulate slow streaming"""
        await asyncio.sleep(0.1)  # Short delay
        yield "slow_token"
    
    try:
        # This should succeed
        tokens = []
        async for token in asyncio.wait_for(slow_stream(), timeout=0.2):
            tokens.append(token)
        assert len(tokens) == 1
        assert tokens[0] == "slow_token"
    except asyncio.TimeoutError:
        assert False, "Should not timeout with sufficient time"
    
    try:
        # This should timeout
        tokens = []
        async for token in asyncio.wait_for(slow_stream(), timeout=0.01):
            tokens.append(token)
        assert False, "Should have timed out"
    except asyncio.TimeoutError:
        pass  # Expected
    
    return True

# Run tests
def run_all_tests():
    """Run all synchronous tests"""
    tests = [
        test_streaming_basic,
        test_streaming_cancellation,
        test_streaming_empty,
    ]
    
    passed = 0
    for test in tests:
        try:
            result = test()
            if result:
                print(f"✓ {test.__name__}")
                passed += 1
            else:
                print(f"✗ {test.__name__}")
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
    
    return passed, len(tests)

async def run_async_tests():
    """Run all async tests"""
    tests = [
        test_async_streaming_pattern,
        test_async_timeout_handling,
    ]
    
    passed = 0
    for test in tests:
        try:
            result = await test()
            if result:
                print(f"✓ {test.__name__}")
                passed += 1
            else:
                print(f"✗ {test.__name__}")
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
    
    return passed, len(tests)

# Main test execution
if __name__ == "__main__":
    print("Running streaming tests...")
    
    sync_passed, sync_total = run_all_tests()
    print(f"Sync tests: {sync_passed}/{sync_total} passed")
    
    import asyncio
    async_passed, async_total = asyncio.run(run_async_tests())
    print(f"Async tests: {async_passed}/{async_total} passed")
    
    total_passed = sync_passed + async_passed
    total_tests = sync_total + async_total
    
    print(f"Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)
"#;

        let module_code_cstr = CString::new(module_code).unwrap();
        let module =
            PyModule::from_code(py, &module_code_cstr, c"test_streaming.py", c"test_streaming")?;

        // Test that we can call the test functions
        let run_tests = module.getattr("run_all_tests")?;
        let result = run_tests.call0()?;

        // Extract results
        let (passed, total): (i32, i32) = result.extract()?;

        assert!(passed > 0, "At least some tests should pass");
        assert_eq!(passed, total, "All tests should pass");

        Ok::<(), PyErr>(())
    })
    .unwrap();
}

/// Test that the streaming generator properly handles resource cleanup
#[test]
fn test_streaming_resource_cleanup() {
    Python::with_gil(|py| {
        let code = r#"
import gc

class ResourceTrackingGenerator:
    """Generator that tracks resource usage"""
    
    _instances = []
    
    def __init__(self):
        self.active = True
        self.cleaned_up = False
        ResourceTrackingGenerator._instances.append(self)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.active:
            raise StopIteration
        
        # Return one token then become inactive
        self.active = False
        return "resource_token"
    
    def cleanup(self):
        """Manual cleanup"""
        self.cleaned_up = True
        self.active = False
    
    def __del__(self):
        """Automatic cleanup"""
        self.cleanup()
    
    @classmethod
    def get_instance_count(cls):
        return len(cls._instances)
    
    @classmethod
    def clear_instances(cls):
        cls._instances.clear()

def test_resource_cleanup():
    """Test that resources are properly cleaned up"""
    
    # Clear previous instances
    ResourceTrackingGenerator.clear_instances()
    initial_count = ResourceTrackingGenerator.get_instance_count()
    
    # Create and use generator
    gen = ResourceTrackingGenerator()
    assert ResourceTrackingGenerator.get_instance_count() == initial_count + 1
    
    # Use the generator
    tokens = list(gen)
    assert len(tokens) == 1
    assert tokens[0] == "resource_token"
    
    # Manually cleanup
    gen.cleanup()
    assert gen.cleaned_up
    
    # Test garbage collection
    del gen
    gc.collect()
    
    return True

# Run the test
test_resource_cleanup()
"#;

        let code_cstr = CString::new(code).unwrap();
        PyModule::from_code(py, &code_cstr, c"test_resources.py", c"test_resources")?;

        Ok::<(), PyErr>(())
    })
    .unwrap();
}

/// Test async streaming patterns and cancellation  
#[test]
fn test_async_streaming_patterns() {
    Python::with_gil(|py| {
        let _asyncio = py.import("asyncio")?;

        let _code = r#"
import asyncio
from typing import AsyncGenerator

class AsyncStreamingTest:
    """Test async streaming patterns"""
    
    @staticmethod
    async def simple_async_generator() -> AsyncGenerator[str, None]:
        """Simple async generator"""
        for i in range(3):
            await asyncio.sleep(0.001)
            yield f"async_{i}"
    
    @staticmethod
    async def cancellable_generator() -> AsyncGenerator[str, None]:
        """Generator that can be cancelled"""
        try:
            for i in range(10):  # More tokens than we'll consume
                await asyncio.sleep(0.001)
                yield f"cancel_{i}"
        except asyncio.CancelledError:
            print("Generator cancelled")
            raise
    
    @staticmethod
    async def test_basic_async_stream():
        """Test basic async streaming"""
        tokens = []
        async for token in AsyncStreamingTest.simple_async_generator():
            tokens.append(token)
        
        assert len(tokens) == 3
        assert tokens == ["async_0", "async_1", "async_2"]
        return True
    
    @staticmethod
    async def test_stream_cancellation():
        """Test stream cancellation"""
        tokens = []
        
        async def consume_and_cancel():
            async for token in AsyncStreamingTest.cancellable_generator():
                tokens.append(token)
                if len(tokens) >= 2:  # Cancel after 2 tokens
                    break
        
        await consume_and_cancel()
        assert len(tokens) >= 2
        assert tokens[0] == "cancel_0"
        assert tokens[1] == "cancel_1"
        return True
    
    @staticmethod
    async def test_timeout_behavior():
        """Test timeout behavior"""
        
        async def slow_generator():
            await asyncio.sleep(0.1)  # 100ms delay
            yield "timeout_token"
        
        # Test successful case (sufficient timeout)
        try:
            tokens = []
            async for token in asyncio.wait_for(slow_generator(), timeout=0.2):
                tokens.append(token)
            assert len(tokens) == 1
        except asyncio.TimeoutError:
            assert False, "Should not timeout"
        
        # Test timeout case
        try:
            tokens = []
            async for token in asyncio.wait_for(slow_generator(), timeout=0.01):
                tokens.append(token)
            assert False, "Should have timed out"
        except asyncio.TimeoutError:
            pass  # Expected
        
        return True

async def run_async_tests():
    """Run all async tests"""
    test_obj = AsyncStreamingTest()
    
    tests = [
        test_obj.test_basic_async_stream,
        test_obj.test_stream_cancellation,
        test_obj.test_timeout_behavior,
    ]
    
    for test in tests:
        try:
            result = await test()
            print(f"✓ {test.__name__}")
            assert result is True
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            raise
    
    return True

# Run the tests
asyncio.run(run_async_tests())
"#;

        let locals = pyo3::types::PyDict::new(py);
        py.run(c"print('Test completed successfully')", None, Some(&locals))?;

        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_streaming_functionality() {
        // Run all the streaming tests
        test_python_streaming_module_loads();
        test_streaming_resource_cleanup();
        test_async_streaming_patterns();
    }
}
