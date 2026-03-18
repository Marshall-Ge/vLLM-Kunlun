import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import pytest
import torch


def test_import():
    """Test that the module can be imported successfully."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    assert TorchCompileWrapperWithCustomDispatcher is not None


def test_basic_instantiation():
    """Test basic wrapper instantiation with mocked dependencies."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    # Create a concrete implementation
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    # Mock all the dependencies
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Verify basic attributes exist
                    assert hasattr(wrapper, 'vllm_config')
                    assert hasattr(wrapper, 'compiled_callable')
                    assert hasattr(wrapper, 'original_code_object')
                    assert hasattr(wrapper, 'compiled_codes')
                    assert isinstance(wrapper.compiled_codes, list)


def test_forward_call():
    """Test that the forward method can be called."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Test calling the wrapper
                    input_tensor = torch.tensor([1.0, 2.0, 3.0])
                    result = wrapper(input_tensor)
                    
                    expected = input_tensor * 2
                    assert torch.allclose(result, expected)


def test_custom_callable():
    """Test wrapper with custom compiled callable."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    custom_func = Mock(return_value=torch.tensor([5.0]))
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                wrapper = TestWrapper(
                    compiled_callable=custom_func,
                    compilation_level=0
                )
                
                # Verify custom callable is used
                assert wrapper.compiled_callable is custom_func
                
                # Call should use custom callable
                result = wrapper(torch.tensor([1.0]))
                assert custom_func.called


def test_bytecode_hook_basic():
    """Test that bytecode hook can be called without errors."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    from types import CodeType
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    mock_config.compilation_config.local_cache_dir = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    wrapper = TestWrapper(compilation_level=0)
                    
                    # Test with wrong code object (should be ignored)
                    wrong_code = MagicMock(spec=CodeType)
                    new_code = MagicMock(spec=CodeType)
                    
                    initial_count = len(wrapper.compiled_codes)
                    wrapper.bytecode_hook(wrong_code, new_code)
                    
                    # Should not add anything
                    assert len(wrapper.compiled_codes) == initial_count


def test_use_custom_dispatcher_flag():
    """Test that use_custom_dispatcher flag is set based on compilation_level."""
    from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
    
    class TestWrapper(TorchCompileWrapperWithCustomDispatcher):
        def forward(self, x):
            return x * 2
    
    mock_config = MagicMock()
    mock_config.compilation_config.init_backend.return_value = "eager"
    mock_config.compilation_config.inductor_compile_config = None
    
    with patch('vllm.config.get_current_vllm_config', return_value=mock_config):
        with patch('vllm.config.CompilationLevel') as mock_level:
            mock_level.DYNAMO_ONCE = 1
            with patch('torch.compile', side_effect=lambda func, **kwargs: func):
                with patch('torch._dynamo.convert_frame.register_bytecode_hook'):
                    # Test with low level
                    wrapper_low = TestWrapper(compilation_level=0)
                    assert wrapper_low.use_custom_dispatcher is False
                    
                    # Test with high level
                    wrapper_high = TestWrapper(compilation_level=2)
                    assert wrapper_high.use_custom_dispatcher is True


def _load_kunlun_ops_module():
    module_name = "_kunlun_ops_test_module"
    module_path = (
        Path(__file__).resolve().parents[2] / "vllm_kunlun" / "ops" / "_kunlun_ops.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_allocate_temp_tensors_uses_workspace_manager():
    """Test workspace-backed scratch tensor allocation."""
    with patch.dict(
        sys.modules,
        {
            "cocopod": MagicMock(),
            "xspeedgate_ops": MagicMock(),
            "kunlun_ops": MagicMock(),
        },
    ):
        module = _load_kunlun_ops_module()

    calls = []

    class FakeWorkspaceManager:
        def get_simultaneous(self, *args):
            calls.append(args)
            return [
                torch.empty(shape, dtype=dtype)
                for shape, dtype in args
            ]

    with patch.object(module, "current_workspace_manager", return_value=FakeWorkspaceManager()):
        zeros_tensor, ones_tensor = module._allocate_temp_tensors(
            torch.device("cpu"),
            (
                ((2, 3), torch.float32, "zeros"),
                ((2,), torch.int32, "ones"),
            ),
        )

    assert len(calls) == 1
    assert calls[0] == (((2, 3), torch.float32), ((2,), torch.int32))
    assert torch.equal(zeros_tensor, torch.zeros((2, 3), dtype=torch.float32))
    assert torch.equal(ones_tensor, torch.ones((2,), dtype=torch.int32))


def test_allocate_temp_tensors_falls_back_when_workspace_unavailable():
    """Test scratch allocation fallback without a workspace manager."""
    with patch.dict(
        sys.modules,
        {
            "cocopod": MagicMock(),
            "xspeedgate_ops": MagicMock(),
            "kunlun_ops": MagicMock(),
        },
    ):
        module = _load_kunlun_ops_module()

    with patch.object(module, "current_workspace_manager", side_effect=AssertionError("not initialized")):
        zeros_tensor, empty_tensor = module._allocate_temp_tensors(
            torch.device("cpu"),
            (
                ((4,), torch.int32, "zeros"),
                ((2, 2), torch.float16, "empty"),
            ),
        )

    assert torch.equal(zeros_tensor, torch.zeros((4,), dtype=torch.int32))
    assert empty_tensor.shape == (2, 2)
    assert empty_tensor.dtype == torch.float16


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
