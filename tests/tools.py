import contextlib
import io
import itertools
import logging
import os
import sys
import tempfile
import unittest

import attrs
import numpy as np

import giatools.metadata
from giatools.typing import (
    Any,
    Literal,
    Tuple,
    Union,
)


class CaptureStderr:

    def __init__(self):
        self.stdout_buf = io.StringIO()

    def __enter__(self):
        self.redirect = contextlib.redirect_stderr(self.stdout_buf)
        self.redirect.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.redirect.__exit__(exc_type, exc_value, traceback)

    def __str__(self):
        return self.stdout_buf.getvalue()


def validate_metadata(testcase: unittest.TestCase, actual: Union[dict, giatools.metadata.Metadata], **expected: Any):
    """
    Verify that the `actual` metadata is identical to the `expected` metadata.
    """
    # Normalize the actual metadata (fill missing fields with None)
    if isinstance(actual, dict):
        actual = giatools.metadata.Metadata(**actual)
    actual = attrs.asdict(actual)

    # Normalize the expected metadata (fill missing fields with None)
    expected_metadata = giatools.metadata.Metadata(**expected)
    expected = attrs.asdict(expected_metadata)

    # Compare
    testcase.assertEqual(actual, expected)


def random_io_test(shape: Tuple, dtype: np.dtype, ext: str):
    def decorator(test_impl):
        def wrapper(self):
            with tempfile.TemporaryDirectory() as temp_path:

                # Create random image data
                np.random.seed(0)
                data = np.random.rand(*shape)
                if np.issubdtype(dtype, np.floating):
                    pass
                elif np.issubdtype(dtype, np.integer):
                    data = (data * np.iinfo(dtype).max).astype(dtype)
                elif np.issubdtype(dtype, bool):
                    data = (data > 0.5).round().astype(bool)
                else:
                    assert False, f'Unsupported dtype {dtype}'

                # Supply a temporary file to write the image to
                filepath = os.path.join(temp_path, f'test.{ext}')

                # Run the test
                test_impl(self, filepath, data)

        return wrapper
    return decorator


def _select_python_version(op: Literal['min', 'max']):
    def create_decorator(major: int, minor: int):
        def decorator(test_impl):
            def wrapper(self, *args, **kwargs):
                if op == 'min':
                    if sys.version_info < (major, minor):
                        self.skipTest(f'Requires Python {major}.{minor} or later')
                    else:
                        test_impl(self, *args, **kwargs)
                elif op == 'max':
                    if sys.version_info > (major, minor):
                        self.skipTest(f'Requires Python {major}.{minor} or earlier')
                    else:
                        test_impl(self, *args, **kwargs)
                else:
                    raise ValueError(f'Unknown operation "{op}"')
            return wrapper
        return decorator
    return create_decorator


minimum_python_version = _select_python_version('min')
maximum_python_version = _select_python_version('max')


def without_logging(test_impl):
    """
    Disable logging for the duration of the test.
    """
    def wrapper(self):
        logger = logging.getLogger()
        previous_level = logger.level
        logger.setLevel(logging.CRITICAL + 1)
        try:
            test_impl(self)
        finally:
            logger.setLevel(previous_level)
    return wrapper


def mock_array(*shape, name: str = 'array'):
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            array = unittest.mock.MagicMock(shape=shape, ndim=len(shape))
            kwargs = dict(kwargs)
            kwargs[name] = array
            return test_func(*args, **kwargs)
        return wrapper
    return decorator


def filenames(*extensions, prefix: str = 'filename', name: str = 'filename'):
    def decorator(test_func):
        def wrapper(self, *args, **kwargs):
            with tempfile.TemporaryDirectory() as temp_path:
                for ext in extensions:
                    with self.subTest(extension=ext):
                        kwargs = dict(kwargs)
                        kwargs[name] = os.path.join(temp_path, f'{prefix}.{ext}')
                        test_func(self, *args, **kwargs)
        return wrapper
    return decorator


def permute_axes(axes: str, name='axes'):
    permutations = list(''.join(axis) for axis in itertools.permutations(axes, len(axes)))

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            for axis in permutations:
                kwargs = dict(kwargs)
                kwargs[name] = axis
                with self.subTest(**dict({name: axis})):
                    func(self, *args, **kwargs)
        return wrapper

    return decorator
