import contextlib
import io
import logging
import os
import sys
import tempfile
import unittest

import numpy as np

from giatools.typing import (
    Any,
    Literal,
    Tuple,
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


def verify_metadata(testcase: unittest.TestCase, metadata: dict, **expected: Any):
    """
    Verify that the metadata is present and correct.
    """
    testcase.assertIsInstance(metadata, dict)
    for key, value in expected.items():
        if value is None:
            testcase.assertNotIn(key, metadata)
        else:
            testcase.assertIn(key, metadata)
            if isinstance(value, tuple):
                np.testing.assert_array_almost_equal(metadata[key], value, err_msg=f'Validation failed for "{key}"')
            else:
                testcase.assertEqual(metadata[key], value)


def random_io_test(shape: Tuple, dtype: np.dtype, ext: str):
    def decorator(test_impl):
        def wrapper(self):
            with tempfile.TemporaryDirectory() as temp_path:

                # Create random image data
                np.random.seed(0)
                data = np.random.rand(*shape)
                if not np.issubdtype(dtype, np.floating):
                    data = (data * np.iinfo(dtype).max).astype(dtype)

                # Supply a temporary file to write the image to
                filepath = os.path.join(temp_path, f'test.{ext}')

                # Run the test
                test_impl(self, filepath, data)

        return wrapper
    return decorator


def _select_python_version(op: Literal['min', 'max']):
    def create_decorator(major: int, minor: int):
        def decorator(test_impl):
            def wrapper(self):
                if op == 'min':
                    if sys.version_info < (major, minor):
                        self.skipTest(f'Requires Python {major}.{minor} or later')
                    else:
                        test_impl(self)
                elif op == 'max':
                    if sys.version_info > (major, minor):
                        self.skipTest(f'Requires Python {major}.{minor} or earlier')
                    else:
                        test_impl(self)
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
            for ext in extensions:
                with self.subTest(extension=ext):
                    kwargs = dict(kwargs)
                    kwargs[name] = f'{prefix}.{ext}'
                    test_func(self, *args, **kwargs)
        return wrapper
    return decorator
