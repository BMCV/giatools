import contextlib
import io
import unittest

import numpy as np

from giatools.typing import Any


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
                np.testing.assert_array_almost_equal(metadata[key], value)
            else:
                testcase.assertEqual(metadata[key], value)
