import contextlib
import io


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
