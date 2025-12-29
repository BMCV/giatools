"""
Unit tests for the `giatools.image_processor` module.
"""

import unittest
import unittest.mock

import giatools.image_processor


class ImageProcessorTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.ProcessorIteration = unittest.mock.patch(
            'giatools.image_processor.ProcessorIteration'
        ).start()

        self.addCleanup(unittest.mock.patch.stopall)


@unittest.mock.patch('giatools.image_processor.ImageProcessor.create_output_image')
class ImageProcessor__process(ImageProcessorTestCase):

    def setUp(self):
        super().setUp()
        self.image1 = unittest.mock.MagicMock(axes='YX')
        self.image_processor = giatools.image_processor.ImageProcessor(self.image1)

    def test__no_inputs(self, mock_create_output_image):
        with self.assertRaises(ValueError):
            giatools.image_processor.ImageProcessor()

    def test__inconsistent_axes(self, mock_create_output_image):
        with self.assertRaises(ValueError):
            giatools.image_processor.ImageProcessor(
                self.image1,
                unittest.mock.MagicMock(axes='XYC'),
            )

    def test__inconsistent_shape(self, mock_create_output_image):
        with self.assertRaises(ValueError):
            giatools.image_processor.ImageProcessor(
                unittest.mock.MagicMock(axes='XY', shape=(100, 80)),
                unittest.mock.MagicMock(axes='XY', shape=(80, 100)),
            )

    def test__errors(self, mock_create_output_image):
        for error in (RuntimeError, ValueError):
            with self.subTest(error=error):
                self.image1.iterate_jointly.side_effect = error()
                with self.assertRaises(error):
                    next(iter(self.image_processor.process('YX')))


class ImageProcessor__create_output_image(ImageProcessorTestCase):

    def setUp(self):
        super().setUp()
        self.image1 = unittest.mock.MagicMock(axes='YX')
        self.image_processor = giatools.image_processor.ImageProcessor(self.image1)

    def test(self):
        for key in ('output_key', 42, ('a', 'b'), None):
            with self.subTest(key=key):
                self.image_processor.create_output_image(key, dtype='uint8')

    def test__value_error(self):
        self.image_processor.create_output_image('output_key', dtype='uint8')
        for dtype in ('uint8', 'uint16'):
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError):
                    self.image_processor.create_output_image('output_key', dtype=dtype)


class ProcessorIteration(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.processor = unittest.mock.MagicMock()
        self.output_slice = unittest.mock.MagicMock()
        self.output_dtype_hints = dict()
        self.input_section1 = unittest.mock.MagicMock()
        self.input_section2 = unittest.mock.MagicMock()
        self.input_section3 = unittest.mock.MagicMock()
        self.processor_iteration = giatools.image_processor.ProcessorIteration(
            input_sections={
                0: self.input_section1,
                1: self.input_section2,
                'input_key': self.input_section3,
            },
            output_slice=self.output_slice,
            output_dtype_hints=self.output_dtype_hints,
            processor=self.processor,
            joint_axes='YX',
        )

    def test__getitem__kwarg(self):
        self.assertIs(self.processor_iteration['input_key'], self.input_section3)

    def test__getitem__spurious_kwarg(self):
        with self.assertRaisesRegex(KeyError, f'No input image with key "spurious_key".'):
            self.processor_iteration['spurious_key']

    def test__getitem__posarg(self):
        self.assertIs(self.processor_iteration[0], self.input_section1)
        self.assertIs(self.processor_iteration[1], self.input_section2)
        self.assertIs(self.processor_iteration[-1], self.input_section2)
        self.assertIs(self.processor_iteration[-2], self.input_section1)

    def test__getitem__invalid_posarg(self):
        for pos in (2, 3, -3, -4):
            with self.subTest(pos=pos):
                with self.assertRaisesRegex(IndexError, f'No input image at position {pos}.'):
                    self.processor_iteration[pos]

    @unittest.mock.patch('giatools.image_processor._Image')
    def test__setitem__(self, mock_image):
        output_data = unittest.mock.MagicMock()
        self.processor_iteration['output_key'] = output_data
        mock_image.assert_called_once_with(data=output_data, axes='YX')
        self.processor.create_output_image.assert_called_once_with(
            'output_key',
            mock_image.return_value.reorder_axes_like().data.dtype,
        )

    @unittest.mock.patch('giatools.image_processor._Image')
    def test__setitem__repeated(self, mock_image):
        self.processor.outputs = {'output_key': unittest.mock.MagicMock()}
        self.processor_iteration['output_key'] = unittest.mock.MagicMock()
        self.processor.create_output_image.assert_not_called()

    @unittest.mock.patch('giatools.image_processor.apply_output_dtype_hint')
    @unittest.mock.patch('giatools.image_processor._Image')
    def test__setitem__with_output_dtype_hints(self, mock_image, mock_apply_output_dtype_hint):
        output_dtype_hint = unittest.mock.Mock()
        self.output_dtype_hints['output_key'] = output_dtype_hint
        output_data = unittest.mock.MagicMock()
        self.processor_iteration['output_key'] = output_data
        mock_image.assert_called_once_with(data=output_data, axes='YX')
        mock_apply_output_dtype_hint.assert_called_once_with(
            self.processor.image0,
            mock_image.return_value.reorder_axes_like(),
            output_dtype_hint,
        )
        self.processor.create_output_image.assert_called_once_with(
            'output_key',
            mock_apply_output_dtype_hint.return_value.data.dtype,
        )


class apply_output_dtype_hint(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.base_image = unittest.mock.MagicMock()
        self.image = unittest.mock.MagicMock()
        self.dtype_hint = unittest.mock.Mock()

    def _setup_np_mock(self, mock_np):
        for dtype in ('float16', 'float32', 'float64', 'int16', 'uint8'):
            setattr(mock_np, dtype, unittest.mock.Mock())
            getattr(mock_np, dtype).name = dtype

        def _issubdtype(dtype, superset):
            self.assertIs(superset, mock_np.floating)
            return dtype in (mock_np.float16, mock_np.float32, mock_np.float64)

        mock_np.issubdtype.side_effect = _issubdtype

    def test__invalid_dtype_hint(self):
        dtype_hint = 'invalid_dtype_hint'
        with self.assertRaisesRegex(
            ValueError,
            f'Invalid dtype hint: "{dtype_hint}"'
        ):
            giatools.image_processor.apply_output_dtype_hint(
                self.base_image,
                self.image,
                dtype_hint,
            )

    def test__bool(self):
        self.image.reset_mock()
        result = giatools.image_processor.apply_output_dtype_hint(
            self.base_image,
            self.image,
            'bool',
        )
        self.image.astype.assert_called_once_with(bool)
        self.assertIs(result, self.image.astype.return_value)

    @unittest.mock.patch('giatools.image_processor._np')
    def test__binary(self, mock_np):
        self._setup_np_mock(mock_np)
        self.image.reset_mock()
        result = giatools.image_processor.apply_output_dtype_hint(
            self.base_image,
            self.image,
            'binary',
        )
        self.image.astype.assert_called_once_with(bool)
        self.image.astype.return_value.astype.assert_called_once_with(mock_np.uint8)
        self.assertIs(result, self.image.astype.return_value.astype.return_value * 255)

    @unittest.mock.patch('giatools.image_processor._np')
    def test__exact_float_types(self, mock_np):
        self._setup_np_mock(mock_np)
        for dtype_hint in ('float16', 'float32', 'float64'):
            with self.subTest(dtype_hint=dtype_hint):
                self.image.reset_mock()
                result = giatools.image_processor.apply_output_dtype_hint(
                    self.base_image,
                    self.image,
                    dtype_hint,
                )
                self.image.clip_to_dtype.assert_called_once_with(getattr(mock_np, dtype_hint))
                self.image.clip_to_dtype.return_value.astype.assert_called_once_with(getattr(mock_np, dtype_hint))
                self.assertIs(result, self.image.clip_to_dtype.return_value.astype.return_value)

    @unittest.mock.patch('giatools.image_processor._np')
    def test__floating(self, mock_np):
        self._setup_np_mock(mock_np)
        for src_dtype in (mock_np.float16, mock_np.int16):
            with self.subTest(src_dtype=src_dtype.name):
                self.image.reset_mock()
                self.image.data.dtype = src_dtype
                result = giatools.image_processor.apply_output_dtype_hint(
                    self.base_image,
                    self.image,
                    'floating',
                )
                if src_dtype is mock_np.float16:
                    self.assertIs(result, self.image)
                else:
                    self.image.clip_to_dtype.assert_called_once_with(mock_np.float64)
                    self.image.clip_to_dtype.return_value.astype.assert_called_once_with(mock_np.float64)
                    self.assertIs(result, self.image.clip_to_dtype.return_value.astype.return_value)

    @unittest.mock.patch('giatools.image_processor._np')
    def test__preserve_floating(self, mock_np):
        self._setup_np_mock(mock_np)
        for src_dtype in (mock_np.float16, mock_np.int16):
            for input_image_dtype in (mock_np.float32, mock_np.uint8):
                with self.subTest(src_dtype=src_dtype.name, input_image_dtype=input_image_dtype.name):
                    self.image.reset_mock()
                    self.base_image.data.dtype = input_image_dtype
                    self.image.data.dtype = src_dtype
                    result = giatools.image_processor.apply_output_dtype_hint(
                        self.base_image,
                        self.image,
                        'preserve_floating',
                    )
                    if src_dtype is mock_np.float16:
                        # Image is already float16, so no conversion needed
                        self.image.clip_to_dtype.assert_not_called()
                        self.image.astype.assert_not_called()
                        self.assertIs(result, self.image)
                    elif input_image_dtype is mock_np.float32:
                        # Image is not floating, but input image is float32, so convert to input image dtype (float32)
                        self.image.clip_to_dtype.assert_called_once_with(mock_np.float32)
                        self.image.clip_to_dtype.return_value.astype.assert_called_once_with(mock_np.float32)
                        self.assertIs(result, self.image.clip_to_dtype.return_value.astype.return_value)
                    elif input_image_dtype is mock_np.uint8:
                        # Image is not floating, input image is uint8, so convert to float64
                        self.image.clip_to_dtype.assert_called_once_with(mock_np.float64)
                        self.image.clip_to_dtype.return_value.astype.assert_called_once_with(mock_np.float64)
                        self.assertIs(result, self.image.clip_to_dtype.return_value.astype.return_value)
                    else:
                        assert False, (  # sanity check (should not be reached)
                            f'src_dtype={src_dtype.name}, input_image_dtype={input_image_dtype.name}'
                        )

    @unittest.mock.patch('giatools.image_processor._np')
    def test__preserve(self, mock_np):
        self._setup_np_mock(mock_np)
        for src_dtype in (mock_np.float16, mock_np.int16):
            for input_image_dtype in (mock_np.float32, mock_np.uint8):
                with self.subTest(src_dtype=src_dtype.name, input_image_dtype=input_image_dtype.name):
                    self.image.reset_mock()
                    self.base_image.data.dtype = input_image_dtype
                    self.image.data.dtype = src_dtype
                    result = giatools.image_processor.apply_output_dtype_hint(
                        self.base_image,
                        self.image,
                        'preserve',
                    )
                    if src_dtype is mock_np.float16 and input_image_dtype is mock_np.float32:
                        # Image must be converted to float32 (input image dtype)
                        self.image.clip_to_dtype.assert_called_once_with(mock_np.float32)
                        self.image.clip_to_dtype.return_value.astype.assert_called_once_with(mock_np.float32)
                        self.assertIs(result, self.image.clip_to_dtype.return_value.astype.return_value)
                    elif src_dtype is mock_np.float16 and input_image_dtype is mock_np.uint8:
                        # Image must be converted to uint8 (input image dtype)
                        self.image.clip_to_dtype.assert_called_once_with(mock_np.uint8)
                        self.image.clip_to_dtype.return_value.astype.assert_called_once_with(mock_np.uint8)
                        self.assertIs(result, self.image.clip_to_dtype.return_value.astype.return_value)
                    elif src_dtype is mock_np.int16 and input_image_dtype is mock_np.float32:
                        # Image must be converted to float32 (input image dtype)
                        self.image.clip_to_dtype.assert_called_once_with(mock_np.float32)
                        self.image.clip_to_dtype.return_value.astype.assert_called_once_with(mock_np.float32)
                        self.assertIs(result, self.image.clip_to_dtype.return_value.astype.return_value)
                    elif src_dtype is mock_np.int16 and input_image_dtype is mock_np.uint8:
                        # Image must be converted to uint8 (input image dtype)
                        self.image.clip_to_dtype.assert_called_once_with(mock_np.uint8)
                        self.image.clip_to_dtype.return_value.astype.assert_called_once_with(mock_np.uint8)
                        self.assertIs(result, self.image.clip_to_dtype.return_value.astype.return_value)
                    else:
                        assert False, (  # sanity check (should not be reached)
                            f'src_dtype={src_dtype.name}, input_image_dtype={input_image_dtype.name}'
                        )
