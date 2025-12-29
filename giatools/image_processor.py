import copy as _copy
from types import MappingProxyType as _ImmutableDict

import numpy as _np

import giatools.typing as _T
from giatools.image import Image as _Image

OutputDTypeHint = _T.Literal[
    'binary',  # like bool, but uses uint8 with 0/255 labels
    'bool',    # boolean dtype
    'float16',
    'float32',
    'float64',
    'floating',           # use the "native" float type passed to the processor, or convert to float64
    'preserve',           # use the same dtype as the input image
    'preserve_floating',  # use the float types of the following precedence: (i) native, (ii) input, (iii) float64
]


def apply_output_dtype_hint(base_image: _Image, image: _Image, dtype_hint: OutputDTypeHint) -> _Image:
    """
    Convert the data type of the `image` according to the specified `dtype_hint`.

    The `image` is not changed in place, a new image is returned (the data can be copied, but not necessarily). The
    `dtype_hint` parameter determines the policy for deriving the target `dtype` of the output image:

    - `'binary'`: Like `'bool'`, but convert to uint8 and use 0/255 labels instead of `False`/`True`.
    - `'bool'`: Convert to boolean type.
    - `'float16'`, `'float32'`, `'float64'`: Convert to the explicitly specified float type.
    - `'floating'`: Use the float type that the `image` already has, if applicable; otherwise, convert to float64.
    - `'preserve'`: Convert to the same dtype as the input image `base_image`.
    - `'preserve_floating'`: Use the float type that the `image` already has, if applicable; otherwise, convert to the
      float type of the input image `base_image`, if applicable; otherwise, convert to float64.

    Raises:
        ValueError: If `dtype_hint` is none of the above.
    """

    # Convert to binary image (uint8 with 0/255 labels)
    if dtype_hint == 'binary':
        return apply_output_dtype_hint(base_image, image, 'bool').astype(_np.uint8) * 255

    # Convert to bool
    if dtype_hint == 'bool':
        return image.astype(bool)

    # Use the specified float dtype
    if dtype_hint in ('float16', 'float32', 'float64', 'floating'):
        if dtype_hint == 'floating':
            if _np.issubdtype(image.data.dtype, _np.floating):
                return image  # no conversion needed, already a float type
            else:
                return image.clip_to_dtype(_np.float64).astype(_np.float64)  # clip and convert to float64
        else:
            dtype = getattr(_np, dtype_hint)
            return image.clip_to_dtype(dtype).astype(dtype)  # clip and convert to specified float type

    # Use the same dtype as the input image if it is a float type; otherwise same as `floating`
    if dtype_hint == 'preserve_floating':
        if _np.issubdtype(image.data.dtype, _np.floating):
            return image  # no conversion needed, already a float type
        elif _np.issubdtype(base_image.data.dtype, _np.floating):
            return apply_output_dtype_hint(base_image, image, 'preserve')  # clip and convert to the input image dtype
        else:
            return apply_output_dtype_hint(base_image, image, 'float64')  # clip and convert to float64

    # Use the same dtype as the input image
    if dtype_hint == 'preserve':
        return image.clip_to_dtype(  # clip and convert to the input image dtype
            base_image.data.dtype,
        ).astype(
            base_image.data.dtype,
        )

    # Invalid dtype hint
    else:
        raise ValueError(
            f'Invalid dtype hint: "{dtype_hint}"'
        )


class ImageProcessor:
    """
    Processes one or more images with the same shape and axes, and yields one or more output images of the same shape.

    Raises:
        ValueError: If no input images are provided, or if the input images do not have the same shape and axes.
    """

    image0: _Image
    """
    An input image that is used to determine the shape, axes, and metadata of the output images.
    """

    inputs: _ImmutableDict[_T.Union[str, int], _Image]
    """
    Dictionary of the input images, keyed by strings (keyword arguments) or integers (positional arguments).
    """

    outputs: _T.Dict[_T.Any, _Image]
    """
    Dictionary of the output images with arbitrary keys.
    """

    def __init__(self, *args: _Image, **kwargs: _Image):
        self.inputs = _ImmutableDict(
            {key: input for key, input in enumerate(args)} | kwargs
        )
        self.outputs = dict()

        # Verify that at least one input image is provided
        if len(self.inputs) == 0:
            raise ValueError('At least one input image must be provided.')

        # Verify that all input images have the same shape and axes
        self.image0 = next(iter(self.inputs.values()))
        for input_image in self.inputs.values():
            if self.image0.axes != input_image.axes or self.image0.data.shape != input_image.data.shape:
                raise ValueError('All input images must have the same shape and axes.')

    def process(
        self,
        joint_axes: str,
        output_dtype_hints: _T.Optional[_T.Dict[_T.Any, OutputDTypeHint]] = None,
    ) -> _T.Iterator['ProcessorIteration']:
        """
        Iterate over all slices of the input images along the given axes, yielding :py:class:`ProcessorIteration`
        objects that provide access to the corresponding sections of the input and output images.

        The axes in the yielded image sections correspond exactly to the `joint_axes` parameter (in the given order).

        The data written to the output images in each iteration is automatically normalized according to the policy
        specified for the respective output image via the `output_dtype_hints` mapping (dictionary of output keys to
        the respective policies; see :py:func:`apply_output_dtype_hint` for a list of possible values).

        .. note::

            This method requires **Python 3.11** or later.

        Example:

            >>> from giatools import Image, ImageProcessor
            >>> image = Image.read('data/input4_uint8.png')
            >>> print(image.axes, image.data.shape)
            QTZYXC (1, 1, 1, 10, 10, 3)
            >>>
            >>> proc = ImageProcessor(image)
            >>> for section in proc.process('XY'):
            ...     section['result'] = (
            ...         section[0].data > section[0].data.mean()
            ...     )
            >>>
            >>> import numpy as np
            >>> expected_result = np.stack(
            ...     [
            ...         image.data[..., c] > image.data[..., c].mean()
            ...         for c in range(image.data.shape[-1])
            ...     ],
            ...     axis=-1,
            ... )
            >>> print(
            ...     np.allclose(
            ...         proc.outputs['result'].data,
            ...         expected_result,
            ...     ),
            ...     proc.outputs['result'].metadata == image.metadata,
            ... )
            True True

        Raises:
            RuntimeError: If Python version is less than 3.11.
            ValueError: If `joint_axes` contains invalid axes (must be a non-empty subset of the image axes); or if
                `output_dtype_hints` contains invalid keys (must be a subset of the output image keys) or values.
        """
        # Validate `output_dtype_hints`
        output_dtype_hints = _ImmutableDict(output_dtype_hints or dict())
        for key, dtype_hint in output_dtype_hints.items():
            if key in self.outputs:
                if dtype_hint not in _T.get_args(OutputDTypeHint):
                    raise ValueError(
                        f'Invalid dtype hint "{dtype_hint}" for output image with key "{key}".'
                    )
            else:
                raise ValueError(f'Output image with key "{key}" does not exist.')

        # Iterate through input images jointly
        input_keys, input_images = zip(*self.inputs.items())
        for inputs_info in zip(*(input_image.iterate_jointly(joint_axes) for input_image in input_images)):
            source_slices, sections = zip(*inputs_info)
            processor_iteration = ProcessorIteration(
                self,
                _ImmutableDict(dict(zip(input_keys, sections))),
                source_slices[0],  # same for all input images (due to same shape and axes)
                joint_axes,
                output_dtype_hints,
            )
            yield processor_iteration

    def create_output_image(self, key: _T.Any, dtype: _np.dtype) -> _Image:
        """
        Create and return an output image with the given key and data type.

        The output image will have the same shape, axes, and metadata as the input images. The metadata is copied.

        Raises:
            ValueError: If an output image with the given key already exists.
        """
        if key not in self.outputs:
            self.outputs[key] = (
                _image := _Image(
                    data=_np.empty(self.image0.data.shape, dtype=dtype),
                    axes=self.image0.axes,
                    original_axes=self.image0.original_axes,
                    metadata=_copy.deepcopy(self.image0.metadata),
                )
            )
            return _image
        else:
            raise ValueError(f'Output image with key "{key}" already exists.')


class ProcessorIteration:
    """
    Represents a single iteration of an :py:class:`ImageProcessor`, providing access to corresponding input and output
    image sections.
    """

    _input_sections: _ImmutableDict[_T.Union[str, int], _Image]

    _output_slice: _T.NDSlice

    _output_dtype_hints: _ImmutableDict[_T.Any, OutputDTypeHint]

    _processor: ImageProcessor

    joint_axes: str
    """
    The axes of the image sections in this iteration.
    """

    def __init__(
        self,
        processor: ImageProcessor,
        input_sections: _ImmutableDict[_T.Union[str, int], _Image],
        output_slice: _T.NDSlice,
        joint_axes: str,
        output_dtype_hints: _ImmutableDict[_T.Any, OutputDTypeHint],
    ):
        self._input_sections = input_sections
        self._output_slice = output_slice
        self._output_dtype_hints = output_dtype_hints
        self._processor = processor
        self.joint_axes = joint_axes

    @property
    def _num_positional_arguments(self) -> int:
        return sum(1 for key in self._input_sections.keys() if isinstance(key, int))

    def __getitem__(self, key: _T.Union[str, int]) -> _Image:
        """
        Get the input image section corresponding to the given key.

        Raises:
            KeyError: If no input image was passed in by keyword argument equal to the given key.
            IndexError: If no input image was passed in by positional argument at the given position.
        """
        if isinstance(key, int):
            pos = key
            if key < 0:
                key += self._num_positional_arguments
            if key not in self._input_sections:
                raise IndexError(f'No input image at position {pos}.')
        elif key not in self._input_sections:
            raise KeyError(f'No input image with key "{key}".')
        return self._input_sections[key]

    def __setitem__(self, key: _T.Any, data: _T.NDArray):
        """
        Set the output image section corresponding to the given key.
        """
        section = _Image(data=data, axes=self.joint_axes).reorder_axes_like(
            ''.join([axis for axis in self._processor.image0.axes if axis in self.joint_axes])
        )

        # Apply output dtype hint (if provided)
        if (output_dtype_hint := self._output_dtype_hints.get(key)):
            section = apply_output_dtype_hint(self._processor.image0, section, output_dtype_hint)

        # Create output image (if it does not exist yet)
        if key not in self._processor.outputs:
            self._processor.create_output_image(key, section.data.dtype)

        # Write data to output image
        self._processor.outputs[key].data[self._output_slice] = section.data
