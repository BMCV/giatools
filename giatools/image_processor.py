import copy as _copy
from types import MappingProxyType as _ImmutableDict

import numpy as _np

import giatools.typing as _T
from giatools.image import Image as _Image


class ImageProcessor:
    """
    Processes one or more images with the same shape and axes, and yields one or more output images of the same shape.

    Raises:
        ValueError: If no input images are provided, or if the input images do not have the same shape and axes.
    """

    inputs: _ImmutableDict[_T.Union[str, int], _Image]
    """
    Dictionary of the input images, keyed by strings (keyword arguments) or integers (positional arguments).
    """

    outputs: _T.Dict[_T.Any, _Image]
    """
    Dictionary of the output images with arbitrary keys.
    """

    image0: _Image
    """
    An input image that is used to determine the shape, axes, and metadata of the output images.
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
        for input in self.inputs.values():
            if self.image0.axes != input.axes or self.image0.data.shape != input.data.shape:
                raise ValueError('All input images must have the same shape and axes.')

    def process(self, joint_axes: str) -> _T.Iterator['ProcessorIteration']:
        """
        Iterate over all slices of the input images along the given axes, yielding :py:class:`ProcessorIteration`
        objects that provide access to the corresponding sections of the input and output images.

        The axes in the yielded image sections correspond exactly to the `joint_axes` parameter (in the given order).

        .. note::

            This method requires **Python 3.11** or later.

        Example:

            .. runblock:: pycon

                >>> from giatools import Image, ImageProcessor
                >>> image = Image.read('data/input4_uint8.png')
                >>> print(image.axes, image.data.shape)
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

        Raises:
            RuntimeError: If Python version is less than 3.11.
            ValueError: If `joint_axes` contains invalid axes (must be a non-empty subset of the image axes).
        """
        input_keys, input_images = zip(*self.inputs.items())
        for inputs_info in zip(*(input_image.iterate_jointly(joint_axes) for input_image in input_images)):
            source_slices, sections = zip(*inputs_info)
            processor_iteration = ProcessorIteration(
                self,
                _ImmutableDict(dict(zip(input_keys, sections))),
                source_slices[0],  # same for all input images (due to same shape and axes)
                joint_axes,
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
    ):
        self._processor = processor
        self._input_sections = input_sections
        self._output_slice = output_slice
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
        if key not in self._processor.outputs:
            self._processor.create_output_image(key, data.dtype)
        section = _Image(data=data, axes=self.joint_axes).reorder_axes_like(
            ''.join([axis for axis in self._processor.image0.axes if axis in self.joint_axes])
        )
        self._processor.outputs[key].data[self._output_slice] = section.data
