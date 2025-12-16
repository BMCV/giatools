"""
Copyright 2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import attrs

from .typing import (
    Literal,
    Optional,
    Tuple,
    get_args,
)

#: Valid units for metadata
Unit = Literal[
    'inch',
    'nm',
    'um',
    'mm',
    'cm',
    'm',
    'km',
]


@attrs.define
class Metadata:
    """
    Image metadata.
    """

    resolution: Optional[Tuple[float, float]] = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            [
                attrs.validators.instance_of(tuple),
                attrs.validators.min_len(2),
                attrs.validators.max_len(2),
                attrs.validators.deep_iterable(
                    member_validator=attrs.validators.instance_of(float),
                    iterable_validator=None,
                ),
            ],
        )
    )
    """
    Pixels per unit in X and Y dimensions.
    """

    @property
    def pixel_size(self) -> Optional[Tuple[float, float]]:
        """
        The pixel size in X and Y dimensions. This is identical to the pixel spacing in X and Y dimensions.
        """
        return (
            1 / self.resolution[0],
            1 / self.resolution[1],
        ) if self.resolution is not None else None

    @pixel_size.setter
    def pixel_size(self, value: Optional[Tuple[float, float]]):
        if value is None:
            self.resolution = None
        else:
            if len(value) != 2 or not all(isinstance(val, float) for val in value):
                raise ValueError('Pixel size must be a tuple of two non-None floats or None.')
            self.resolution = (
                1 / value[0],
                1 / value[1],
            )

    z_spacing: Optional[float] = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(float),
        ),
    )
    """
    The pixel spacing in the Z dimension.
    """

    z_position: Optional[float] = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(float),
        ),
    )
    """
    The position of the image in the Z dimension.
    """

    unit: Optional[Unit] = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.in_(get_args(Unit)),
        ),
    )
    """
    The unit of measurement (e.g., inch, nm, um, mm, cm, m, km).
    """
