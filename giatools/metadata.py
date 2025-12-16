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
