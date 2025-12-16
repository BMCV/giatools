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

    z_spacing: Optional[float] = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(float),
        ),
    )

    z_position: Optional[float] = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(float),
        ),
    )

    unit: Optional[Unit] = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.in_(get_args(Unit)),
        ),
    )
