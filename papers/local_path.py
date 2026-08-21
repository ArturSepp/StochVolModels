"""Compatibility facade for the package-wide local path resolver.

New paper and example code imports ``local_path`` from ``stochvolmodels``.
These re-exports preserve older reproduction scripts while ensuring every path
is resolved from the same package-adjacent ``settings.yaml`` configuration.
"""

from stochvolmodels.local_path import (
    get_local_resource_path,
    get_output_path,
    get_paths,
    get_resource_path,
)

__all__ = (
    'get_local_resource_path',
    'get_output_path',
    'get_paths',
    'get_resource_path',
)
