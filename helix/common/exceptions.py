"""Project-specific exception types used across Helix infrastructure."""

from __future__ import annotations


class HelixError(Exception):
    """Base class for Helix-specific errors."""


class HelixConfigurationError(HelixError):
    """Raised when a configuration file or field is invalid."""


class HelixPathError(HelixError):
    """Raised when a path cannot be resolved within expected boundaries."""


class HelixDependencyError(HelixError):
    """Raised when an optional dependency required for an operation is unavailable."""
