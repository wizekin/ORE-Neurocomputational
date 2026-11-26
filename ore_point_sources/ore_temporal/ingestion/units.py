"""Explicit, reversible unit validation and conversion (specification Step 12)."""
from __future__ import annotations

import math

__all__ = ("UnitConversionError", "UnitRegistry", "convert_units")


class UnitConversionError(ValueError):
    """Raised when a unit conversion is incompatible or undocumented."""


class UnitRegistry:
    """A registry of explicitly declared, reversible unit conversions.

    Units are never inferred from magnitude; conversions must exist in the registry
    and are recorded so provenance can reverse them.
    """

    #: Documented same-dimension conversions expressed as (source_unit, target_unit): factor.
    CONVERSIONS: dict[tuple[str, str], float] = {
        ("s", "s"): 1.0,
        ("ms", "s"): 1e-3,
        ("s", "ms"): 1e3,
        ("us", "s"): 1e-6,
        ("ns", "s"): 1e-9,
        ("V", "mV"): 1e3,
        ("mV", "V"): 1e-3,
        ("V", "V"): 1.0,
        ("mV", "mV"): 1.0,
        ("degC", "K"): None,  # affine conversion handled explicitly below
        ("K", "degC"): None,
        ("degC", "degC"): 1.0,
        ("K", "K"): 1.0,
    }

    #: Documented unit dimension families used for compatibility checks.
    DIMENSIONS: dict[str, tuple[str, ...]] = {
        "time": ("s", "ms", "us", "ns"),
        "voltage": ("V", "mV"),
        "temperature": ("degC", "K"),
    }

    def dimension_of(self, unit: str) -> str | None:
        """Return the dimension family of a unit, or None when undocumented."""
        for dimension, members in self.DIMENSIONS.items():
            if unit in members:
                return dimension
        return None

    def compatible(self, source_unit: str, target_unit: str) -> bool:
        """Report whether two units belong to the same documented dimension."""
        source_dimension = self.dimension_of(source_unit)
        target_dimension = self.dimension_of(target_unit)
        if source_dimension is None or target_dimension is None:
            return source_unit == target_unit
        return source_dimension == target_dimension

    def convert(self, value: float, source_unit: str, target_unit: str) -> tuple[float, str]:
        """Convert a value between units, returning the converted value and a reversible record.

        Raises UnitConversionError for incompatible or undocumented conversions.
        """
        numeric = float(value)
        if not math.isfinite(numeric):
            raise UnitConversionError("values subject to unit conversion must be finite")
        if not self.compatible(source_unit, target_unit):
            raise UnitConversionError(
                f"incompatible unit conversion requested: {source_unit!r} -> {target_unit!r}")
        if source_unit == target_unit:
            return (numeric, f"identity:{source_unit}")
        key = (source_unit, target_unit)
        if key not in self.CONVERSIONS:
            raise UnitConversionError(
                f"no documented conversion for {source_unit!r} -> {target_unit!r}")
        factor = self.CONVERSIONS[key]
        if factor is None:
            if source_unit == "degC" and target_unit == "K":
                return (numeric + 273.15, "affine:degC->K:+273.15")
            if source_unit == "K" and target_unit == "degC":
                return (numeric - 273.15, "affine:K->degC:-273.15")
            raise UnitConversionError(f"malformed affine conversion entry for {key}")
        return (numeric * factor, f"linear:{source_unit}->{target_unit}:x{factor}")

    def validate_observation_units(self, units: tuple[str, ...],
                                   schema_units: tuple[str, ...]) -> list[str]:
        """Validate that observation units are declared and dimensionally compatible."""
        issues: list[str] = []
        for index, (observed, declared) in enumerate(zip(units, schema_units)):
            if observed != declared and not self.compatible(observed, declared):
                issues.append(
                    f"channel {index} unit {observed!r} is incompatible with the declared "
                    f"schema unit {declared!r}")
        return issues


def convert_units(value: float, source_unit: str, target_unit: str) -> tuple[float, str]:
    """Convert a single value using the default registry."""
    return UnitRegistry().convert(value, source_unit, target_unit)
