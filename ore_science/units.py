"""Typed SI units, dimension vectors, and dimensional homogeneity checks for scientific tasks.

This module implements the mathematical validation core required by Stage 2 of the
scientific-layer roadmap: unit normalization, conversion mappings, and dimensional
consistency. Dimensions are represented as integer exponent vectors over the seven SI
base quantities (mass, length, time, electric current, thermodynamic temperature,
amount of substance, luminous intensity). Temperature units with offsets (Celsius,
Fahrenheit) are handled explicitly so that numerically similar values in different
units can never be silently treated as equivalent.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

__all__ = [
    "BASE_QUANTITIES",
    "DIM_AREA",
    "DIM_DIFFUSIVITY",
    "DIM_FORCE",
    "DIM_HEAT_RATE",
    "DIM_LENGTH",
    "DIM_TEMPERATURE",
    "DIM_TIME",
    "DIM_VELOCITY",
    "UNITS",
    "Dimension",
    "Quantity",
    "Unit",
    "dimensional_homogeneity",
    "dimensionless",
    "parse_unit",
    "quantity_from_string",
]


BASE_QUANTITIES: tuple[str, ...] = (
    "mass", "length", "time", "current", "temperature", "amount", "luminous_intensity"
)

_MASS, _LENGTH, _TIME, _CURRENT, _TEMPERATURE, _AMOUNT, _LUM = range(7)


def _exponents(**kwargs: int) -> tuple[int, ...]:
    """Build a dimension exponent vector from named base-quantity exponents."""
    values = [0] * 7
    for name, exponent in kwargs.items():
        index = BASE_QUANTITIES.index(name)
        values[index] = int(exponent)
    return tuple(values)


@dataclass(frozen=True, slots=True)
class Dimension:
    """Integer exponent vector over the seven SI base quantities."""

    exponents: tuple[int, ...]

    def __post_init__(self) -> None:
        """Require exactly seven integer exponents."""
        if len(self.exponents) != 7 or any(not isinstance(v, int) for v in self.exponents):
            raise ValueError("Dimension requires seven integer exponents")

    def is_dimensionless(self) -> bool:
        """Return True when no base quantity appears in the dimension."""
        return all(value == 0 for value in self.exponents)

    def __mul__(self, other: Dimension) -> Dimension:
        """Multiplying quantities adds their dimension exponents."""
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def __truediv__(self, other: Dimension) -> Dimension:
        """Dividing quantities subtracts their dimension exponents."""
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def __pow__(self, power: int) -> Dimension:
        """Raising a quantity to an integer power scales its exponents."""
        return Dimension(tuple(value * int(power) for value in self.exponents))

    def __eq__(self, other: object) -> bool:
        """Compare dimension vectors; dimensionless equals only dimensionless."""
        return isinstance(other, Dimension) and self.exponents == other.exponents

    def __hash__(self) -> int:
        """Hash the exponent vector so dimensions can key mappings."""
        return hash(self.exponents)

    def describe(self) -> str:
        """Render the dimension in base-quantity symbols, for example m^2 s^-1."""
        symbols = ("kg", "m", "s", "A", "K", "mol", "cd")
        parts = []
        for symbol, exponent in zip(symbols, self.exponents):
            if exponent == 0:
                continue
            parts.append(symbol if exponent == 1 else f"{symbol}^{exponent}")
        return " ".join(parts) if parts else "1 (dimensionless)"


dimensionless = Dimension((0, 0, 0, 0, 0, 0, 0))

# Common derived dimensions used by the scientific layer.
DIM_LENGTH = Dimension(_exponents(length=1))
DIM_AREA = Dimension(_exponents(length=2))
DIM_TIME = Dimension(_exponents(time=1))
DIM_TEMPERATURE = Dimension(_exponents(temperature=1))
DIM_DIFFUSIVITY = Dimension(_exponents(length=2, time=-1))
DIM_HEAT_RATE = Dimension(_exponents(length=2, time=-1, temperature=1))
DIM_VELOCITY = Dimension(_exponents(length=1, time=-1))
DIM_FORCE = Dimension(_exponents(mass=1, length=1, time=-2))


@dataclass(frozen=True, slots=True)
class Unit:
    """A unit of measure: a dimension plus a linear or affine mapping to SI.

    ``factor`` converts this unit to the coherent SI unit of the same dimension.
    ``offset`` supports affine units such as degrees Celsius: value_si = factor * value + offset.
    """

    name: str
    symbol: str
    dimension: Dimension
    factor: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        """Validate names, symbols, and finite conversion parameters."""
        if not self.name.strip() or not self.symbol.strip():
            raise ValueError("unit name and symbol must be non-empty")
        if not math.isfinite(self.factor) or self.factor <= 0.0:
            raise ValueError("unit factor must be finite and positive")
        if not math.isfinite(self.offset):
            raise ValueError("unit offset must be finite")
        if self.offset != 0.0 and self.dimension != DIM_TEMPERATURE:
            raise ValueError("affine offsets are only supported for temperature units")

    def is_affine(self) -> bool:
        """Return True for units whose conversion includes an additive offset."""
        return self.offset != 0.0

    def to_si(self, value: float) -> float:
        """Convert a numeric value in this unit to the coherent SI value."""
        return self.factor * float(value) + self.offset

    def from_si(self, value_si: float) -> float:
        """Convert a coherent SI value back into this unit."""
        return (float(value_si) - self.offset) / self.factor

    def compatible(self, other: Unit) -> bool:
        """Two units are compatible when they share the same dimension."""
        return isinstance(other, Unit) and self.dimension == other.dimension

    def describe_short(self) -> str:
        """Render symbol and dimension compactly for error messages."""
        return f"{self.symbol} [{self.dimension.describe()}]"

    def __mul__(self, other: Unit) -> Unit:
        """Compose a derived unit by multiplying conversion factors."""
        if not isinstance(other, Unit):
            return NotImplemented
        if self.is_affine() or other.is_affine():
            raise ValueError("affine units cannot be multiplied; convert to SI first")
        return Unit(
            name=f"{self.name}·{other.name}",
            symbol=f"{self.symbol}·{other.symbol}",
            dimension=self.dimension * other.dimension,
            factor=self.factor * other.factor,
        )

    def __truediv__(self, other: Unit) -> Unit:
        """Compose a derived unit by dividing conversion factors."""
        if not isinstance(other, Unit):
            return NotImplemented
        if self.is_affine() or other.is_affine():
            raise ValueError("affine units cannot be divided; convert to SI first")
        return Unit(
            name=f"{self.name}/{other.name}",
            symbol=f"{self.symbol}/{other.symbol}",
            dimension=self.dimension / other.dimension,
            factor=self.factor / other.factor,
        )

    def __pow__(self, power: int) -> Unit:
        """Raise a linear unit to an integer power."""
        if self.is_affine():
            raise ValueError("affine units cannot be raised to powers; convert to SI first")
        return Unit(
            name=f"{self.name}^{power}",
            symbol=f"{self.symbol}^{power}",
            dimension=self.dimension ** int(power),
            factor=self.factor ** int(power),
        )


_SI_BASE_UNITS: tuple[Unit, ...] = (
    Unit("kilogram", "kg", Dimension(_exponents(mass=1))),
    Unit("metre", "m", Dimension(_exponents(length=1))),
    Unit("second", "s", Dimension(_exponents(time=1))),
    Unit("ampere", "A", Dimension(_exponents(current=1))),
    Unit("kelvin", "K", DIM_TEMPERATURE),
    Unit("mole", "mol", Dimension(_exponents(amount=1))),
    Unit("candela", "cd", Dimension(_exponents(luminous_intensity=1))),
)


@dataclass(frozen=True, slots=True)
class Quantity:
    """A numeric value bound to a unit, with dimension-checked arithmetic.

    Arithmetic never silently drops units: addition and subtraction require identical
    dimensions, and results carry the composed unit. Affine temperature units are
    normalized to kelvin before arithmetic to prevent false equivalence between
    numerically similar values in different units.
    """

    value: float
    unit: Unit

    def __post_init__(self) -> None:
        """Require a finite numeric value and a typed unit."""
        if not math.isfinite(float(self.value)):
            raise ValueError("quantity value must be finite")
        if not isinstance(self.unit, Unit):
            raise TypeError("quantity unit must be a Unit")

    # -- construction helpers -------------------------------------------------
    def in_si(self) -> Quantity:
        """Return the same physical value expressed in coherent SI units."""
        return Quantity(self.unit.to_si(self.value), _si_unit_for(self.unit.dimension))

    def convert_to(self, target: Unit) -> Quantity:
        """Convert into a compatible unit, preserving the physical value exactly."""
        if not self.unit.compatible(target):
            raise ValueError(
                f"cannot convert {self.unit.describe_short()} to {target.describe_short()}: "
                "dimensions differ"
            )
        return Quantity(target.from_si(self.unit.to_si(self.value)), target)

    # -- dimension-checked arithmetic ----------------------------------------
    def __add__(self, other: Quantity) -> Quantity:
        """Add two quantities of the same dimension, normalizing to a common unit."""
        if not isinstance(other, Quantity):
            return NotImplemented
        if self.unit.dimension != other.unit.dimension:
            raise ValueError(
                f"cannot add {self.unit.describe_short()} and {other.unit.describe_short()}: "
                "incompatible dimensions"
            )
        left, right = self.in_si(), other.in_si()
        return Quantity(left.value + right.value, left.unit)

    def __sub__(self, other: Quantity) -> Quantity:
        """Subtract two quantities of the same dimension, normalizing to a common unit."""
        if not isinstance(other, Quantity):
            return NotImplemented
        if self.unit.dimension != other.unit.dimension:
            raise ValueError(
                f"cannot subtract {self.unit.symbol} and {other.unit.symbol}: incompatible dimensions"
            )
        left, right = self.in_si(), other.in_si()
        return Quantity(left.value - right.value, left.unit)

    def __mul__(self, other: Quantity | float) -> Quantity:
        """Multiply by a quantity (composing units) or by a bare number."""
        if isinstance(other, Quantity):
            left, right = self.in_si(), other.in_si()
            return Quantity(left.value * right.value, left.unit * right.unit)
        return Quantity(float(self.value) * float(other), self.unit)

    def __truediv__(self, other: Quantity | float) -> Quantity:
        """Divide by a quantity (composing units) or by a bare number."""
        if isinstance(other, Quantity):
            left, right = self.in_si(), other.in_si()
            if right.value == 0.0:
                raise ZeroDivisionError("quantity division by zero")
            return Quantity(left.value / right.value, left.unit / right.unit)
        return Quantity(float(self.value) / float(other), self.unit)

    def __pow__(self, power: int) -> Quantity:
        """Raise a linear-unit quantity to an integer power."""
        return Quantity(float(self.value) ** int(power), self.unit ** int(power))

    def describe(self) -> str:
        """Render the quantity with its unit symbol."""
        return f"{self.value:g} {self.unit.symbol}"


def _si_unit_for(dimension: Dimension) -> Unit:
    """Return the coherent SI unit composite for a dimension, or dimensionless 1."""
    parts = []
    for base, exponent in zip(_SI_BASE_UNITS, dimension.exponents):
        if exponent == 0:
            continue
        parts.append(base if exponent == 1 else base ** exponent)
    if not parts:
        return UNITS["dimensionless"]
    result = parts[0]
    for part in parts[1:]:
        result = result * part
    return result


def _build_unit_table() -> dict[str, Unit]:
    """Assemble the standard unit table with SI prefixes and common derived units."""
    si = {unit.symbol: unit for unit in _SI_BASE_UNITS}
    table: dict[str, Unit] = {
        **si,
        "dimensionless": Unit("dimensionless", "1", dimensionless),
        "g": Unit("gram", "g", Dimension(_exponents(mass=1)), factor=1e-3),
        "ms": Unit("millisecond", "ms", Dimension(_exponents(time=1)), factor=1e-3),
        "min": Unit("minute", "min", Dimension(_exponents(time=1)), factor=60.0),
        "h": Unit("hour", "h", Dimension(_exponents(time=1)), factor=3600.0),
        "km": Unit("kilometre", "km", DIM_LENGTH, factor=1e3),
        "cm": Unit("centimetre", "cm", DIM_LENGTH, factor=1e-2),
        "mm": Unit("millimetre", "mm", DIM_LENGTH, factor=1e-3),
        "degC": Unit("degree Celsius", "°C", DIM_TEMPERATURE, factor=1.0, offset=273.15),
        "degF": Unit(
            "degree Fahrenheit", "°F", DIM_TEMPERATURE,
            factor=5.0 / 9.0, offset=255.3722222222222,
        ),
        "N": Unit("newton", "N", DIM_FORCE),
        "J": Unit("joule", "J", Dimension(_exponents(mass=1, length=2, time=-2))),
        "W": Unit("watt", "W", Dimension(_exponents(mass=1, length=2, time=-3))),
        "Pa": Unit("pascal", "Pa", Dimension(_exponents(mass=1, length=-1, time=-2))),
        "Hz": Unit("hertz", "Hz", Dimension(_exponents(time=-1))),
        "C_charge": Unit("coulomb", "C", Dimension(_exponents(time=1, current=1))),
        "eV": Unit(
            "electronvolt", "eV", Dimension(_exponents(mass=1, length=2, time=-2)),
            factor=1.602176634e-19,
        ),
    }
    table["m2"] = table["m"] ** 2
    table["m3"] = table["m"] ** 3
    table["s2"] = table["s"] ** 2
    table["m2_s"] = table["m2"] / table["s"]
    table["m_s"] = table["m"] / table["s"]
    table["K_s"] = table["K"] / table["s"]
    table["m2_s2"] = table["m2"] / table["s2"]
    table["K_m"] = table["K"] / table["m"]
    table["K_m2"] = table["K"] / table["m2"]
    table["N_m"] = table["N"] / table["m"]
    return table


UNITS: Mapping[str, Unit] = _build_unit_table()

# Symbols accepted in compact textual quantity expressions.
_SYMBOL_ALIASES: Mapping[str, str] = {
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "s": "s", "sec": "s", "second": "s", "seconds": "s",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "g": "g", "gram": "g", "grams": "g",
    "K": "K", "kelvin": "K", "kelvins": "K",
    "degC": "degC", "celsius": "degC", "°C": "degC",
    "degF": "degF", "fahrenheit": "degF", "°F": "degF",
    "ms": "ms", "millisecond": "ms", "milliseconds": "ms",
    "min": "min", "minute": "min", "minutes": "min",
    "h": "h", "hour": "h", "hours": "h", "hr": "h",
    "km": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm",
    "mm": "mm", "millimeter": "mm", "millimeters": "mm",
    "N": "N", "newton": "N", "newtons": "N",
    "J": "J", "joule": "J", "joules": "J",
    "W": "W", "watt": "W", "watts": "W",
    "Pa": "Pa", "pascal": "Pa", "pascals": "Pa",
    "Hz": "Hz", "hertz": "Hz",
    "mol": "mol", "mole": "mol", "moles": "mol",
    "A": "A", "ampere": "A", "amperes": "A",
    "eV": "eV", "electronvolt": "eV", "electronvolts": "eV",
    "m2": "m2", "m²": "m2", "m^2": "m2", "square meter": "m2", "square meters": "m2",
    "m2_s": "m2_s", "m²/s": "m2_s", "m^2/s": "m2_s",
    "m_s": "m_s", "m/s": "m_s",
    "K_s": "K_s", "K/s": "K_s",
    "K_m": "K_m", "K/m": "K_m",
    "K_m2": "K_m2", "K/m²": "K_m2", "K/m^2": "K_m2",
    "N_m": "N_m", "N/m": "N_m",
    "m3": "m3", "m³": "m3", "m^3": "m3",
    "km_h": "km_h", "km/h": "km_h", "kph": "km_h",
}


def parse_unit(text: str) -> Unit:
    """Parse a unit from a symbol or name, raising ValueError for unknown units."""
    key = _SYMBOL_ALIASES.get(text.strip())
    if key is None:
        raise ValueError(f"unknown unit: {text!r}")
    return UNITS[key]


def quantity_from_string(text: str) -> Quantity:
    """Parse strings such as ``'1.5 m'`` or ``'60 s'`` into a typed Quantity.

    Raises ValueError when the string does not contain a numeric value followed by a
    known unit symbol, so unknown units fail loudly instead of being treated as numbers.
    """
    parts = text.strip().split()
    if len(parts) != 2:
        raise ValueError(f"expected '<value> <unit>' but received {text!r}")
    value, unit_text = parts
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"expected a numeric value but received {value!r}") from error
    return Quantity(number, parse_unit(unit_text))


def dimensional_homogeneity(terms: Iterable[Quantity]) -> tuple[bool, str]:
    """Check that every term of an equation shares one dimension.

    Returns ``(True, '')`` when homogeneous. Otherwise returns ``False`` and a message
    naming the first offending pair, which callers must surface as a formalization
    failure rather than silently coercing the value.
    """
    checked: list[Quantity] = []
    for term in terms:
        normalized = term.in_si()
        checked.append(normalized)
        if len(checked) > 1 and checked[-1].unit.dimension != checked[0].unit.dimension:
            return (
                False,
                f"dimensional mismatch: term '{checked[0].describe()}' has dimension "
                f"{checked[0].unit.dimension.describe()} but term '{checked[-1].describe()}' "
                f"has dimension {checked[-1].unit.dimension.describe()}",
            )
    return True, ""
