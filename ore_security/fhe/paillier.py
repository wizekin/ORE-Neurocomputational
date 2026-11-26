"""Homomorphic encryption adapter: additive Paillier reference (spec sections 1, 4).

The adapter implements the FHE contract ``Eval(f, E(pk, x)) = E(pk, f(x))`` for the
supported operation algebra: ciphertext addition, scalar multiplication and negation.
Requests outside that algebra are rejected, and decryption requires an explicit
authorization decision, so homomorphic output stays encrypted until policy permits
release.

Honest scope: this is a *partially* homomorphic scheme (additive), used to exercise
the contract and the acceptance tests. Real FHE schemes (BFV/BGV/CKKS) with noise
management and parameter sets must come from a vetted library; the adapter interface
stays identical so a production backend can replace this implementation.

Security: Paillier indistinguishability relies on the decisional composite residuosity
assumption. The default 1024-bit modulus is reduced-strength for fast tests; use
2048+ bits for anything sensitive.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Callable

from ..contracts import PolicyDecision
from ..exceptions import AuthorizationError, HomomorphicEncryptionError

Clock = Callable[[], float]
SUPPORTED_OPERATIONS = frozenset({"add", "scalar_mul", "negate", "sub"})
ADAPTER_ID = "paillier-additive-reference-1.0.0"
PARAMETER_SETS = frozenset({"paillier-1024-test", "paillier-2048", "paillier-3072"})
_PARAMETER_BITS = {"paillier-1024-test": 512, "paillier-2048": 1024, "paillier-3072": 1536}


def _is_probable_prime(n: int, rounds: int = 16) -> bool:
    """Miller-Rabin primality test with CSPRNG witnesses."""
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if n % prime == 0:
            return n == prime
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _generate_prime(bits: int) -> int:
    while True:
        candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(candidate):
            return candidate


@dataclass(frozen=True, slots=True)
class PaillierPublicKey:
    """Public key (n, g = n + 1) for encryption and homomorphic evaluation."""

    n: int
    n_squared: int

    @property
    def g(self) -> int:
        return self.n + 1


@dataclass(frozen=True, slots=True)
class PaillierCiphertext:
    """One ciphertext value; supports only the documented homomorphic algebra."""

    value: int
    public_key: PaillierPublicKey

    def __post_init__(self) -> None:
        if not 0 <= self.value < self.public_key.n_squared:
            raise HomomorphicEncryptionError("ciphertext outside the valid range")

    def __add__(self, other: "PaillierCiphertext") -> "PaillierCiphertext":
        return homomorphic_add(self, other)

    def __mul__(self, scalar: int) -> "PaillierCiphertext":
        return homomorphic_scalar_mul(self, scalar)


class PaillierContext:
    """Key holder and evaluator implementing the homomorphic-encryption contract."""

    def __init__(self, parameter_set: str = "paillier-1024-test") -> None:
        if parameter_set not in PARAMETER_SETS:
            raise HomomorphicEncryptionError(
                f"unsupported parameter set {parameter_set!r}; choose from {sorted(PARAMETER_SETS)}"
            )
        self.parameter_set = parameter_set
        bits = _PARAMETER_BITS[parameter_set]
        p = _generate_prime(bits)
        q = _generate_prime(bits)
        while q == p:
            q = _generate_prime(bits)
        n = p * q
        self.n = n
        self.n_squared = n * n
        self.public_key = PaillierPublicKey(n, self.n_squared)
        self._lambda = (p - 1) * (q - 1) // _gcd(p - 1, q - 1)
        self._mu = pow(_l_function(pow(self.public_key.g, self._lambda, self.n_squared), n),
                       -1, n)

    def encrypt(self, plaintext: int) -> PaillierCiphertext:
        """Encrypt an integer message with fresh randomness."""
        if not 0 <= plaintext < self.n:
            raise HomomorphicEncryptionError(
                f"plaintext must be in [0, n); got {plaintext} with n of {self.n.bit_count()} bits"
            )
        while True:
            r = secrets.randbelow(self.n - 1) + 1
            if _gcd(r, self.n) == 1:
                break
        c = (pow(self.public_key.g, plaintext, self.n_squared)
             * pow(r, self.n, self.n_squared)) % self.n_squared
        return PaillierCiphertext(c, self.public_key)

    def decrypt(self, ciphertext: PaillierCiphertext,
                authorization: PolicyDecision | None = None) -> int:
        """Decrypt only when the policy decision explicitly allows the release."""
        if authorization is None or not authorization.allowed:
            raise AuthorizationError(
                "homomorphic output remains encrypted: decryption requires an allowed policy decision"
            )
        if ciphertext.public_key.n != self.n:
            raise HomomorphicEncryptionError("ciphertext belongs to a different key")
        return (_l_function(pow(ciphertext.value, self._lambda, self.n_squared), self.n)
                * self._mu) % self.n

    def supported_operations(self) -> frozenset[str]:
        return SUPPORTED_OPERATIONS


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _l_function(u: int, n: int) -> int:
    return (u - 1) // n


def homomorphic_add(left: PaillierCiphertext,
                    right: PaillierCiphertext) -> PaillierCiphertext:
    """Eval(+): multiply ciphertexts; adds the underlying plaintexts."""
    if left.public_key.n != right.public_key.n:
        raise HomomorphicEncryptionError("cannot add ciphertexts under different keys")
    return PaillierCiphertext((left.value * right.value) % left.public_key.n_squared,
                              left.public_key)


def homomorphic_scalar_mul(ciphertext: PaillierCiphertext,
                           scalar: int) -> PaillierCiphertext:
    """Eval(k*x): exponentiate the ciphertext by the scalar."""
    if not isinstance(scalar, int):
        raise HomomorphicEncryptionError("scalar multiplication requires an integer scalar")
    if scalar < 0:
        raise HomomorphicEncryptionError("negative scalars require negation support; use sub")
    return PaillierCiphertext(pow(ciphertext.value, scalar, ciphertext.public_key.n_squared),
                              ciphertext.public_key)


def homomorphic_negate(ciphertext: PaillierCiphertext) -> PaillierCiphertext:
    """Eval(-x): modular inverse of the ciphertext."""
    return PaillierCiphertext(pow(ciphertext.value, -1, ciphertext.public_key.n_squared),
                              ciphertext.public_key)


def homomorphic_sub(left: PaillierCiphertext,
                    right: PaillierCiphertext) -> PaillierCiphertext:
    """Eval(x - y) = x + (-y)."""
    return homomorphic_add(left, homomorphic_negate(right))


def homomorphic_multiply(left: PaillierCiphertext,
                         right: PaillierCiphertext) -> PaillierCiphertext:
    """Explicitly reject ciphertext-times-ciphertext: outside the supported algebra."""
    raise HomomorphicEncryptionError(
        "multiplication of two ciphertexts is not supported by the additive "
        "Paillier adapter; a fully homomorphic backend is required for Eval(x*y)"
    )
