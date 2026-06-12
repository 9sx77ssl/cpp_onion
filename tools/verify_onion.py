#!/usr/bin/env python3
"""Independent oracle for cpp_onion results.

Re-derives the public key from the secret scalar using a from-scratch
pure-Python ed25519 (no shared code with the C++ implementation or
libsodium), recomputes the onion address, and checks it against the
hostname file. Exit code 0 = OK, 1 = failure.

Usage:
  verify_onion.py <result-dir>     verify a generated key directory
  verify_onion.py --self-test      run internal KATs (RFC 8032 TEST1)
"""

import base64
import hashlib
import sys
from pathlib import Path

P = 2**255 - 19


def inv(x: int) -> int:
    return pow(x, P - 2, P)


D = -121665 * inv(121666) % P
BY = 4 * inv(5) % P


def xrecover(y: int) -> int:
    xx = (y * y - 1) * inv(D * y * y + 1) % P
    x = pow(xx, (P + 3) // 8, P)
    if (x * x - xx) % P != 0:
        x = x * pow(2, (P - 1) // 4, P) % P
    if x % 2 != 0:
        x = P - x
    return x


BX = xrecover(BY)
B = (BX, BY, 1, BX * BY % P)  # extended coordinates (X, Y, Z, T)
IDENTITY = (0, 1, 1, 0)


def pt_add(p, q):
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = (y1 - x1) * (y2 - x2) % P
    b = (y1 + x1) * (y2 + x2) % P
    c = 2 * t1 * t2 * D % P
    d = 2 * z1 * z2 % P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % P, g * h % P, f * g % P, e * h % P)


def scalarmult(p, e: int):
    q = IDENTITY
    while e > 0:
        if e & 1:
            q = pt_add(q, p)
        p = pt_add(p, p)
        e >>= 1
    return q


def encodepoint(p) -> bytes:
    x, y, z, _ = p
    zi = inv(z)
    x, y = x * zi % P, y * zi % P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def onion_address(pub: bytes) -> str:
    chk = hashlib.sha3_256(b".onion checksum" + pub + b"\x03").digest()[:2]
    return base64.b32encode(pub + chk + b"\x03").decode().lower() + ".onion"


SECRET_TAG = b"== ed25519v1-secret: type0 ==" + b"\x00" * 3
PUBLIC_TAG = b"== ed25519v1-public: type0 ==" + b"\x00" * 3


def fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_dir(result_dir: Path):
    secret = (result_dir / "hs_ed25519_secret_key").read_bytes()
    public = (result_dir / "hs_ed25519_public_key").read_bytes()
    hostname = (result_dir / "hostname").read_text().strip()

    if len(secret) != 96 or secret[:32] != SECRET_TAG:
        fail("malformed hs_ed25519_secret_key")
    if len(public) != 64 or public[:32] != PUBLIC_TAG:
        fail("malformed hs_ed25519_public_key")

    a = int.from_bytes(secret[32:64], "little")
    if a & 7:
        fail("scalar low 3 bits not clear (clamping violated)")

    derived_pub = encodepoint(scalarmult(B, a))
    if derived_pub != public[32:]:
        fail("public key file does not equal scalar*B")
    if onion_address(derived_pub) != hostname:
        fail("hostname does not match address derived from key")
    print(f"OK: {hostname}")


def self_test():
    seed = bytes.fromhex(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    pub = encodepoint(scalarmult(B, a))
    assert pub.hex() == ("d75a980182b10ab7d54bfed3c964073a"
                         "0ee172f3daa62325af021a68f707511a"), "RFC8032 TEST1 pubkey"
    assert onion_address(pub) == (
        "25njqamcweflpvkl73j4szahhihoc4xt3ktcgjnpaingr5yhkenl5sid.onion"
    ), "TEST1 onion address"
    print("self-test OK")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "--self-test":
        self_test()
    else:
        check_dir(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
