"""OSC 1.0 over UDP: message / bundle codec and a sender to named targets.

Arguments are (type tag, value) pairs. Supported tags:

    i int32   h int64   f float32   d float64   s string (UTF-8)   b blob (bytes)
    T true    F false   N nil       I impulse          (T/F/N/I carry no data: value is ignored)

`OscSender` resolves every target's host when the targets are set, through its `resolver` (a hostname
lookup can block for seconds; the server gives it a cache of addresses it looks up in the background), so
`send` only hands a datagram to the OS and never waits on the network.
"""
from __future__ import annotations

import math
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

OscArg = tuple[str, Any]

INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1
INT64_MIN, INT64_MAX = -(2 ** 63), 2 ** 63 - 1
FLOAT32_MAX = 3.4028234663852886e38
NO_DATA_TAGS = frozenset("TFNI")
TAGS = frozenset("ihfdsb") | NO_DATA_TAGS
BUNDLE_TAG = b"#bundle\x00"
# Not allowed in an OSC address (OSC 1.0 forbids space, '#' and ',' in names; control characters and
# non-ASCII are not printable ASCII). The pattern characters * ? [ ] { } and '//' are allowed, and ','
# inside { } (the pattern {foo,bar}).
_ADDRESS_FORBIDDEN = frozenset(" #,")
MAX_BUNDLE_DEPTH = 32         # nested bundles in a received packet (deeper: OscError, not RecursionError)


class OscError(ValueError):
    """A message that cannot be encoded, or a packet that cannot be decoded."""


def validate_address(address: str) -> str:
    """`address` if it is a valid OSC address (pattern), else OscError."""
    if not isinstance(address, str) or not address.startswith("/"):
        raise OscError(f"an OSC address starts with '/': {address!r}")
    bad, in_braces = set(), False
    for c in address:
        in_braces = (in_braces or c == "{") and c != "}"
        if (c in _ADDRESS_FORBIDDEN and not (c == "," and in_braces)) or not (c.isascii() and c.isprintable()):
            bad.add(c)
    bad = sorted(bad)
    if bad:
        raise OscError(f"OSC address {address!r} contains characters that are not allowed: "
                       + " ".join(repr(c) for c in bad))
    return address


# ----------------------------------------------------------------------------------------- encoding


def _osc_string(s: str) -> bytes:
    try:
        data = s.encode("utf-8")
    except UnicodeEncodeError as e:                      # lone surrogates (e.g. from a JSON "\ud800")
        raise OscError(f"OSC strings must be valid Unicode: {e}") from None
    if b"\x00" in data:
        raise OscError(f"OSC strings cannot contain NUL characters: {s!r}")
    return data + b"\x00" * (4 - len(data) % 4)          # at least one terminating NUL, 4-byte aligned


def _osc_blob(b: bytes) -> bytes:
    return struct.pack(">i", len(b)) + b + b"\x00" * (-len(b) % 4)


def _int(v: Any, lo: int, hi: int, tag: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
        raise OscError(f"type '{tag}' needs an integer, got {v!r}")
    v = int(v)
    if not lo <= v <= hi:
        raise OscError(f"{v} does not fit type '{tag}' ({lo}..{hi})")
    return v


def _float(v: Any, tag: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise OscError(f"type '{tag}' needs a number, got {v!r}")
    v = float(v)
    if tag == "f" and math.isfinite(v) and abs(v) > FLOAT32_MAX:
        raise OscError(f"{v} is too large for type 'f' (float32); use 'd' (float64)")
    return v


def encode_arg(tag: str, value: Any) -> bytes:
    if tag == "i":
        return struct.pack(">i", _int(value, INT32_MIN, INT32_MAX, tag))
    if tag == "h":
        return struct.pack(">q", _int(value, INT64_MIN, INT64_MAX, tag))
    if tag == "f":
        return struct.pack(">f", _float(value, tag))
    if tag == "d":
        return struct.pack(">d", _float(value, tag))
    if tag == "s":
        if not isinstance(value, str):
            raise OscError(f"type 's' needs a string, got {value!r}")
        return _osc_string(value)
    if tag == "b":
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise OscError(f"type 'b' needs bytes, got {type(value).__name__}")
        return _osc_blob(bytes(value))
    if tag in NO_DATA_TAGS:
        return b""
    raise OscError(f"unsupported OSC type tag {tag!r}")


def encode_message(address: str, args: Sequence[OscArg] = ()) -> bytes:
    """One OSC message: address, type tag string, argument data."""
    validate_address(address)
    tags, data = [","], []
    for item in args:
        try:
            tag, value = item
        except (TypeError, ValueError):
            raise OscError(f"an argument is a (type tag, value) pair, got {item!r}") from None
        data.append(encode_arg(tag, value))
        tags.append(tag)
    return _osc_string(address) + _osc_string("".join(tags)) + b"".join(data)


def encode_bundle(messages: Sequence[bytes], timetag: int = 1) -> bytes:
    """A bundle of already encoded messages / bundles (timetag 1 = immediately)."""
    return BUNDLE_TAG + struct.pack(">Q", timetag) + b"".join(struct.pack(">i", len(m)) + m for m in messages)


# ----------------------------------------------------------------------------------------- decoding


def _read_string(buf: bytes, pos: int) -> tuple[str, int]:
    end = buf.find(b"\x00", pos)
    if end < 0:
        raise OscError("unterminated OSC string")
    nxt = pos + ((end - pos) // 4 + 1) * 4
    if nxt > len(buf):
        raise OscError("OSC string padding runs past the end of the packet")
    try:
        return buf[pos:end].decode("utf-8"), nxt
    except UnicodeDecodeError as e:
        raise OscError(f"OSC string is not UTF-8: {e}") from None


def _take(buf: bytes, pos: int, n: int) -> bytes:
    if pos + n > len(buf):
        raise OscError("argument data runs past the end of the packet")
    return buf[pos:pos + n]


def decode_message(packet: bytes) -> tuple[str, list[OscArg]]:
    """(address, [(tag, value), ...]) of one OSC message (not a bundle)."""
    buf = bytes(packet)
    if not buf.startswith(b"/"):
        raise OscError("not an OSC message (does not start with '/')")
    address, pos = _read_string(buf, 0)
    if pos >= len(buf):                      # OSC 1.0 allows a message without a type tag string
        return address, []
    tags, pos = _read_string(buf, pos)
    if not tags.startswith(","):
        raise OscError(f"type tag string must start with ',': {tags!r}")
    args: list[OscArg] = []
    for tag in tags[1:]:
        if tag == "i":
            args.append((tag, struct.unpack(">i", _take(buf, pos, 4))[0]))
            pos += 4
        elif tag == "h":
            args.append((tag, struct.unpack(">q", _take(buf, pos, 8))[0]))
            pos += 8
        elif tag == "f":
            args.append((tag, struct.unpack(">f", _take(buf, pos, 4))[0]))
            pos += 4
        elif tag == "d":
            args.append((tag, struct.unpack(">d", _take(buf, pos, 8))[0]))
            pos += 8
        elif tag == "s":
            s, pos = _read_string(buf, pos)
            args.append((tag, s))
        elif tag == "b":
            n = struct.unpack(">i", _take(buf, pos, 4))[0]
            if n < 0:
                raise OscError("negative blob size")
            args.append((tag, _take(buf, pos + 4, n)))
            pos += 4 + n + (-n % 4)
        elif tag in NO_DATA_TAGS:
            args.append((tag, {"T": True, "F": False, "N": None, "I": None}[tag]))
        else:
            raise OscError(f"unsupported OSC type tag {tag!r}")
    return address, args


def decode_packet(packet: bytes) -> list[tuple[str, list[OscArg]]]:
    """Every message in a packet: a single message, or a (nested) bundle flattened in order."""
    return _decode_packet(bytes(packet), 0)


def _decode_packet(buf: bytes, depth: int) -> list[tuple[str, list[OscArg]]]:
    if not buf.startswith(BUNDLE_TAG):
        return [decode_message(buf)]
    if depth >= MAX_BUNDLE_DEPTH:
        raise OscError(f"OSC bundles nested deeper than {MAX_BUNDLE_DEPTH}")
    if len(buf) < 16:
        raise OscError("truncated OSC bundle")
    out: list[tuple[str, list[OscArg]]] = []
    pos = 16
    while pos < len(buf):
        n = struct.unpack(">i", _take(buf, pos, 4))[0]
        if n <= 0:
            raise OscError("bad bundle element size")
        out.extend(_decode_packet(_take(buf, pos + 4, n), depth + 1))
        pos += 4 + n
    return out


# ----------------------------------------------------------------------------------------- sending


@dataclass
class Target:
    """A receiver of OSC messages, referred to by name in cue files."""
    name: str
    host: str = "127.0.0.1"
    port: int = 9000
    latency_ms: float = 0.0       # this receiver's output delay: its cues are sent this much earlier
    enabled: bool = True
    test_message: str = ""        # cue-file message text sent by Test / the audio loopback ("" = /oscsender/test)


@dataclass
class SendResult:
    ok: bool
    target: str
    error: str | None = None
    t: float = 0.0                # clock time the datagram was handed to the OS
    nbytes: int = 0


def resolve(host: str, port: int) -> tuple[int, tuple]:
    """(address family, socket address) for a UDP destination; IPv4 preferred. Raises OSError."""
    if not 1 <= int(port) <= 65535:
        raise OSError(f"port {port} out of range 1..65535")
    infos = socket.getaddrinfo(host.strip().strip("[]"), int(port), type=socket.SOCK_DGRAM)
    if not infos:
        raise OSError(f"cannot resolve {host!r}")
    infos.sort(key=lambda i: i[0] != socket.AF_INET)
    family, _, _, _, sockaddr = infos[0]
    return family, sockaddr


class OscSender:
    """Thread-safe UDP sender to named targets (one socket per address family)."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter,
                 resolver: Callable[[str, int], tuple[int, tuple]] = resolve):
        self.clock, self._resolver = clock, resolver
        self._lock = threading.Lock()
        self._targets: dict[str, Target] = {}
        self._dest: dict[str, tuple[int, tuple] | None] = {}
        self._resolve_error: dict[str, str | None] = {}
        self._socks: dict[int, socket.socket] = {}
        self._sent: dict[str, int] = {}
        self._errors: dict[str, int] = {}
        self._last_error: dict[str, str | None] = {}

    # -- targets ---------------------------------------------------------------------------------
    def set_targets(self, targets: Iterable[Target]) -> dict[str, str | None]:
        """Replace the targets and resolve their hosts (may block on DNS). Returns name -> resolution
        error (None when resolved). Counters of targets that keep their name are kept."""
        targets = list(targets)
        dest: dict[str, tuple[int, tuple] | None] = {}
        errors: dict[str, str | None] = {}
        for t in targets:
            try:
                dest[t.name], errors[t.name] = self._resolver(t.host, t.port), None
            except (OSError, UnicodeError, ValueError) as e:
                dest[t.name], errors[t.name] = None, f"cannot resolve {t.host}:{t.port}: {e}"
        with self._lock:
            self._targets = {t.name: t for t in targets}
            self._dest, self._resolve_error = dest, errors
        return errors

    @property
    def targets(self) -> dict[str, Target]:
        with self._lock:
            return dict(self._targets)

    def target(self, name: str) -> Target | None:
        with self._lock:
            return self._targets.get(name)

    # -- sending ---------------------------------------------------------------------------------
    def _sock(self, family: int) -> socket.socket:
        s = self._socks.get(family)
        if s is None:
            s = socket.socket(family, socket.SOCK_DGRAM)
            if family == socket.AF_INET:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)   # x.x.x.255 receivers
            self._socks[family] = s
        return s

    def _count(self, name: str, error: str | None) -> None:
        if error is None:
            self._sent[name] = self._sent.get(name, 0) + 1
        else:
            self._errors[name] = self._errors.get(name, 0) + 1
            self._last_error[name] = error

    def send(self, target: str, address: str, args: Sequence[OscArg] = (), *,
             packet: bytes | None = None) -> SendResult:
        """Send one message to a named target. Never raises for network or target problems: the
        result carries the error (unknown / disabled / unresolved target, encoding, OS error)."""
        try:
            data = packet if packet is not None else encode_message(address, args)
        except (OscError, TypeError) as e:                 # TypeError: args is not a sequence of pairs
            return SendResult(False, target, f"cannot encode the message: {e}")
        with self._lock:
            t = self._targets.get(target)
            if t is None:
                err = f"unknown target {target!r}"
            elif not t.enabled:
                err = f"target {target!r} is disabled"
            elif self._dest.get(target) is None:
                err = self._resolve_error.get(target) or f"target {target!r} is not resolved"
            else:
                family, sockaddr = self._dest[target]
                try:
                    now = self.clock()
                    self._sock(family).sendto(data, sockaddr)
                except OSError as e:
                    err = f"send to {t.host}:{t.port} failed: {e}"
                else:
                    self._count(target, None)
                    return SendResult(True, target, None, now, len(data))
            if t is not None:
                self._count(target, err)
            return SendResult(False, target, err, self.clock(), len(data))

    def stats(self) -> dict[str, dict]:
        with self._lock:
            out = {}
            for name, t in self._targets.items():
                dest = self._dest.get(name)
                out[name] = {"sent": self._sent.get(name, 0), "errors": self._errors.get(name, 0),
                             "last_error": self._last_error.get(name),
                             "resolve_error": self._resolve_error.get(name),
                             "resolved": None if dest is None else f"{dest[1][0]}:{dest[1][1]}",
                             "enabled": t.enabled}
            return out

    def close(self) -> None:
        with self._lock:
            socks, self._socks = list(self._socks.values()), {}
        for s in socks:
            s.close()
