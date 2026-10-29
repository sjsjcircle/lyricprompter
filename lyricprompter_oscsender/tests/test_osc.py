from __future__ import annotations

import math
import random
import re
import socket
import struct
import threading

import pytest
from pythonosc.osc_bundle import OscBundle
from pythonosc.osc_bundle_builder import IMMEDIATELY, OscBundleBuilder
from pythonosc.osc_message import OscMessage
from pythonosc.osc_message_builder import OscMessageBuilder

from lyricprompter_oscsender import osc
from lyricprompter_oscsender.osc import (
    MAX_BUNDLE_DEPTH,
    OscError,
    OscSender,
    SendResult,
    Target,
    decode_message,
    decode_packet,
    encode_bundle,
    encode_message,
    resolve,
    validate_address,
)


def f32(v: float) -> float:
    return struct.unpack(">f", struct.pack(">f", v))[0]


# python-osc decodes values the same way we do, except that it does not know 'I' (impulse)
ARGS = [
    ("i", 0), ("i", 1), ("i", -1), ("i", 2 ** 31 - 1), ("i", -(2 ** 31)),
    ("h", 0), ("h", 2 ** 31), ("h", 2 ** 63 - 1), ("h", -(2 ** 63)),
    ("f", 0.0), ("f", 0.5), ("f", -1.25), ("f", f32(0.1)), ("f", osc.FLOAT32_MAX), ("f", -osc.FLOAT32_MAX),
    ("f", f32(1e-45)),
    ("d", 0.0), ("d", 0.1), ("d", 1e300), ("d", -2.0 ** -1074),
    ("s", ""), ("s", "a"), ("s", "abc"), ("s", "abcd"), ("s", "abcde"), ("s", "한글 text"), ("s", "😀"),
    ("b", b""), ("b", b"\x00"), ("b", b"abc"), ("b", b"abcd"), ("b", bytes(range(256))),
    ("T", True), ("F", False), ("N", None),
]


# ----------------------------------------------------------------------------------------- addresses


@pytest.mark.parametrize("address", ["/", "/a", "/a/b/c", "/a/*", "/a?/b", "/a/[1-3]", "/a/[!x]", "/a/{x,y}",
                                     "/a/{x,y}/{1,2,3}", "//a", "/A_b-c.d~!$%&()+;<=>@^`|", "/cue/1/go"])
def test_valid_addresses(address):
    assert validate_address(address) == address


@pytest.mark.parametrize("address", ["", "a", "a/b", " /a", "/a b", "/a#", "/a,b", "/a/{x,y},z", "/a}, {",
                                     "/a\x00", "/a\n", "/a\x7f", "/é", "/한", None, 5, b"/a"])
def test_invalid_addresses(address):
    with pytest.raises(OscError):
        validate_address(address)


# ----------------------------------------------------------------------------------------- codec


@pytest.mark.parametrize("arg", ARGS + [("I", None)], ids=lambda a: f"{a[0]}:{str(a[1])[:12]}")
def test_round_trip_every_tag(arg):
    packet = encode_message("/test", [arg])
    assert len(packet) % 4 == 0
    assert decode_message(packet) == ("/test", [arg])
    assert decode_packet(packet) == [("/test", [arg])]


def test_round_trip_all_args_at_once():
    args = ARGS + [("I", None)]
    assert decode_message(encode_message("/many/args", args)) == ("/many/args", args)


def test_no_data_tags_ignore_value_and_decode_canonically():
    packet = encode_message("/x", [("T", 0), ("F", "anything"), ("N", 5), ("I", True)])
    assert packet == encode_message("/x", [("T", None), ("F", None), ("N", None), ("I", None)])
    assert decode_message(packet)[1] == [("T", True), ("F", False), ("N", None), ("I", None)]


def test_float32_rounding_and_specials():
    (_, v), = decode_message(encode_message("/x", [("f", 0.1)]))[1]
    assert v == f32(0.1) and v != 0.1
    (_, v), = decode_message(encode_message("/x", [("f", math.inf)]))[1]
    assert v == math.inf
    (_, v), = decode_message(encode_message("/x", [("d", math.nan)]))[1]
    assert math.isnan(v)


def test_integral_floats_accepted_for_int_tags():
    assert decode_message(encode_message("/x", [("i", 3.0), ("h", -2.0)]))[1] == [("i", 3), ("h", -2)]


def test_string_padding_lengths():
    for n in range(13):
        packet = encode_message("/" + "a" * n, [("s", "b" * n)])
        assert len(packet) % 4 == 0
        assert decode_message(packet) == ("/" + "a" * n, [("s", "b" * n)])


def test_message_without_type_tags():
    assert decode_message(b"/abc\x00\x00\x00\x00") == ("/abc", [])
    assert decode_message(encode_message("/abc")) == ("/abc", [])
    assert encode_message("/abc") == b"/abc\x00\x00\x00\x00,\x00\x00\x00"


@pytest.mark.parametrize("address, args, fragment", [
    ("/x", [("i", 2 ** 31)], "does not fit type 'i'"),
    ("/x", [("i", -(2 ** 31) - 1)], "does not fit type 'i'"),
    ("/x", [("h", 2 ** 63)], "does not fit type 'h'"),
    ("/x", [("i", 1.5)], "needs an integer"),
    ("/x", [("i", True)], "needs an integer"),
    ("/x", [("i", "1")], "needs an integer"),
    ("/x", [("h", None)], "needs an integer"),
    ("/x", [("f", 3.5e38)], "too large for type 'f'"),
    ("/x", [("f", -1e39)], "too large for type 'f'"),
    ("/x", [("f", "1.0")], "needs a number"),
    ("/x", [("d", False)], "needs a number"),
    ("/x", [("s", 5)], "needs a string"),
    ("/x", [("s", "a\x00b")], "NUL"),
    ("/x", [("s", "\ud800")], "valid Unicode"),
    ("/x", [("b", "text")], "needs bytes"),
    ("/x", [("x", 1)], "unsupported OSC type tag"),
    ("/x", [("ii", 1)], "unsupported OSC type tag"),
    ("/x", [5], "(type tag, value) pair"),
    ("/x", [("i", 1, 2)], "(type tag, value) pair"),
    ("x", [], "starts with '/'"),
    ("/a b", [], "not allowed"),
])
def test_encode_errors(address, args, fragment):
    with pytest.raises(OscError, match=re.escape(fragment)):
        encode_message(address, args)


def test_blob_accepts_bytes_likes():
    for v in (bytearray(b"ab"), memoryview(b"ab")):
        assert decode_message(encode_message("/b", [("b", v)]))[1] == [("b", b"ab")]


@pytest.mark.parametrize("packet, fragment", [
    (b"", "does not start with '/'"),
    (b"abc\x00", "does not start with '/'"),
    (b"/abc", "unterminated OSC string"),
    (b"/ab\x00\x00", "padding runs past the end"),
    (b"/a\x00\x00i\x00\x00\x00", "must start with ','"),
    (b"/a\x00\x00,i\x00\x00", "runs past the end"),
    (b"/a\x00\x00,i\x00\x00\x00\x00\x01", "runs past the end"),
    (b"/a\x00\x00,h\x00\x00\x00\x00\x00\x00\x00\x01", "runs past the end"),
    (b"/a\x00\x00,d\x00\x00\x00\x00\x00\x00", "runs past the end"),
    (b"/a\x00\x00,s\x00\x00abcd", "unterminated"),
    (b"/a\x00\x00,s\x00\x00\xff\xfe\x00\x00", "not UTF-8"),
    (b"/\xff\x00\x00,\x00\x00\x00", "not UTF-8"),
    (b"/a\x00\x00,b\x00\x00\xff\xff\xff\xff", "negative blob size"),
    (b"/a\x00\x00,b\x00\x00\x00\x00\x00\x08ab\x00\x00", "runs past the end"),
    (b"/a\x00\x00,x\x00\x00", "unsupported OSC type tag"),
    (b"/a\x00\x00,r\x00\x00\x00\x00\x00\x00", "unsupported OSC type tag"),
    (b"#bundle\x00\x00\x00\x00", "truncated OSC bundle"),
    (b"#bundle\x00" + b"\x00" * 7 + b"\x01" + b"\x00\x00\x00\x00", "bad bundle element size"),
    (b"#bundle\x00" + b"\x00" * 7 + b"\x01" + b"\xff\xff\xff\xf0", "bad bundle element size"),
    (b"#bundle\x00" + b"\x00" * 7 + b"\x01" + b"\x00\x00\x00\x10/a\x00\x00", "runs past the end"),
    (b"#bundle\x00" + b"\x00" * 7 + b"\x01" + b"\x00\x00", "runs past the end"),
])
def test_malformed_packets(packet, fragment):
    with pytest.raises(OscError, match=fragment):
        decode_packet(packet)


def test_corrupted_packets_only_raise_osc_error():
    rng = random.Random(3)
    valid = [encode_message("/x/y", ARGS[i:i + 5]) for i in range(0, len(ARGS), 5)]
    valid.append(encode_bundle([valid[0], encode_bundle(valid[1:3])]))
    for _ in range(3000):
        p = bytearray(rng.choice(valid))
        for _ in range(rng.randint(1, 4)):
            op = rng.random()
            if op < 0.4 and p:
                p[rng.randrange(len(p))] = rng.randrange(256)
            elif op < 0.7 and p:
                del p[rng.randrange(len(p)):]
            else:
                p.insert(rng.randrange(len(p) + 1), rng.randrange(256))
        try:
            decode_packet(bytes(p))
        except OscError:
            pass


# ----------------------------------------------------------------------------------------- python-osc


@pytest.mark.parametrize("arg", ARGS, ids=lambda a: f"{a[0]}:{str(a[1])[:12]}")
def test_ours_decoded_by_python_osc(arg):
    msg = OscMessage(encode_message("/po/x", [arg]))
    assert msg.address == "/po/x"
    assert msg.params == [arg[1]]


def test_ours_decoded_by_python_osc_all_args():
    msg = OscMessage(encode_message("/po/all", ARGS))
    assert msg.params == [v for _, v in ARGS]


@pytest.mark.parametrize("arg", ARGS, ids=lambda a: f"{a[0]}:{str(a[1])[:12]}")
def test_python_osc_decoded_by_ours_and_bytes_equal(arg):
    if arg == ("b", b""):
        pytest.skip("python-osc cannot build an empty blob")
    b = OscMessageBuilder("/po/x")
    b.add_arg(arg[1], arg[0])
    dgram = b.build().dgram
    assert decode_message(dgram) == ("/po/x", [arg])
    assert dgram == encode_message("/po/x", [arg])


def test_python_osc_inferred_types():
    b = OscMessageBuilder("/infer")
    for v in (1, 2 ** 40, 0.25, "s", b"\x01", True, False, None):
        b.add_arg(v)
    assert decode_message(b.build().dgram)[1] == [("i", 1), ("h", 2 ** 40), ("f", 0.25), ("s", "s"),
                                                  ("b", b"\x01"), ("T", True), ("F", False), ("N", None)]
    assert decode_message(OscMessageBuilder("/empty").build().dgram) == ("/empty", [])


def test_bundles_flatten_in_order():
    m1 = encode_message("/one", [("i", 1)])
    m2 = encode_message("/two", [("s", "x")])
    m3 = encode_message("/three", [("T", True)])
    packet = encode_bundle([m1, encode_bundle([m2, encode_bundle([m3])]), m1], timetag=12345)
    assert decode_packet(packet) == [("/one", [("i", 1)]), ("/two", [("s", "x")]), ("/three", [("T", True)]),
                                     ("/one", [("i", 1)])]
    assert struct.unpack(">Q", packet[8:16])[0] == 12345
    assert decode_packet(encode_bundle([])) == []
    with pytest.raises(OscError):
        decode_message(packet)                          # a bundle is not a message


def test_bundle_cross_check_with_python_osc():
    m1, m2 = encode_message("/a", [("i", 7)]), encode_message("/b", [("d", 0.5), ("s", "z")])
    ours = encode_bundle([m1, encode_bundle([m2])])
    po = OscBundle(ours)
    assert po.timestamp == IMMEDIATELY and po.num_contents == 2
    assert po.content(0).address == "/a" and po.content(0).params == [7]
    inner = po.content(1)
    assert isinstance(inner, OscBundle) and inner.content(0).params == [0.5, "z"]

    builder = OscBundleBuilder(IMMEDIATELY)
    builder.add_content(OscMessage(m1))
    nested = OscBundleBuilder(IMMEDIATELY)
    nested.add_content(OscMessage(m2))
    builder.add_content(nested.build())
    theirs = builder.build().dgram
    assert decode_packet(theirs) == [("/a", [("i", 7)]), ("/b", [("d", 0.5), ("s", "z")])]
    assert theirs == ours


def test_bundle_nesting_limit():
    p = encode_message("/deep")
    for _ in range(MAX_BUNDLE_DEPTH):
        p = encode_bundle([p])
    assert decode_packet(p) == [("/deep", [])]
    with pytest.raises(OscError, match="nested deeper"):
        decode_packet(encode_bundle([p]))
    for _ in range(2000):                               # far beyond the recursion limit: still an OscError
        p = encode_bundle([p])
    with pytest.raises(OscError):
        decode_packet(p)


# ----------------------------------------------------------------------------------------- resolve


def test_resolve():
    family, addr = resolve("127.0.0.1", 9000)
    assert family == socket.AF_INET and addr == ("127.0.0.1", 9000)
    family, addr = resolve(" 127.0.0.1 ", 1)
    assert addr[0] == "127.0.0.1"
    if socket.has_ipv6:
        family, addr = resolve("[::1]", 9000)
        assert family == socket.AF_INET6 and addr[:2] == ("::1", 9000)
    for port in (0, 65536, -1):
        with pytest.raises(OSError):
            resolve("127.0.0.1", port)


# ----------------------------------------------------------------------------------------- sender


class Clock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


@pytest.fixture
def receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2.0)
    yield sock
    sock.close()


@pytest.fixture
def sender():
    s = OscSender(clock=Clock())
    yield s
    s.close()


def recv(sock: socket.socket) -> list:
    data, _ = sock.recvfrom(65536)
    return decode_packet(data)


def test_send_to_local_udp(receiver, sender):
    port = receiver.getsockname()[1]
    assert sender.set_targets([Target("main", "127.0.0.1", port)]) == {"main": None}
    args = [("i", 3), ("f", 0.5), ("s", "intro.mp4"), ("T", True), ("N", None), ("h", 2 ** 40), ("d", 0.1)]
    r = sender.send("main", "/clip/play", args)
    assert r == SendResult(True, "main", None, 101.0, len(encode_message("/clip/play", args)))
    assert recv(receiver) == [("/clip/play", args)]
    msg = OscMessage(receiver.recvfrom(65536)[0]) if sender.send("main", "/po", [("i", 9)]).ok else None
    assert msg is not None and msg.address == "/po" and msg.params == [9]
    st = sender.stats()["main"]
    assert st == {"sent": 2, "errors": 0, "last_error": None, "resolve_error": None,
                  "resolved": f"127.0.0.1:{port}", "enabled": True}


def test_send_raw_packet(receiver, sender):
    sender.set_targets([Target("main", "127.0.0.1", receiver.getsockname()[1])])
    bundle = encode_bundle([encode_message("/a"), encode_message("/b", [("i", 1)])])
    r = sender.send("main", "/ignored", packet=bundle)
    assert r.ok and r.nbytes == len(bundle)
    assert recv(receiver) == [("/a", []), ("/b", [("i", 1)])]


def test_send_ipv6(sender):
    if not socket.has_ipv6:
        pytest.skip("no IPv6")
    sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    try:
        sock.bind(("::1", 0))
    except OSError:
        sock.close()
        pytest.skip("::1 is not available")
    sock.settimeout(2.0)
    try:
        sender.set_targets([Target("v6", "[::1]", sock.getsockname()[1])])
        assert sender.send("v6", "/six", [("s", "x")]).ok
        assert recv(sock) == [("/six", [("s", "x")])]
    finally:
        sock.close()


def test_failures_never_raise_and_are_counted(receiver):
    def resolver(host: str, port: int):
        if host == "nowhere":
            raise OSError("no such host")
        return resolve(host, port)

    s = OscSender(clock=Clock(), resolver=resolver)
    port = receiver.getsockname()[1]
    errors = s.set_targets([Target("ok", "127.0.0.1", port), Target("off", "127.0.0.1", port, enabled=False),
                            Target("lost", "nowhere", 9000), Target("badport", "127.0.0.1", 0)])
    assert errors["ok"] is None and "cannot resolve nowhere:9000" in errors["lost"]
    assert errors.get("badport")

    r = s.send("unknown", "/a")
    assert not r.ok and r.target == "unknown" and "unknown target 'unknown'" in r.error
    r = s.send("off", "/a")
    assert not r.ok and "disabled" in r.error
    r = s.send("lost", "/a")
    assert not r.ok and "cannot resolve" in r.error
    r = s.send("badport", "/a")
    assert not r.ok
    r = s.send("ok", "no slash")
    assert not r.ok and "cannot encode" in r.error
    r = s.send("ok", "/a", [("i", 2 ** 40)])
    assert not r.ok and "cannot encode" in r.error
    r = s.send("ok", "/a", None)                                  # not a sequence: still no exception
    assert not r.ok and "cannot encode" in r.error
    r = s.send("ok", "/a", [("s", "x" * 70000)])                  # too large for one UDP datagram
    assert not r.ok and "failed" in r.error and r.nbytes > 65507

    st = s.stats()
    assert "unknown" not in st
    assert st["off"]["errors"] == 1 and st["off"]["enabled"] is False
    assert st["lost"]["errors"] == 1 and st["lost"]["resolved"] is None and "nowhere" in st["lost"]["resolve_error"]
    assert st["ok"]["errors"] == 1 and st["ok"]["sent"] == 0 and "failed" in st["ok"]["last_error"]
    assert s.send("ok", "/fine").ok
    assert s.stats()["ok"]["sent"] == 1 and s.stats()["ok"]["last_error"] is not None   # last error is kept
    assert recv(receiver) == [("/fine", [])]
    s.close()


def test_unresolvable_send_error_on_bad_sockaddr():
    s = OscSender(clock=Clock(), resolver=lambda host, port: (socket.AF_INET, ("256.0.0.1", port)))
    s.set_targets([Target("weird", "x", 9)])
    r = s.send("weird", "/a")
    assert not r.ok and "send to x:9 failed" in r.error and r.t > 0
    s.close()


def test_targets_and_counters_survive_set_targets(receiver, sender):
    port = receiver.getsockname()[1]
    sender.set_targets([Target("a", "127.0.0.1", port), Target("b", "127.0.0.1", port)])
    for _ in range(3):
        assert sender.send("a", "/x").ok
    assert sender.send("b", "/x").ok
    sender.set_targets([Target("a", "127.0.0.1", port, latency_ms=20.0)])
    assert set(sender.targets) == {"a"} and sender.target("a").latency_ms == 20.0 and sender.target("b") is None
    assert sender.stats()["a"]["sent"] == 3 and "b" not in sender.stats()
    assert not sender.send("b", "/x").ok


def test_close_then_send_reopens(receiver, sender):
    sender.set_targets([Target("a", "127.0.0.1", receiver.getsockname()[1])])
    assert sender.send("a", "/1").ok
    sender.close()
    assert sender.send("a", "/2").ok
    assert [recv(receiver)[0][0] for _ in range(2)] == ["/1", "/2"]


def test_concurrent_sends_are_counted(receiver):
    s = OscSender()
    s.set_targets([Target("a", "127.0.0.1", receiver.getsockname()[1])])
    results: list[bool] = []

    def worker():
        for i in range(50):
            results.append(s.send("a", "/c", [("i", i)]).ok)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 200 and all(results) and s.stats()["a"]["sent"] == 200
    s.close()
