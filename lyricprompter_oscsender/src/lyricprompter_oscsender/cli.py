"""`oscsender <command> ...` — thin dispatcher; each command is a `cli(argv) -> int` in its module."""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "serve": ("lyricprompter_oscsender.server.app", "cli", "Run the console web server"),
    "add": ("lyricprompter_oscsender.project", "cli_add", "Create a project from a reference recording + cue file"),
    "list": ("lyricprompter_oscsender.project", "cli_list", "List projects"),
    "check": ("lyricprompter_oscsender.cues", "cli", "Validate a cue file and print its cues"),
    "simulate": ("lyricprompter_oscsender.live.simulate", "cli_simulate",
                 "Track a recording as if it were live and print the cues it fires"),
    "eval": ("lyricprompter_oscsender.live.simulate", "cli_eval",
             "Measure cue timing accuracy on synthetic live takes of a project"),
    "echo": ("lyricprompter_oscsender.latency", "cli_echo", "Answer OSC pings (run on a receiving machine)"),
    "latency": ("lyricprompter_oscsender.latency", "cli_latency", "Measure the OSC round trip to a receiver"),
    "devices": ("lyricprompter_oscsender.live.audio_input", "cli", "List audio input devices"),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print("usage: oscsender <command> [args]\n\ncommands:")
        for name, (_, _, doc) in COMMANDS.items():
            print(f"  {name:<10} {doc}")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    module, func, _ = COMMANDS[argv[0]]
    return int(getattr(importlib.import_module(module), func)(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
