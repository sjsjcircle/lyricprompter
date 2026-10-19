"""`lyricprompter <command> ...` — thin dispatcher; each command lives in its module's `cli(argv)`."""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "prep": ("lyricprompter.prep.pipeline", "Separate, parse and align a song -> songs/<id>/"),
    "serve": ("lyricprompter.server.app", "Run the console + prompter web server"),
    "devices": ("lyricprompter.live.audio_input", "List audio input devices"),
    "simulate": ("lyricprompter.live.session", "Track a file as if it were live (prints positions)"),
    "eval-align": ("lyricprompter.eval.align_eval", "Evaluate offline lyrics alignment"),
    "eval-track": ("lyricprompter.eval.track_eval", "Evaluate live tracking (synthetic + real pairs)"),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print("usage: lyricprompter <command> [args]\n\ncommands:")
        for name, (_, doc) in COMMANDS.items():
            print(f"  {name:<11} {doc}")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    import lyricprompter.config  # noqa: F401  (pins model caches before heavy imports)
    module = importlib.import_module(COMMANDS[argv[0]][0])
    return int(module.cli(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
