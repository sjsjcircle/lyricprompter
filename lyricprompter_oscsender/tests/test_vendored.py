"""The tracking path is lyricprompter's, unchanged: the copied modules must equal the originals
(only the import of STYLES differs). Skipped when the parent lyricprompter checkout is not next to us."""
from __future__ import annotations

from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "src" / "lyricprompter_oscsender"
UPSTREAM = Path(__file__).resolve().parents[2] / "src" / "lyricprompter"
COPIED = ["features/__init__.py", "features/extractor.py", "tracking/__init__.py", "tracking/oltw.py",
          "tracking/relocate.py", "tracking/tracker.py", "live/audio_input.py", "audio_io.py"]
# the only edits: (copy line, upstream line)
EDITS = {"tracking/tracker.py": [("from ..config import STYLES", "from ..text.model import STYLES")]}


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="lyricprompter checkout not found next to this project")
@pytest.mark.parametrize("rel", COPIED)
def test_copied_module_matches_lyricprompter(rel):
    ours = (HERE / rel).read_text(encoding="utf-8")
    for mine, theirs in EDITS.get(rel, []):
        assert ours.count(mine) == 1
        ours = ours.replace(mine, theirs)
    assert ours == (UPSTREAM / rel).read_text(encoding="utf-8"), (
        f"{rel} differs from lyricprompter's: copy it again (and re-apply the edits listed in {__file__})")
