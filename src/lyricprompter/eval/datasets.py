"""Alignment evaluation datasets behind one structure: audio paths, lyrics text and ground-truth onsets.

- edelweiss (en, word): ref repo Edelweiss.lrc, BS-RoFormer stems in data/stems/edelweiss
- edelweiss_short (en, word): Edelweiss_short.lrc (second performance, first 4 lines)
- kkotbat (ko, syllable): 꽃밭에서 score CSV onsets (offset x 0.5 s) warped to _ref.wav (offline DTW path)
- jamendo_* (en, word): JamendoLyrics word onsets / ends, htdemucs vocal stems
- tts_multilingual (ko/ja/en, line): research clip, mix = sil+ko+gap+ja+gap+en+sil (recovered from the clips);
  tts_multilingual_long: the same clip tiled 13x (240 s)
- tts_ko / tts_ja / tts_en / tts_mixed (word = phrase): generated with macOS `say`, per-phrase clips at known
  offsets; tts_backing: "(...)" call-and-response backing parts in a second voice

Reference-repo data is private / unlicensed: read for evaluation only (never redistributed).
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from .. import config

STEMS_DIR = config.DATA_DIR / "stems"
TESTSETS_DIR = config.DATA_DIR / "research-testsets"
JAMENDO_DIR = TESTSETS_DIR / "jamendo"
TTS_RESEARCH_DIR = TESTSETS_DIR / "tts-multilingual"
TTS_GEN_DIR = TESTSETS_DIR / "tts-gen"
WP_FPS = 25.0                        # offline warping-path CSVs: 16 kHz / hop 640


@dataclass
class GTToken:
    text: str                        # surface text of the GT unit (word / syllable / line)
    t: float                         # onset (s, on the dataset's audio)
    line: int                        # GT line index (groups tokens for line-level stats)
    end: float | None = None         # offset when the source has one
    backing: bool = False            # token is a backing vocal "(...)"


@dataclass
class AlignDataset:
    name: str
    mix: Path
    vocals: Path
    lyrics: str                      # lyrics text exactly as a user would paste it
    gt: list[GTToken]
    granularity: str                 # word | syllable | line
    language: str                    # en | ko | ja | mixed
    default_lang: str = "auto"
    notes: str = ""
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------- text helpers
def norm_chars(text: str) -> str:
    """Matching form: NFKC, case-folded, letters and digits only."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if unicodedata.category(c)[0] in "LN")


def parse_lrc(path: Path) -> list[list[tuple[float, str, bool]]]:
    """Reference-repo LRC flavour '[mm:ss.xx]tok  <mm:ss.xx>tok  <mm:ss.xx> tok' -> per lyric line a list of
    (time, token, starts_word); a tag whose text begins with a space starts a new word. Blank lines are
    returned as empty lists (stanza breaks); header tags are skipped."""
    out: list[list[tuple[float, str, bool]]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            if out and out[-1]:
                out.append([])
            continue
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", raw)
        if not m:
            continue
        mi, se, rest = m.groups()
        rest = rest.strip()
        if not rest.startswith("<"):
            rest = f"<{mi}:{se}> " + rest
        toks = [(int(a) * 60 + float(b), txt, txt.startswith(" ") or i == 0)
                for i, (a, b, txt) in enumerate(re.findall(r"<(\d+):(\d+(?:\.\d+)?)>([^<]*)", rest))]
        out.append([(t, txt.strip(), ws) for t, txt, ws in toks if txt.strip()])
    while out and not out[-1]:
        out.pop()
    return out


def lrc_words(path: Path) -> tuple[str, list[GTToken]]:
    """LRC -> (lyrics text with blank lines between stanzas, word-onset GT)."""
    text_lines, gt = [], []
    li = 0
    for toks in parse_lrc(path):
        if not toks:
            text_lines.append("")
            continue
        words: list[list] = []
        for t, txt, ws in toks:
            if ws or not words:
                words.append([t, txt])
            else:
                words[-1][1] += txt
        text_lines.append(" ".join(w for _, w in words))
        gt += [GTToken(w, float(t), li) for t, w in words]
        li += 1
    return "\n".join(text_lines), gt


def assign_lines(tokens: list[str], lyrics: str) -> list[int]:
    """Line index (non-blank lyric lines) of every token, by character-level sequence alignment."""
    lines = [ln for ln in lyrics.splitlines() if ln.strip()]
    ref_chars, ref_line = [], []
    for i, ln in enumerate(lines):
        c = norm_chars(ln)
        ref_chars.append(c)
        ref_line += [i] * len(c)
    ref = "".join(ref_chars)
    tok_chars = [norm_chars(t) for t in tokens]
    hyp = "".join(tok_chars)
    m = {}
    for a, b, n in SequenceMatcher(None, hyp, ref, autojunk=False).get_matching_blocks():
        for k in range(n):
            m[a + k] = b + k
    out, pos, last = [], 0, 0
    for c in tok_chars:
        hit = next((m[pos + k] for k in range(len(c)) if pos + k in m), None)
        last = ref_line[hit] if hit is not None else last
        out.append(last)
        pos += len(c)
    return out


# ----------------------------------------------------------------------------------------- loaders
def _ref_resources() -> Path:
    ref = config.ref_repo_dir()
    if ref is None:
        raise FileNotFoundError("reference repo not found (set LLT_REF_REPO or clone it into "
                                "./live-lyrics-tracking)")
    return ref / "resources"


def _stems(name: str) -> tuple[Path, Path]:
    d = STEMS_DIR / name
    return d / "mix.wav", d / "vocals.wav"


def load_edelweiss(short: bool = False) -> AlignDataset:
    res = _ref_resources()
    name = "edelweiss_short" if short else "edelweiss"
    lyrics, gt = lrc_words(res / "lyrics" / ("Edelweiss_short.lrc" if short else "Edelweiss.lrc"))
    mix, voc = _stems(name)
    return AlignDataset(name, mix, voc, lyrics, gt, "word", "en", "en",
                        notes="LRC word onsets (syllable tags merged into words); BS-RoFormer vocals")


def kkotbat_gt(score_csv: Path, wp_csv: Path, sec_per_quarter: float = 0.5) -> list[tuple[float, str]]:
    """Score syllable onsets (offset x sec_per_quarter) mapped to _ref.wav time via the offline warping path
    (score_frame, ref_frame) at 25 fps, keeping only steps that advance both axes."""
    with open(score_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["lyric"]]
    wp = np.loadtxt(wp_csv, delimiter=",")
    keep = np.r_[True, (np.diff(wp[:, 0]) > 0) & (np.diff(wp[:, 1]) > 0)]
    keep[-1] = True
    wp = wp[keep] / WP_FPS
    t_score = np.array([float(r["offset"]) * sec_per_quarter for r in rows])
    t_ref = np.interp(t_score, wp[:, 0], wp[:, 1])
    return [(float(t), unicodedata.normalize("NFC", r["lyric"])) for t, r in zip(t_ref, rows)]


def load_kkotbat() -> AlignDataset:
    res = _ref_resources()
    lyrics = (res / "lyrics" / "꽃밭에서_lyrics.txt").read_text(encoding="utf-8")
    syl = kkotbat_gt(res / "annots" / "꽃밭에서_score.csv", res / "alignments" / "꽃밭에서_offline.csv")
    li = assign_lines([s for _, s in syl], lyrics)
    gt = [GTToken(s, t, i) for (t, s), i in zip(syl, li)]
    mix, voc = _stems("kkotbat_ref")
    return AlignDataset("kkotbat", mix, voc, lyrics, gt, "syllable", "ko", "ko",
                        notes="pseudo-GT: score onsets warped by the reference repo's offline DTW path "
                              "(its own error is unknown, probably ~0.1 s)")


JAMENDO_SONGS = {"jamendo_hila": "HILA_-_Give_Me_the_Same", "jamendo_embers": "Avercage_-_Embers",
                 "jamendo_rxbyn": "Rxbyn_-_Bad_Side", "jamendo_cortez": "Cortez_-_Feel__Stripped_"}


def load_jamendo(name: str) -> AlignDataset:
    song = JAMENDO_SONGS[name]
    lyrics = (JAMENDO_DIR / "lyrics" / f"{song}.txt").read_text(encoding="utf-8")
    words = [(li, w) for li, ln in enumerate(x for x in lyrics.splitlines() if x.strip()) for w in ln.split()]
    with open(JAMENDO_DIR / "words" / f"{song}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(words):
        raise ValueError(f"{song}: {len(rows)} GT words vs {len(words)} lyric words")
    gt = [GTToken(w, float(r["word_start"]), li, float(r["word_end"])) for (li, w), r in zip(words, rows)]
    vocals = JAMENDO_DIR / "sep" / "htdemucs" / song / "vocals.wav"
    return AlignDataset(name, JAMENDO_DIR / "mp3" / f"{song}.mp3", vocals, lyrics, gt, "word", "en", "en",
                        notes="JamendoLyrics word onsets/ends; htdemucs vocal stem")


# research clip: mix = sil(1.5) + ko + gap(3) + ja + gap(3) + en + sil(1.5); speech onsets inside the clips
# measured from the clips (ko line 2 starts after the TTS sentence-boundary digital silence at 2.364 s).
_KO_S, _JA_S = 69499 / 16000, 43949 / 16000          # clip lengths (samples / sr)
_TTS_RESEARCH_ONSETS = (1.5 + 0.010, 1.5 + 2.364, 1.5 + _KO_S + 3.0 + 0.090, 1.5 + _KO_S + 3.0 + _JA_S + 3.0)


def load_tts_research(long: bool = False) -> AlignDataset:
    d = TTS_RESEARCH_DIR
    base = [ln for ln in (d / "lyrics.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    period = 294756 / 16000                          # len(mix.wav) / sr
    reps = 13 if long else 1
    lines = base * reps
    gt = [GTToken(lines[k * 4 + i], round(k * period + t, 4), k * 4 + i)
          for k in range(reps) for i, t in enumerate(_TTS_RESEARCH_ONSETS)]
    wav = d / ("long240.wav" if long else "mix.wav")
    name = "tts_multilingual_long" if long else "tts_multilingual"
    return AlignDataset(name, wav, wav, "\n".join(lines), gt, "line", "mixed",
                        notes="macOS TTS speech; GT line onsets recovered from clip offsets "
                              "(+-0.03 s for the 2nd Korean line)")


# ----------------------------------------------------------------------------------------- generated TTS
VOICES = {"ko": "Yuna", "ja": "Kyoko", "en": "Samantha"}
BACKING_VOICES = {"ko": "Grandma (한국어(대한민국))", "ja": "Grandma (일본어(일본))", "en": "Fred"}
TTS_SETS: dict[str, tuple[str, list[str]]] = {
    "tts_ko": ("ko", [
        "하늘 위로 날아가는 새처럼", "우리의 노래는 끝나지 않아", "바람이 불어오는 언덕 위에서",
        "너의 목소리가 들려와", "작은 불빛 하나 켜 두고", "오늘 밤 별들이 빛나네",
        "꿈속에서 다시 만나요", "시간이 멈춘 듯한 거리", "햇살 가득한 아침이 오면",
        "함께 걷던 길을 기억해", "마음속 깊이 간직할게", "내일은 더 좋은 날이 될 거야"]),
    "tts_ja": ("ja", [
        "夜明けの 空に 願いを 込めて", "遠い 街の 灯りが 揺れている", "君と 歩いた 坂道を 思い出す",
        "冷たい 風が 頬を 撫でていく", "小さな 約束を 胸に 抱いて", "青い 海の 向こうへ 旅立とう",
        "静かな 夜に 星が 降る", "心の 奥で 響く 声", "春の 雨が 花びらを 濡らす",
        "今日も 明日も 君を 想う", "白い 雲が 流れる 午後", "名前を 呼べば 振り向いて",
        "季節は 巡り 時計が 進む", "光の 中で 踊り 続ける", "古い 手紙を 読み 返す",
        "世界の 果てまで 走り出せ", "優しい 歌が 聞こえる", "窓辺に 置いた 一輪の 花",
        "永遠に 続く 物語", "何度でも 立ち上がる"]),
    "tts_en": ("en", [
        "Morning light is breaking through the clouds", "Every step I take brings me closer home",
        "Whisper softly when the evening falls", "We were dancing on the edge of time",
        "Silver rivers running to the sea", "Paper boats are drifting down the stream",
        "Counting stars until the sun comes up", "Every heartbeat echoes in the dark",
        "Open windows letting summer in", "Thunder rolling over distant hills",
        "Remember how we used to sing", "Carry me across the frozen lake"]),
    "tts_mixed": ("mixed", [
        "사랑해 Baby 君が 好き", "Hello 내 사랑 また 会おう", "오늘도 Shine 輝いて",
        "星空の 下 We sing together", "다시 한번 One more time", "君の 笑顔 내 맘속에",
        "Dreaming 夢の 中で 너를 찾아", "강물처럼 흘러가는 時間", "Tonight 우리 함께 踊ろう",
        "風に 乗って Fly away", "눈을 감으면 君が 見える", "Forever 永遠に 함께해"]),
    "tts_backing": ("mixed", [
        "I keep on running (running) through the night", "Hold me closer (closer) never let go", "(ooh ooh)",
        "Shining brightly like the stars (the stars) above", "We will carry on (carry on) till the end",
        "(yeah yeah yeah)", "Call my name (call my name) when you are lonely", "너를 사랑해 (사랑해) 영원히",
        "Dancing in the rain (in the rain)", "(oh oh)", "Never giving up (no no) never backing down",
        "Every morning (every morning) you will see"]),
}


def _script_lang(chunk: str) -> str:
    if re.search(r"[가-힣]", chunk):
        return "ko"
    if re.search(r"[぀-ヿ㐀-鿿]", chunk):
        return "ja"
    return "en"


def _synth(text: str, voice: str, sr: int = 16000) -> np.ndarray:
    """macOS `say` -> float32 mono at sr, leading/trailing silence trimmed (-40 dB re. clip peak, 10 ms)."""
    import soundfile as sf
    import soxr
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "x.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(out), text], check=True, capture_output=True)
        y, ysr = sf.read(out, dtype="float32", always_2d=True)
    y = soxr.resample(y.mean(axis=1), ysr, sr).astype(np.float32)
    hop = sr // 100
    n = len(y) // hop
    db = 10 * np.log10(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    act = np.flatnonzero(db > db.max() - 40.0)
    if len(act) == 0:
        return y
    return y[act[0] * hop: (act[-1] + 1) * hop]


def build_tts_set(name: str, out_root: Path = TTS_GEN_DIR, seed: int = 7, sr: int = 16000) -> Path:
    """Deterministically synthesise one generated TTS set: per-phrase clips at known offsets, short gaps
    between phrases, longer gaps between lines and an 'instrumental' break every 5 lines, plus a -65 dBFS
    noise floor. Writes vocals.wav, lyrics.txt and gt.json under out_root/name."""
    import soundfile as sf
    if shutil.which("say") is None:
        raise RuntimeError("macOS `say` is required to generate TTS test sets")
    lang, lines = TTS_SETS[name]
    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    gt: list[dict] = []
    t = 1.0
    pieces.append(np.zeros(int(t * sr), np.float32))
    for li, line in enumerate(lines):
        for chunk in re.findall(r"\([^)]*\)|\S+", line):
            backing = chunk.startswith("(")
            text = chunk.strip("()")
            voice = (BACKING_VOICES if backing else VOICES)[_script_lang(text)]
            clip = _synth(text, voice, sr)
            gt.append({"text": chunk, "line": li, "t": round(t, 4), "end": round(t + len(clip) / sr, 4),
                       "backing": backing})
            gap = float(rng.uniform(0.05, 0.2))
            pieces += [clip, np.zeros(round(gap * sr), np.float32)]
            t += len(clip) / sr + round(gap * sr) / sr
        gap = float(rng.uniform(0.5, 2.0)) + (float(rng.uniform(6.0, 10.0)) if li % 5 == 4 else 0.0)
        pieces.append(np.zeros(round(gap * sr), np.float32))
        t += round(gap * sr) / sr
    pieces.append(np.zeros(int(1.5 * sr), np.float32))
    y = np.concatenate(pieces)
    y = 0.5 * y / max(float(np.abs(y).max()), 1e-6)
    y += (10 ** (-65 / 20) * rng.standard_normal(len(y))).astype(np.float32)
    d = out_root / name
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / "vocals.wav", y.astype(np.float32), sr, subtype="PCM_16")
    (d / "lyrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "gt.json").write_text(json.dumps({"lang": lang, "voices": VOICES, "backing_voices": BACKING_VOICES,
                                           "seed": seed, "sr": sr, "tokens": gt},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def load_tts_gen(name: str, build: bool = True) -> AlignDataset:
    d = TTS_GEN_DIR / name
    if not (d / "gt.json").is_file():
        if not build:
            raise FileNotFoundError(d / "gt.json")
        build_tts_set(name)
    meta = json.loads((d / "gt.json").read_text(encoding="utf-8"))
    gt = [GTToken(x["text"], x["t"], x["line"], x["end"], x.get("backing", False)) for x in meta["tokens"]]
    lang = meta["lang"]
    lyrics = (d / "lyrics.txt").read_text(encoding="utf-8")
    return AlignDataset(name, d / "vocals.wav", d / "vocals.wav", lyrics, gt, "word", lang,
                        lang if lang != "mixed" else "auto",
                        notes="macOS TTS speech, phrase-level GT (whitespace chunks synthesised separately)",
                        meta={"voices": meta["voices"]})


# ----------------------------------------------------------------------------------------- registry
LOADERS: dict[str, Callable[[], AlignDataset]] = {
    "edelweiss": lambda: load_edelweiss(False),
    "edelweiss_short": lambda: load_edelweiss(True),
    "kkotbat": load_kkotbat,
    **{k: (lambda k=k: load_jamendo(k)) for k in JAMENDO_SONGS},
    "tts_multilingual": lambda: load_tts_research(False),
    "tts_multilingual_long": lambda: load_tts_research(True),
    **{k: (lambda k=k: load_tts_gen(k)) for k in TTS_SETS},
}
GROUPS = {
    "jamendo": list(JAMENDO_SONGS),
    "tts": ["tts_multilingual", "tts_multilingual_long", *TTS_SETS],
    "ref": ["edelweiss", "edelweiss_short", "kkotbat"],
}
GROUPS["all"] = GROUPS["ref"] + GROUPS["jamendo"] + GROUPS["tts"]


def dataset_names(spec: str | list[str] = "all") -> list[str]:
    """Expand a comma list / group names ('all', 'ref', 'jamendo', 'tts') into dataset names."""
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for it in (x.strip() for x in items if x.strip()):
        for n in GROUPS.get(it, [it]):
            if n not in LOADERS:
                raise KeyError(f"unknown dataset {n!r}; known: {sorted(LOADERS)} or groups {sorted(GROUPS)}")
            if n not in out:
                out.append(n)
    return out


def load_dataset(name: str) -> AlignDataset:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}")
    ds = LOADERS[name]()
    for p in (ds.vocals, ds.mix):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{name}: missing {p}")
    return ds


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--build-tts"]:
        for n in TTS_SETS:
            print(build_tts_set(n))
    for n in dataset_names(sys.argv[2] if sys.argv[1:2] == ["--build-tts"] and len(sys.argv) > 2 else "all"):
        ds = load_dataset(n)
        n_lines = len({g.line for g in ds.gt})
        print(f"{n:22s} {ds.language:5s} {ds.granularity:8s} gt={len(ds.gt):4d} lines={n_lines}")
