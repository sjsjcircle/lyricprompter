# Lyrics alignment — evaluation and chosen defaults

> **요약.** 기본 정렬기는 MMS_FA 전역 CTC 정렬이다. vocal stem 활동으로 gating하고, 줄 사이에만 `*`를 넣는다.
> - **gating:** 잡음 바닥에 맞춰 임계값을 조정한다. gating이 없으면 Jamendo Embers의 AAE가 27.6 s, Rxbyn이 6.6 s이고, 있으면 각각 3.8 s, 2.1 s다.
> - **refine:** 신뢰도 < 0.10인 줄만 Qwen3-ForcedAligner로 창 단위 재검사한다. 두 정렬기가 일치하면 검토 플래그를 지우고, Qwen이 확신(≥ 0.15)하면서 불일치할 때만 Qwen 시간을 채택한다. 퇴화한 Qwen 출력(한 순간에 몰림, 창 끝에 붙음, 무음 위)은 일치로도 채택으로도 쓰지 않는다. 재검사한 줄의 35%가 이런 출력이었다.
> - **효과:** 하이브리드는 어떤 데이터셋에서도 MMS 단독보다 나쁘지 않다. 실제 곡 7개 1543 토큰의 AAE가 1.21 s→1.06 s, PCO@0.3이 0.77→0.79로 좋아졌다.
> - **검토 플래그:** 신뢰도 < 0.15 또는 MMS/Qwen 불일치이다. 줄의 18%를 표시하고 1 s 넘게 틀린 줄의 68%를 잡는다(설계 초안의 0.10은 53%).
> - **음절 끝:** 큰 소리 기준 −20 dB 활동이 이어지는 동안 늘리고, 0.15 s보다 짧은 틈은 닫는다.
> - **백보컬:** 기본은 건너뛰고 `*`를 남긴 뒤 보간한다.
> - **언어별 결과:** 한국어·일본어·영어 TTS는 AAE 0.014–0.036 s이다. 일본어 한자 묶음도 문제없었다.

Everything below is produced by `lyricprompter eval-align --datasets all --refine qwen --ablations
--json docs/eval/align_results.json` (raw per-line results are in that JSON). Times are in seconds.

## 1. Datasets

| name | lang | GT unit | audio given to the aligner | GT source / notes |
|---|---|---|---|---|
| edelweiss | en | word (74) | BS-RoFormer vocals `data/stems/edelweiss` | `Edelweiss.lrc` (ref repo); syllable tags merged into words |
| edelweiss_short | en | word (19) | `data/stems/edelweiss_short` | `Edelweiss_short.lrc`: a second performance, first 4 lines |
| kkotbat | ko | syllable (144) | `data/stems/kkotbat_ref` | **pseudo-GT**: `꽃밭에서_score.csv` onsets × 0.5 s mapped to `_ref.wav` through the ref repo's offline DTW path (25 fps); its own error is unknown (a 1.5–3 s miss at 67.8 s is likely a GT error: the stem is silent there) |
| jamendo_{hila,embers,rxbyn,cortez} | en | word (322/189/440/355) | htdemucs vocals (`data/research-testsets/jamendo/sep`) | JamendoLyrics word onsets and ends |
| tts_multilingual | ko+ja+en | line (4) | research clip `mix.wav` | GT recovered exactly: `mix = sil(1.5)+ko+gap(3)+ja+gap(3)+en+sil` (sample-identical); in-clip speech onsets measured; the 2nd Korean line starts after the TTS sentence-boundary silence (±0.03 s) |
| tts_multilingual_long | ko+ja+en | line (52) | research `long240.wav` | the same clip tiled 13× (verified sample-identical) |
| tts_ko / tts_ja / tts_en / tts_mixed | ko / ja / en / mixed | phrase (48/74/77/49) | generated (`data/research-testsets/tts-gen`) | macOS `say` (Yuna, Kyoko, Samantha); every whitespace chunk is synthesised separately and placed at a known offset. Gaps are 0.05–0.2 s inside a line and 0.5–2 s between lines, with a 6–10 s "instrumental" break every 5 lines and a −65 dBFS noise floor. tts_ja has 20 kanji-heavy lines (今日/明日/何 heteronyms included); tts_mixed mixes scripts inside lines |
| tts_backing | en+ko | phrase (63, 12 backing) | generated | lines with `(...)` call-and-response parts in a second voice (Fred / Grandma) plus whole `(ooh ooh)` lines |

TTS speech is much easier than singing (clean onsets, no accompaniment bleed). It verifies the
multilingual front-end → aligner path (Korean syllables, Japanese kanji groups / morae, mixed lines,
backing markup), not singing accuracy. There is no Japanese singing GT.

## 2. Measurement

Lyrics are parsed with the real `text.parse_lyrics`. GT tokens are matched to our display syllables by
character-level sequence alignment of normalised text (NFKC, case-folded, letters and digits only;
`difflib`), so word, syllable and line GT all work with any tokenisation. A GT onset maps to the start of
the syllable holding its first matched character, interpolated inside the syllable when the GT token
starts mid-syllable. A GT offset maps to the end of the syllable holding its last character. All 1543 real-song tokens and all
TTS tokens were matched.

Metrics: AAE, MedAE, median signed error (− = early), PCO@0.3/0.5/1.0 (fraction within τ), line-start
error (first GT token of a line vs `Line.start`), offset error where GT has ends. **Review metrics:** a line
is *bad* if its worst onset error is > 1 s; recall and precision of `needs_review` are computed against
that.

## 3. Results with the chosen defaults

`MMS+gate` = `align_lines(refine="none")`; `hybrid` = `align_lines(refine="qwen"|"auto")`.

| dataset | lang | GT | dur s | lines | config | AAE | MedAE | med signed | PCO@.3 | PCO@.5 | PCO@1 | line start Med / p90 | bad / flagged / caught |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edelweiss | en | word 74 | 125 | 16 | MMS+gate | 0.202 | 0.215 | -0.190 | 0.80 | 0.99 | 1.00 | 0.250 / 0.34 | 0 / 4 / 0 |
|  |  |  |  |  | hybrid | 0.202 | 0.215 | -0.190 | 0.80 | 0.99 | 1.00 | 0.250 / 0.34 | 0 / 4 / 0 |
| edelweiss_short | en | word 19 | 39 | 4 | MMS+gate | 0.221 | 0.190 | -0.150 | 0.90 | 0.90 | 1.00 | 0.410 / 0.82 | 0 / 1 / 0 |
|  |  |  |  |  | hybrid | 0.221 | 0.190 | -0.150 | 0.90 | 0.90 | 1.00 | 0.410 / 0.82 | 0 / 1 / 0 |
| kkotbat | ko | syllable 144 | 240 | 15 | MMS+gate | 0.436 | 0.230 | -0.105 | 0.66 | 0.78 | 0.88 | 0.380 / 0.99 | 6 / 2 / 1 |
|  |  |  |  |  | hybrid | 0.436 | 0.230 | -0.105 | 0.66 | 0.78 | 0.88 | 0.380 / 0.99 | 6 / 2 / 1 |
| jamendo_hila | en | word 322 | 234 | 58 | MMS+gate | 0.156 | 0.058 | +0.037 | 0.93 | 0.95 | 0.96 | 0.050 / 0.25 | 4 / 3 / 2 |
|  |  |  |  |  | hybrid | 0.156 | 0.058 | +0.037 | 0.93 | 0.95 | 0.96 | 0.050 / 0.25 | 4 / 3 / 2 |
| jamendo_embers | en | word 189 | 243 | 42 | MMS+gate | 3.815 | 0.229 | +0.142 | 0.55 | 0.59 | 0.62 | 0.194 / 8.86 | 18 / 28 / 16 |
|  |  |  |  |  | hybrid | 3.815 | 0.229 | +0.142 | 0.55 | 0.59 | 0.62 | 0.194 / 8.86 | 18 / 27 / 16 |
| jamendo_rxbyn | en | word 440 | 217 | 72 | MMS+gate | 2.108 | 0.076 | +0.037 | 0.66 | 0.68 | 0.72 | 0.091 / 7.33 | 23 / 31 / 18 |
|  |  |  |  |  | hybrid | 1.602 | 0.066 | +0.035 | 0.72 | 0.74 | 0.78 | 0.088 / 4.60 | 18 / 21 / 13 |
| jamendo_cortez | en | word 355 | 241 | 42 | MMS+gate | 0.236 | 0.058 | +0.040 | 0.91 | 0.92 | 0.95 | 0.054 / 0.28 | 6 / 9 / 5 |
|  |  |  |  |  | hybrid | 0.197 | 0.058 | +0.040 | 0.92 | 0.93 | 0.96 | 0.050 / 0.24 | 5 / 7 / 4 |
| tts_multilingual | mixed | line 4 | 18 | 4 | MMS+gate | 0.017 | 0.015 | -0.002 | 1.00 | 1.00 | 1.00 | 0.015 / 0.03 | 0 / 1 / 0 |
|  |  |  |  |  | hybrid | 0.017 | 0.015 | -0.002 | 1.00 | 1.00 | 1.00 | 0.015 / 0.03 | 0 / 0 / 0 |
| tts_multilingual_long | mixed | line 52 | 239 | 52 | MMS+gate | 0.163 | 0.014 | +0.004 | 0.92 | 0.92 | 0.96 | 0.014 / 0.03 | 2 / 8 / 1 |
|  |  |  |  |  | hybrid | 0.084 | 0.014 | +0.004 | 0.96 | 0.96 | 0.98 | 0.014 / 0.03 | 1 / 4 / 0 |
| tts_ko | ko | word 48 | 62 | 12 | MMS+gate | 0.015 | 0.012 | -0.007 | 1.00 | 1.00 | 1.00 | 0.006 / 0.02 | 0 / 0 / 0 |
| tts_ja | ja | word 74 | 107 | 20 | MMS+gate | 0.014 | 0.013 | -0.004 | 1.00 | 1.00 | 1.00 | 0.011 / 0.03 | 0 / 0 / 0 |
| tts_en | en | word 77 | 75 | 12 | MMS+gate | 0.036 | 0.018 | +0.009 | 0.99 | 0.99 | 1.00 | 0.006 / 0.01 | 0 / 0 / 0 |
| tts_mixed | mixed | word 49 | 63 | 12 | MMS+gate | 0.018 | 0.014 | -0.002 | 1.00 | 1.00 | 1.00 | 0.009 / 0.02 | 0 / 0 / 0 |
| tts_backing | mixed | word 63 | 72 | 12 | MMS+gate | 0.068 | 0.021 | +0.014 | 0.97 | 0.97 | 0.98 | 0.013 / 0.22 | 1 / 0 / 0 |
| **real songs pooled** (1543 tokens) | | | | | MMS+gate | 1.208 | | | 0.77 | | 0.84 | | |
| **real songs pooled** (1543 tokens) | | | | | hybrid | **1.055** | | | **0.79** | | **0.86** | | |

(The hybrid is identical to MMS+gate on the four non-refined TTS sets, so those rows are omitted.)

Reading the table:
- **Typical accuracy is good:** median onset errors are 0.06 s on Jamendo, 0.2 s on the slow / classical recordings and ≤ 0.02 s on TTS.
- **The mean is dominated by whole regions misplaced on two hard songs** (Embers, Rxbyn). Those regions have
  low confidence and are mostly flagged: 16/18 and 18/23 bad lines.
- **Edelweiss is early by ~0.17 s on average.** The LRC tags sit near the sung vowel, while CTC fires at the
  consonant ("Sm-", "Bl-"). Jamendo and TTS show ~0 bias, so no global offset is applied (`time_offset_s = 0`).
- **Variance caveat:** Embers is unstable under tiny input changes. Resampling the same stem with torchaudio's
  sinc instead of soxr moves the gated MMS AAE from 3.95 to 1.98 (this explains the research notes' 1.98).
  Differences of a few tenths on Embers are noise.

## 4. Ablations (onset AAE per dataset)

| config | edelw | edelw_s | kkotbat | hila | embers | rxbyn | cortez | tts_mul | tts_mul_L | tts_ko | tts_ja | tts_en | tts_mix | tts_bk | mean | real pooled AAE / PCO@.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **mms_gate (default)** | 0.202 | 0.221 | 0.436 | 0.156 | 3.815 | 2.108 | 0.236 | 0.017 | 0.163 | 0.015 | 0.014 | 0.036 | 0.018 | 0.068 | 0.536 | 1.208 / 0.769 |
| **hybrid (default refine)** | 0.202 | 0.221 | 0.436 | 0.156 | 3.815 | 1.602 | 0.197 | 0.017 | 0.084 | 0.015 | 0.014 | 0.036 | 0.018 | 0.068 | **0.491** | **1.055 / 0.789** |
| mms_nogate | 0.202 | 0.221 | 0.436 | 0.156 | 27.599 | 6.643 | 0.236 | 0.017 | 0.163 | 0.015 | 0.014 | 0.036 | 0.018 | 0.068 | 2.559 | 5.415 / 0.702 |
| gate_fixed_-35dB | 0.202 | 0.221 | 0.608 | 0.156 | 3.951 | 2.108 | 0.236 | 0.017 | 0.163 | 0.015 | 0.014 | 0.036 | 0.018 | 0.068 | 0.558 | 1.241 / 0.766 |
| snap_none | 0.204 | 0.221 | 0.438 | 0.156 | 3.817 | 2.109 | 0.237 | 0.027 | 0.175 | 0.016 | 0.015 | 0.039 | 0.018 | 0.077 | 0.539 | 1.209 / 0.770 |
| snap_all | 0.207 | 0.227 | 0.437 | 0.156 | 3.813 | 2.107 | 0.234 | 0.017 | 0.163 | 0.013 | 0.010 | 0.019 | 0.010 | 0.055 | 0.533 | 1.208 / 0.767 |
| snap_on_gate_mask_1s | 0.203 | 0.208 | 0.433 | 0.162 | 3.819 | 2.112 | 0.242 | 0.024 | 0.173 | 0.015 | 0.014 | 0.036 | 0.018 | 0.063 | 0.537 | 1.212 / 0.764 |
| backing_included | 0.202 | 0.221 | 0.436 | 0.156 | 3.815 | 2.108 | 0.236 | 0.017 | 0.163 | 0.015 | 0.014 | 0.036 | 0.018 | 0.059 | 0.535 | 1.208 / 0.769 |
| backing_no_star | 0.202 | 0.221 | 0.436 | 0.156 | 3.815 | 2.108 | 0.236 | 0.017 | 0.163 | 0.015 | 0.014 | 0.036 | 0.018 | 0.109 | 0.539 | 1.208 / 0.769 |
| stars_between_words | 0.194 | 0.219 | 1.222 | 0.141 | 3.661 | 1.966 | 0.155 | 0.017 | 0.144 | 0.015 | 0.014 | 0.030 | 0.018 | 0.110 | 0.565 | 1.200 / 0.786 |
| stars_between_words_pen1 | 0.219 | 0.193 | 1.480 | 0.145 | 3.837 | 1.976 | 0.162 | 0.017 | 0.144 | 0.015 | 0.015 | 0.029 | 0.018 | 0.078 | 0.595 | 1.252 / 0.773 |
| no_stars_between_lines | 0.224 | 0.221 | 1.166 | 0.159 | 4.314 | 1.484 | 0.246 | 0.017 | 0.161 | 0.015 | 0.014 | 0.036 | 0.018 | 0.068 | 0.582 | 1.164 / 0.769 |
| hybrid_refine_0.15 | 0.202 | 0.221 | 0.483 | 0.156 | 3.815 | 1.834 | 0.147 | 0.017 | 0.084 | 0.015 | 0.014 | 0.036 | 0.018 | 0.068 | 0.508 | 1.114 / 0.790 |
| hybrid_no_anchors | 0.202 | 0.221 | 0.436 | 0.161 | 3.839 | 1.638 | 0.289 | 0.017 | 0.084 | 0.015 | 0.014 | 0.036 | 0.018 | 0.068 | 0.503 | 1.091 / 0.783 |
| hybrid_win60 | 0.202 | 0.221 | 0.436 | 0.156 | 3.815 | 1.962 | 0.197 | 0.017 | 0.084 | 0.015 | 0.014 | 0.036 | 0.018 | 0.068 | 0.517 | 1.158 / 0.775 |
| hybrid_adopt_any_conf | 0.202 | 0.221 | 0.436 | 0.156 | 3.774 | 1.603 | 0.197 | 0.017 | 0.084 | 0.015 | 0.014 | 0.036 | 0.018 | 0.068 | 0.489 | 1.050 / 0.789 |
| qwen_windows_all | 0.441 | 0.221 | 0.528 | 0.152 | 3.815 | 1.897 | 0.114 | 0.017 | 0.084 | 0.015 | 0.014 | 0.036 | 0.018 | 0.068 | 0.530 | 1.139 / 0.796 |

`ends_*` and `review_0.10` leave onsets unchanged (identical rows omitted). Their effect is in §4.5 and §4.6.

### 4.1 Vocal-activity gate — essential, with an adaptive threshold

- **No gate is catastrophic on songs with long instrumental stretches.** Embers' AAE goes 27.6 → 3.8 and
  Rxbyn's 6.6 → 2.1. Lines drift into the instrumental parts.
- **The design-draft threshold, loud(p99) − 35 dB, cuts soft singing.** kkotbat (classical, pianissimo phrases
  42 dB below its peak) gets 0.608 with it vs 0.436 without a gate, and the Edelweiss pickup note is lost.
- **A sweep during development** (MMS only, before onset snapping), AAE kkotbat / Embers:

  | gate threshold | −35 dB | −45 | −50 | −55 | none | adaptive margin 10 | **adaptive margin 15** | **adaptive margin 20** |
  |---|---|---|---|---|---|---|---|---|
  | kkotbat / embers | 0.610 / 3.95 | 0.567 / 3.82 | 0.446 / 3.83 | 0.438 / 15.99 | 0.438 / 27.60 | 0.438 / 15.99 | 0.438 / 3.83 | 0.438 / 3.82 |

  htdemucs bleed in Embers sits 59–65 dB below its loud level, while BS-RoFormer stems are near
  digital silence (−75 to −100 dB).
- **Chosen threshold:** active means louder than noise floor (p10 of frame dB) + 20 dB, clipped to
  [loud − 55, loud − 35] dB, then dilated by 0.3 s. Margin 20 rather than 15 leaves headroom against noisier stems.
  `vad.vocal_activity` keeps the contract defaults; the adaptive mode is opt-in via `noise_margin_db`.

### 4.2 Star tokens — at both ends and between lines only

- **Stars between lines are kept.** Removing them hurts kkotbat (1.17), Embers (4.31) and Edelweiss. Rxbyn alone
  improves (1.48), so it is mixed, but the pooled PCO@0.3 does not change.
- **Stars between words** help pop songs (Cortez 0.236→0.155, HILA 0.156→0.141) but wreck slow legato
  singing (kkotbat 0.436→1.222). The free `*` swallows sustained vowels, and the next word starts late (bias
  +0.7 s). A star penalty of 1.0 does not fix that (kkotbat 1.48) and hurts elsewhere. The default stays
  conservative; `AlignOptions(star_between_words=True)` is available for dense pop.

### 4.3 Backing vocals — skip with a `*` placeholder and interpolate (default)

| config | all tokens AAE | lead AAE / max | backing AAE / MedAE |
|---|---|---|---|
| skip + '*' placeholder + interpolate (**default**) | 0.068 | 0.029 / 0.24 | 0.231 / 0.009 |
| include in main pass | 0.059 | 0.029 / 0.24 | 0.183 / 0.024 |
| skip, no placeholder | 0.109 | 0.062 / 1.03 | 0.312 / 0.017 |

- **The `*` placeholder matters.** Without it the unexplained backing audio drags a lead word by 1 s.
- **Interpolation** starts a skipped backing run at the first vocal onset inside its gap, at most 0.25 s per
  unit. Lead units without letters (a line-final `っ`, a word the parser could not read) that end or start a
  line before / after a long gap stay with their own line instead: they start at a vocal onset at most 1 s
  from their line's neighbouring aligned unit, else right next to it. (The first-onset rule had put a
  line-final `っ` on the next breath or on the next line's onset, stretching `Line.end` across the whole
  instrumental.) This leaves every evaluation number unchanged; the one affected token, tts_mixed's
  unread line-final `時間` after a 0.34 s pause, keeps its onset (37.16 s, GT 37.15 s).
- **Including backing** times the backing words a bit better on this call-and-response data and leaves lead
  words unchanged. It still stays opt-in (`include_backing=True`): overlapping harmonies would be forced into
  the sequential path, and we have no data for that case.
- **A rejected variant** aligned skipped backing in a second constrained pass inside its `*` gap. It was worse
  (backing AAE 0.41), because echo backing matches the lead's own sustain, so it was removed.
- Backing lines get `confidence=None` and are never flagged.

### 4.4 Line-onset snapping — small, safe gain

- **The problem:** CTC sometimes fires late inside a held pickup note ("E——very") or a soft attack.
- **The fix:** if a line's first unit starts inside a loud vocal run (≥ loud − 25 dB) that began after the
  previous unit ended, its start moves back to the run onset, by at most 0.5 s.
- **Line-start MedAE / p90:**

  | config | edelw | edelw_s | kkotbat | hila | rxbyn | cortez | tts_mul | tts_ko | tts_en | tts_bk |
  |---|---|---|---|---|---|---|---|---|---|---|
  | snap_none | 0.250 / 0.35 | 0.410 / 0.82 | 0.380 / 1.10 | 0.046 / 0.23 | 0.091 / 7.33 | 0.061 / 0.28 | 0.023 / 0.04 | 0.012 / 0.03 | 0.021 / 0.04 | 0.025 / 0.64 |
  | snap on gate mask, 1 s | 0.255 / 0.35 | 0.240 / 0.72 | 0.290 / 1.10 | 0.052 / **0.73** | 0.102 / 7.33 | 0.101 / **0.58** | 0.022 / 0.04 | 0.005 / 0.02 | 0.006 / 0.01 | 0.006 / 0.02 |
  | **default (−25 dB, 0.5 s)** | 0.250 / 0.34 | 0.410 / 0.82 | 0.380 / 0.99 | 0.050 / 0.25 | 0.091 / 7.33 | 0.054 / 0.28 | 0.015 / 0.03 | 0.006 / 0.02 | 0.006 / 0.01 | 0.013 / 0.22 |

- **Snapping on the permissive gate mask** fixes Edelweiss_short / kkotbat line starts. It also snaps
  Jamendo lines to breaths and bleed: the HILA p90 rises from 0.23 to 0.73. The strict mask keeps the TTS and
  Cortez gains without that tail. Edelweiss_short's held "E" (−36..−40 dB) is the case the strict mask
  gives up.

### 4.5 Syllable ends (karaoke wipe) — extend through sustained singing, close tiny gaps

- **Measured on GT word ends:** Jamendo and the generated TTS sets have them. Cells are AAE / MedAE / median signed.

  | dataset | raw CTC end | close gaps < 0.15 s | extend on gate mask | extend on loud−20 dB mask (**default**) |
  |---|---|---|---|---|
  | jamendo_hila | 0.228 / 0.069 / -0.045 | 0.239 / 0.087 / +0.016 | 0.264 / 0.106 / +0.091 | 0.214 / 0.091 / +0.073 |
  | jamendo_embers | 3.878 / 0.363 / +0.065 | 3.896 / 0.356 / +0.105 | 3.860 / 0.308 / +0.169 | 3.830 / 0.273 / +0.147 |
  | jamendo_rxbyn | 2.095 / 0.108 / -0.014 | 2.098 / 0.121 / +0.031 | 2.124 / 0.121 / +0.063 | 2.095 / 0.105 / +0.052 |
  | jamendo_cortez | 0.271 / 0.073 / -0.059 | 0.270 / 0.078 / -0.000 | 0.320 / 0.094 / +0.072 | 0.242 / 0.063 / +0.038 |
  | tts_ko | 0.114 / 0.118 / -0.118 | 0.115 / 0.122 / -0.118 | 0.044 / 0.012 / +0.011 | 0.044 / 0.026 / -0.009 |
  | tts_ja | 0.055 / 0.055 / -0.055 | 0.056 / 0.055 / -0.051 | 0.050 / 0.014 / +0.014 | 0.043 / 0.016 / -0.003 |
  | tts_en | 0.104 / 0.078 / -0.078 | 0.106 / 0.079 / -0.078 | 0.066 / 0.024 / +0.006 | 0.062 / 0.034 / -0.023 |
  | tts_mixed | 0.086 / 0.071 / -0.066 | 0.087 / 0.072 / -0.066 | 0.046 / 0.015 / +0.007 | 0.041 / 0.025 / -0.014 |
  | tts_backing | 0.142 / 0.087 / -0.081 | 0.144 / 0.088 / -0.081 | 0.099 / 0.052 / +0.014 | 0.097 / 0.034 / -0.007 |

- **Raw CTC ends are early.** The token fires once and blanks cover the rest of the vowel.
- **Extending on the permissive gate mask runs into reverb tails** on htdemucs stems (Cortez 0.320).
- **The policy:**
  1. Extend each unit end while the stem stays within 20 dB of its loud level, capped at 1.5 s and at the next unit's start.
  2. Then close gaps under 0.15 s.
  3. Ends before a real pause are kept.

  This is best or tied-best on every dataset.

### 4.6 Confidence and review thresholds

Line confidence is the mean posterior of the line's lead units (`AlignUnit.score`). Pooled over all lines
with a confidence (bad = worst onset error > 1 s):

| rule | lines | bad (>1 s) | flagged (share) | caught | recall | precision | recall w/o kkotbat |
|---|---|---|---|---|---|---|---|
| MMS conf < 0.10 (design draft) | 373 | 60 | 63 (17%) | 32 | 0.53 | 0.51 | 0.59 |
| MMS conf < 0.15 (default without Qwen) | 373 | 60 | 87 (23%) | 43 | 0.72 | 0.49 | 0.78 |
| hybrid: refine < 0.10, flag < 0.15 or Qwen disagrees (**default**) | 373 | 53 | 69 (18%) | 36 | 0.68 | 0.52 | 0.74 |
| hybrid: refine < 0.15 | 373 | 52 | 57 (15%) | 32 | 0.62 | 0.56 | 0.67 |

- **Development sweep (MMS only), conf threshold → flagged share / recall:** 0.05 → 10% / 0.37; 0.10 → 17% / 0.53;
  0.15 → 24% / 0.72; 0.20 → 30% / 0.80; 0.30 → 45% / 0.97.
- **Defaults:** `review_threshold = 0.15` (an operator would rather check a few extra lines) and
  `refine_threshold = 0.10`. Refining the 0.10–0.15 lines too makes times worse (`hybrid_refine_0.15` in §4:
  pooled AAE 1.114 vs 1.055, kkotbat 0.483 vs 0.436, Rxbyn 1.834 vs 1.602).
- **kkotbat's bad lines** have ordinary confidence (0.15–0.28) and are mostly pseudo-GT disagreements.

### 4.7 Qwen3-ForcedAligner hybrid (`refine.py`)

Qwen3-FA-0.6B on its own is not an option for sung material: `qwen_windows_all` (Qwen-check every line in
≤ 30 s windows) hurts the slow recordings (Edelweiss 0.44, kkotbat 0.53, vs 0.20 / 0.44 for MMS). The
research notes also showed it drifting on whole songs (max 36 s at 240 s). Its value is as a second opinion
on the lines MMS is unsure of.
Decision rules, all chosen on this data:

- **Targets:** lines with MMS confidence < 0.10.
- **Windows:** the gap between the nearest trusted lines, padded by 0.5 s.
  - Trusted neighbour lines are added as anchors when the window stays ≤ 30 s.
  - Longer gaps are split at the midpoints between the MMS line times. A window starts at its split point
    unless its first line would not fit; then it starts just early enough to hold that line (+ pad). Every
    window therefore contains the MMS span of each line it refines. A line longer than a window is not
    refined (it stays flagged).
- **Units:** display syllables (a Japanese kanji group is one unit). Qwen times map 1:1 onto `Line.syllables()`.
- **Degenerate Qwen output** is neither agreement nor a candidate for adoption: the line stays flagged. It
  is degenerate when any of these holds:
  - *collapsed:* more than half of the line's onsets fall inside one 80 ms Qwen bin, or the whole line is
    shorter than 50 ms per syllable;
  - *pinned:* more than half of the onsets lie within one bin of the window start / end (Qwen clamps the
    timestamps it cannot place to the window edge);
  - *off vocals:* fewer than half of the onsets lie within 0.1 s of vocal activity (Qwen has no gate).

  22 of the 63 refined lines in the default run were degenerate (all of them collapsed). Qwen's own
  confidence does not reveal
  this: Embers L27 had all 10 syllables at the window start (137.32 s) with confidence 0.186 and used to be
  adopted, so the earlier hybrid number for Embers (3.583) came entirely from that one collapsed line.
- **Agree** (median |Qwen − MMS| syllable onset ≤ 0.5 s): keep the MMS times (20 ms resolution vs Qwen's 80 ms;
  "adopt on agreement" made individual syllables worse in development) and **clear the review flag**
  (and the `low_confidence` flag).
- **Disagree:** adopt Qwen only if all of these hold, otherwise keep MMS:
  - its mean timestamp-class probability is ≥ 0.15;
  - its times lie inside the trusted gap (±0.5 s);
  - the anchors reproduce their MMS times (median error ≤ 1 s).

  Either way the line stays flagged (`mms_qwen_disagree`).

Thresholds of the degenerate check, from the per-line Qwen outputs of every refined line:
- Every adoption, and every agreement except HILA L41 (below), had at most 50% of its onsets in one bin,
  apart from single-syllable lines like "oh". Examples: Rxbyn L56 (40%: four onsets at the window start,
  12.4 → 0.37 s) and L23 (25%, 7.6 → 0.12 s).
- The bad ones had 67–100%: Embers L27, Rxbyn L62/63, and a synthetic long-intro case.
- A rule on zero-length or shared-start syllables at 30–50% would have rejected Rxbyn L56. Raw Qwen
  output often gives a syllable zero length, so that count does not separate good from bad.
- The only default-run line whose outcome changed apart from Embers L27 is HILA L41. Its collapsed Qwen
  output used to "agree" and clear the flag; it now stays flagged (worst error 0.53 s, so precision
  drops by one line).

Why `min_qwen_conf = 0.15`: with the degenerate output removed, the 9 default adoptions have Qwen confidence
0.17–0.72, and all of them end with a worst word error ≤ 0.95 s. Examples:
- Rxbyn L23: 7.57 → 0.12 s
- Rxbyn L56: 12.4 → 0.37 s
- Rxbyn L57: 13.2 → 0.40 s
- tts_long L28: 3.07 → 0.01 s

`hybrid_adopt_any_conf` adds four adoptions below 0.15 (confidence 0.025–0.058). Two make the line worse
(Embers L23 0.39 → 1.33 s, L34 2.41 → 3.18 s), one stays wrong (Embers L31 7.8 → 5.8 s) and one is
unchanged (Rxbyn L60). Its slightly lower pooled AAE (1.050 vs 1.055) comes from L31. Before the degenerate
check this variant was clearly worse (Embers 4.42, Edelweiss_short 0.39), because most low-confidence
adoptions were collapsed output.

The anchor check and the 30 s window are each worth keeping:
- `hybrid_no_anchors` is worse on HILA, Embers, Rxbyn and Cortez.
- `hybrid_win60` is worse on Rxbyn.

| dataset | lines refined | Qwen agrees (review cleared) | adopted Qwen | degenerate Qwen output | other disagreement, kept MMS | Qwen run s |
|---|---|---|---|---|---|---|
| edelweiss_short | 1 | 0 | 0 | 1 | 0 | 0.18 |
| jamendo_hila | 3 | 0 | 0 | 2 | 1 | 0.31 |
| jamendo_embers | 23 | 1 | 0 | 15 | 7 | 1.73 |
| jamendo_rxbyn | 25 | 10 | 5 | 4 | 6 | 1.74 |
| jamendo_cortez | 3 | 2 | 1 | 0 | 0 | 0.47 |
| tts_multilingual | 1 | 1 | 0 | 0 | 0 | 0.12 |
| tts_multilingual_long | 7 | 4 | 3 | 0 | 0 | 0.90 |

`refine="auto"` runs only when some line is below 0.10 and every model file is in the local HF cache
(`huggingface_hub.try_to_load_from_cache` for the weights, config, processor, tokenizer and chat template; then
`local_files_only=True`). It never downloads. It is best effort: if loading or running Qwen fails, the MMS
alignment is kept and the error is recorded in `AlignReport.details["refine"]["error"]`. `refine="qwen"` may
download and raises on failure.

**Language.** The forced-aligner chat template only interleaves the audio with the words and
`<timestamp><timestamp>` markers. The processor's `language` argument just picks its word splitter (nagisa for
Japanese, soynlp for Korean, whitespace otherwise), so it never reaches the model. We pass our own units, so
no splitter is needed. The dominant language per window (ko / ja / en by syllable count; mixed → dominant) is
recorded in `AlignReport.details["refine"]["windows"]`.

## 5. Speed and memory (Apple M5 Pro, MPS)

Cold `align_lines` on a 217 s pop song (72 lines, 25 refined) took 11.7 s:
- emission 4.9 s, including the MMS load;
- Viterbi 0.05 s;
- Qwen refine 6.4 s, including its load.

A 240 s ballad without refinement took 4.9 s. Peak RSS is 6.3 GB. The models are loaded one at a time:
MMS is released before Qwen loads, and Qwen is released afterwards.

## 6. Remaining limitations

- **Two hard pop songs still misplace whole regions.** On Embers and Rxbyn, line-start p90 is 4.6–8.9 s.
  These lines are mostly flagged. The UI's per-line review is the remedy.
- **No Japanese singing GT.** Japanese is only verified on TTS speech (AAE 0.014 s, kanji / heteronym lyrics).
- **kkotbat GT is pseudo-GT** from an offline DTW path. Its residual ~0.2 s median error is an upper bound.
- **Overlapping backing harmonies are untested.** Backing timings are interpolated by default.
- **Qwen3-FA often returns degenerate output on sung lines** (22 of 63 refined lines). The checks reject it,
  so on those lines the hybrid is just MMS plus a review flag. Embers gains nothing from Qwen.

## 7. Reproduce

```bash
uv run lyricprompter eval-align --datasets all --refine qwen --ablations --json docs/eval/align_results.json
uv run lyricprompter eval-align --datasets jamendo,ref          # defaults only (MMS+gate)
uv run python -m lyricprompter.eval.datasets --build-tts        # regenerate the macOS-TTS sets (deterministic)
```

MMS emissions are cached in `.cache/eval/emissions/`. The key covers the vocals file (path, size, mtime), the
window, the MMS bundle, the torch / torchaudio versions and `align_eval.EMISSION_VERSION` (bump it when
`compute_emission` changes). Pass `--no-cache` to recompute them; recomputed emissions are bit-identical on
this machine. Qwen configs run only with `--refine qwen` (may download) or `--refine auto` (only if cached);
`--ablations` alone runs the MMS ablations.
