# 실시간 추적 평가 (모듈 C)

`features/extractor.py` → `tracking/{oltw,tracker,relocate}.py` → `timeline.py` 경로의 평가 결과와, 이를 근거로 정한 기본값을 정리한다.
원자료는 `docs/eval/track_results.json`(최종 기본값으로 돌린 per-pair 지표)에 있다.

```
uv run lyricprompter eval-track                      # 모든 pair, mix+vocal
uv run lyricprompter eval-track --list               # 사용 가능한 pair
uv run lyricprompter eval-track --pairs edelweiss,synth_hila --modes vocal --json out.json
uv run lyricprompter eval-track --feat w_mfcc=0.25 --trk weights=2,1,1   # 기본값 위에 덮어쓰기(튜닝용)
```

## 1. 평가 방법

- **실제 라이브와 같은 경로로 스트리밍한다.** 라이브 오디오(16 kHz mono)를 1024샘플 블록으로 `FeatureExtractor.push()`에 넣고, 나온 프레임을 `LiveTracker.push()`에 넘긴다. 블록마다 발행된 `ref_time`을 블록 끝 시각에 기록한다. 실시간보다 빠르게 돌린다. reference 특징은 같은 코드(`compute_offline` = `push` 반복)로 만든다.
- **곡 튜닝**: reference mix에서 `librosa.estimate_tuning`으로 한 번 추정해 양쪽 chroma 필터뱅크에 같은 값을 쓴다.
- **지표** (`prompter_metrics`, `docs/research/artifacts/reference-repo/scripts/gt_and_metrics.py`를 옮김): 정답 가사 onset(라이브 시각)마다, 그 시각까지 마지막으로 발행된 위치(sample-and-hold)와 정답 ref 시각의 차이.
  - AAE, MedAE, bias(+는 앞섬), PCO@200/300/500/1000 ms
  - latency: 발행 위치의 누적 최댓값이 정답 ref onset에 처음 도달한 시각 − 정답 라이브 onset (+는 늦음)
  - lost_frac: |오차| > 2 s 비율
  - conf_auroc: 발행된 confidence가 "|오차| ≤ 0.5 s"를 얼마나 잘 가려내는지(AUROC)
  - 프레임당 CPU 시간(특징, 트래커의 평균과 p99)
- 기본값은 `latency_s = 0`, `lag_comp_s = 0`이다. 즉 보정 없는 원래 추적 지연을 그대로 잰다.

### 1.1 평가 쌍

| pair | 종류 | 내용 | 정답(GT) |
|---|---|---|---|
| `edelweiss` | 실제 | ref `Edelweiss.wav`(OST) ↔ live `Edelweiss_short.wav`(튜토리얼). 다른 가수·편곡이고 **live가 +2반음 높다**(C장조 vs B♭장조, chroma 평균 OTI로 확인). 그래서 운영자 key preset `set_transpose(2)`를 준다. | 양쪽 LRC 앞 26토큰 |
| `kkotbat_s2r` | 실제(mix만) | ref = MuseScore 렌더 `꽃밭에서_score.wav` ↔ live = 실연 `꽃밭에서_ref.wav`. live가 약 0.55배 느리다. | 144음절. `alignments/꽃밭에서_offline.csv`(오프라인 DTW 경로, 25 fps)를 거쳐 매핑하므로 GT 자체에 수백 ms 오차가 있다. |
| `kkotbat_r2s` | 실제(mix만) | 위의 반대 방향. 국소 템포비 ρ의 중앙값이 1.8이고 95%가 3.15, 최대 5.3이다(스트레스 테스트). | 같음 |
| `synth_<곡>` | 합성 | ref(mix 또는 보컬 stem)를 구간별로 time-stretch(2–8 s 구간, ρ∈[0.8,1.25] log-uniform)했다. 1 s 노이즈 lead-in, EQ tilt ±6 dB, 잔향(RT60 0.6 s, wet 0.25), pink noise SNR 20 dB, 게인 −6…−18 dB를 넣는다. vocal 모드는 반주 bleed −20 dB를 섞는다. | time map에서 정확히 계산. lead-in의 평평한 구간은 마지막 점(음악 시작)에서 역변환한다. 이전 버전은 첫 점에서 역변환해, 첫 구간 이벤트의 GT live 시각이 최대 lead_s만큼 일렀다(synth_kkotbat 144개 중 6개, −219…−620 ms). |
| `synth_*_t±1` | 합성 | 위에 ±1반음 조옮김을 더하고, 운영자 preset을 준다. | 같음 |
| `synth_hila_skip` / `synth_badside_repeat` | 합성 | 구조 편집이 있다. ref 80 s에서 +25 s로 건너뛰거나, ref 120 s에서 −30 s로 되돌아가 반복한다. 전역 재탐색을 시험한다. | 같음(반복 구간의 이벤트는 두 번 센다) |

- **곡**: Edelweiss(EN), 꽃밭에서(KO), JamendoLyrics 4곡(EN pop: embers, feel, hila, badside, 단어 단위 GT).
- **stem**: vocal 모드는 양쪽 모두 `data/stems/<name>/vocals.wav`(BS-RoFormer)를 쓴다. Jamendo는 htdemucs stem을 쓴다.
- **`꽃밭에서_score.wav`의 vocal stem은 −118 dBFS(사실상 무음)이다.** MuseScore 렌더의 선율은 악기 음색이라 분리기가 보컬로 보지 않는다. 그래서 kkotbat 실제 쌍은 mix 모드에서만 평가한다.
- **합성 쌍의 한계**
  - phase vocoder(phase locking 없음)는 transient를 번지게 하고 phasiness를 남긴다. 템포가 프레임마다 바뀌므로 구간 경계에는 인공물이 없다.
  - live가 ref 자체를 변형한 것이어서 낙관적이다. vocal 합성은 같은 분리 stem을 쓰므로 특히 그렇다.
  - 실제 공연의 프레이징, 애드리브, 편곡 차이는 실제 쌍(edelweiss)에서만 드러난다.

## 2. 최종 결과 (기본값 `FeatureConfig.for_mode` / `TrackerParams.for_mode`)

| pair | mode | n | AAE ms | MedAE | bias | lat med / p90 | PCO@200 | @300 | @500 | @1000 | lost | AUROC | reloc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edelweiss (실제, +2 preset) | mix | 26 | 181 | 100 | −125 | 46 / 193 | 73.1 | 84.6 | 88.5 | **100.0** | 0.00 | 0.74 | 0 |
| kkotbat_s2r (실제) | mix | 144 | 265 | 150 | +226 | −424 / 102 | 57.6 | 71.5 | 81.9 | 96.5 | 0.01 | 0.68 | 0 |
| kkotbat_r2s (실제, ρ≈1.8–5) | mix | 144 | 811 | 500 | −768 | 273 / 936 | 21.5 | 28.5 | 50.0 | 71.5 | 0.10 | 0.71 | 1 |
| synth_edelweiss | mix | 108 | 73 | 50 | −64 | 41 / 111 | 94.4 | 97.2 | 99.1 | 100.0 | 0.00 | 0.50 | 0 |
| synth_edelweiss_t+1 | mix | 108 | 77 | 55 | −68 | 47 / 115 | 94.4 | 96.3 | 98.1 | 100.0 | 0.00 | 0.50 | 0 |
| synth_kkotbat | mix | 144 | 74 | 40 | −46 | 27 / 102 | 92.4 | 97.2 | 99.3 | 100.0 | 0.00 | 0.50 | 0 |
| synth_kkotbat_t-1 | mix | 144 | 81 | 50 | −63 | 27 / 134 | 91.0 | 97.9 | 99.3 | 100.0 | 0.00 | 0.50 | 0 |
| synth_embers | mix | 189 | 51 | 44 | −49 | 44 / 95 | 99.5 | 100.0 | 100.0 | 100.0 | 0.00 | – | 0 |
| synth_feel | mix | 355 | 177 | 72 | −153 | 68 / 306 | 77.5 | 84.2 | 90.4 | 96.6 | 0.00 | 0.82 | 0 |
| synth_hila | mix | 322 | 89 | 50 | −76 | 42 / 136 | 93.5 | 96.6 | 97.8 | 98.8 | 0.00 | 0.84 | 0 |
| synth_hila_skip | mix | 286 | 1132 | 52 | −1117 | 47 / 156 | 90.6 | 94.4 | 95.1 | 96.2 | 0.04 | 0.93 | 1 |
| synth_badside | mix | 440 | 93 | 48 | −80 | 47 / 108 | 96.8 | 97.5 | 98.0 | 98.4 | 0.01 | 0.98 | 0 |
| synth_badside_repeat | mix | 499 | 786 | 45 | +651 | 39 / 104 | 94.4 | 95.4 | 95.6 | 96.4 | 0.03 | 0.99 | 1 |
| **mix 평균 (13쌍)** | | | 299 | 97 | | 25 | 82.8 | **87.8** | 91.8 | **96.5** | 0.015 | | |
| edelweiss (실제, +2 preset) | vocal | 26 | 299 | 100 | −252 | 129 / 821 | 73.1 | 76.9 | 76.9 | **92.3** | 0.00 | 0.64 | 0 |
| synth_edelweiss | vocal | 108 | 78 | 50 | −68 | 42 / 105 | 94.4 | 97.2 | 97.2 | 100.0 | 0.00 | 0.64 | 0 |
| synth_edelweiss_t+1 | vocal | 108 | 86 | 50 | −74 | 53 / 151 | 93.5 | 93.5 | 98.1 | 100.0 | 0.00 | 0.70 | 0 |
| synth_kkotbat | vocal | 144 | 216 | 70 | −160 | 34 / 265 | 78.5 | 88.2 | 91.7 | 95.8 | 0.03 | 0.71 | **0** |
| synth_kkotbat_t-1 | vocal | 144 | 269 | 80 | −10 | 40 / 262 | 71.5 | 79.2 | 86.8 | 94.4 | 0.03 | 0.73 | 0 |
| synth_embers | vocal | 189 | 102 | 51 | −95 | 46 / 124 | 92.6 | 94.7 | 96.8 | 98.4 | 0.00 | 0.63 | 0 |
| synth_feel | vocal | 355 | 61 | 46 | −51 | 48 / 123 | 96.9 | 99.2 | 100.0 | 100.0 | 0.00 | – | 0 |
| synth_hila | vocal | 322 | 101 | 55 | −44 | 52 / 117 | 92.9 | 95.7 | 97.5 | 98.4 | 0.01 | 0.86 | 0 |
| synth_hila_skip | vocal | 286 | 795 | 57 | −780 | 48 / 136 | 90.6 | 93.4 | 95.5 | 96.2 | 0.04 | 0.92 | 1 |
| synth_badside | vocal | 440 | 74 | 50 | −60 | 46 / 119 | 94.1 | 96.4 | 98.6 | 99.8 | 0.00 | 0.80 | 0 |
| synth_badside_repeat | vocal | 499 | 793 | 53 | +551 | 41 / 115 | 91.0 | 93.2 | 95.0 | 96.2 | 0.03 | 0.86 | 1 |
| **vocal 평균 (11쌍)** | | | 261 | 60 | | 53 | 88.1 | **91.6** | 94.0 | **97.4** | 0.013 | | |

- **skip/repeat 쌍의 큰 AAE**는 재탐색 전 약 6–7 s 동안 생긴 25–30 s 오차가 평균을 끌어올린 것이다. 재탐색은 쌍마다 1회 일어났고, MedAE는 50 ms 안팎이다. 그 밖의 재탐색은 kkotbat_r2s(mix) 1회뿐이다.
- **synth_kkotbat vocal은 이제 재탐색이 필요 없다.** 이전 버전은 92–132 s 간주(반주 bleed −20 dB) 동안 오차가 +15.7 s까지 커졌고 재탐색 1회로만 복구됐다. 지금은 간주 안 오차가 +2.6…−1.8 s에 머문다(§4.7).
- **AUROC '–'** 는 오차가 0.5 s를 넘는 이벤트가 없어 계산할 수 없다는 뜻이다. 0.50은 그런 이벤트가 1–2개뿐이고 그 시점 confidence가 포화(1.0)돼 순위가 같다는 뜻이다. 계산 가능한 22쌍 평균은 0.74이다.
- 이전 표(리뷰 전)와 비교: mix는 synth_kkotbat(94 → 97 @300)과 synth_kkotbat_t-1(97 → 98)만 바뀌었고, 둘 다 GT 역변환 수정 때문이다. vocal 평균은 91.8 / 97.2에서 91.6 / 97.4가 됐다. 수정된 GT로 이전 코드를 다시 재면 92.2 / 97.3이므로, vocal 수정의 순효과는 −0.6 / +0.1 pt이다(§4.7).
- **참고: 논문 기준선**. ICASSP'24(Park et al.) online Chroma+MFCC(5)는 winterreise에서 AAE 453 ms, MedAE 144 ms, PCO@1000 90.6%였다. 데이터가 다르므로 직접 비교는 아니다. 같은 성격의 실제 쌍(edelweiss, 다른 연주·다른 편곡)에서 mix 모드는 AAE 181 ms, MedAE 100 ms, PCO@1000 100%가 나왔다.

### 2.1 운영자 key preset이 없을 때

| pair | mode | preset 사용 PCO@300 / @1000 | preset 없음·자동 키 끔 PCO@300 / @1000 | preset 없음·자동 키(기본) PCO@300 / @1000 |
|---|---|---|---|---|
| edelweiss (+2) | mix | 84.6 / 100 | 11.5 / 34.6 (lost 0.50) | 73.1 / 88.5 (+2로 전환) |
| edelweiss (+2) | vocal | 76.9 / 92.3 | 38.5 / 61.5 (lost 0.19) | 38.5 / 61.5 (전환 안 함, 제안만) |
| synth_edelweiss_t+1 | mix / vocal | 96.3 / 100 · 93.5 / 100 | 79.6 / 97.2 · 87.0 / 98.1 | 96.3 / 100 · 94.4 / 100 |
| synth_kkotbat_t-1 | mix / vocal | 97.9 / 100 · 79.2 / 94.4 | 76.4 / 95.1 · 36.8 / 65.3 | 97.9 / 100 · 78.5 / 93.8 |

조옮김된 공연을 원래 키로 추적하면 위처럼 무너진다. 그래서 **자동 키 검출이 기본으로 켜져 있다**(§7).

- **반주가 있는 mix**: 키를 확실하게 찾는다. 위 세 쌍 모두 정답 키로 한 번 전환했고, 결과가 preset을 준 경우와 거의 같다.
- **보컬 단독(vocal)**: 합성 쌍은 정답 키로 전환했다. 실제 Edelweiss 보컬은 +2와 이웃 키가 비슷해 전환하지 않았고, 라이브 36 s에야 `key_suggest=+2` 제안이 떴다(mix는 9.5 s에 +2로 전환). 이때는 운영자가 키를 미리 정하거나(`set_transpose`) 제안을 확인해 적용해야 한다.
- 키를 알고 있으면 운영자 설정(`set_transpose`)이 여전히 가장 확실하다. 운영자가 키를 정하면 자동 검출은 꺼진다.

### 2.2 CPU (M5 Pro, 다른 에이전트 4개와 공유)

| 항목 | 측정값 |
|---|---|
| 특징 추출 (1024샘플 블록) | 평균 0.04–0.16 ms/frame, p99 ≤ 2.1 ms (다른 에이전트와 CPU 공유 중 측정) |
| 트래커 step | 평균 0.05–0.20 ms/frame, p99 ≤ 3.2 ms (p99 최댓값은 재탐색 검사가 든 블록) |
| 전역 재탐색 검사 (`_try_relocate`, 4분 곡) | 약 3.5 ms. 신뢰도가 낮을 때만, 최대 0.5 s마다 |
| 프레임 예산 | 40 ms. 과제 기준 특징 < 2 ms/frame을 만족한다 |

## 3. 기본값 (코드에 반영됨)

```python
FeatureConfig.for_mode("mix",   tuning)  # n_fft 2048, chroma_gamma 1 (거의 선형), w_chroma 1, w_mfcc 0.5,
                                          # n_mfcc 20 (c1..c19), mfcc_cmn_s 5, mfcc_anchor 2, w_flux 0
FeatureConfig.for_mode("vocal", tuning)  # 같고, w_mfcc 1.0
TrackerParams.for_mode(mode)             # weights (diag, hold, ref-only) = (1.8, 1.2, 1.0), win ±10 s,
                                          # max_adv 3, smooth 1.0 (raw), lag_comp 0, 전역 재탐색 on
# vocal 재획득: relock_back_s 3, relock_stop_back_s 0.5, relock_fwd_s 3, relock_free_s 0.5,
#               relock_confirm_s 0.3, onset_rest_max_s 0 (줄 안 쉼표 onset은 opt-in), inline_wait_s 1
```

- **특징 그룹**: 그룹마다 L2 정규화한 뒤 √(w/Σw)를 곱해 이어 붙인다. 그래서 프레임 벡터의 크기는 1이고 거리는 `1 − a·b`(그룹 가중 cosine)이다.
- **무음 처리**: chroma에는 eps 0.02의 균일 성분을, MFCC에는 상수 anchor 성분을 둔다. 그래서 무음끼리의 거리는 0이다.
- **vocal 모드 전용**
  - 무음 0.36 s(grace) 동안은 DP가 계속 따라간다. ref 쉼표 구간에서는 argmin이 쉼표 끝(가수가 다시 들어올 자리)으로 달려가는데, 이것이 정보가 된다. 발행 위치는 coast limit을 넘지 않는다. 그 뒤에는 COASTING(DP 정지, 템포 0.8–1.25로 외삽)이다.
  - 발행 위치는 coast limit(다음 줄 시작. `line_starts`가 없으면 ref stem 활동 p99−25 dB의 rising edge)을 넘지 않는다. 현재 위치에 걸친 onset(줄 시작에서 start/jump/hold 해제)은 그 자체가 limit이라 목소리가 올 때까지 그 자리에서 기다린다. 이때 `tempo`는 0으로 보고하고, latency 보정도 limit을 넘기지 않는다.
  - 보컬이 2프레임 이어지면 다시 잡는다(relock). 예측 위치 −3…+3 s(다음 onset+1 s까지 확장)에 soft prior β=0.3/s를 걸고 ref가 비활성인 프레임에는 +0.3을 더한다. 다만 목소리가 멈춘 자리(또는 운영자 anchor) −0.5 s 앞으로는 가지 않는다.
  - relock 뒤 0.5 s 동안은 발행 위치가 argmin을 따라 **뒤로도** 움직일 수 있다(앞으로는 여전히 프레임당 max_adv). coasting이 넘어간 만큼을 되돌린다.
  - relock은 목소리가 0.3 s 이어질 때까지 잠정적이다. 그동안 표시는 coast limit을 coasting 속도 이상으로 넘지 않는다. 확정 전에 grace보다 긴 무음이 오면 relock을 취소하고 coasting 상태(위치 궤적, limit, 템포, 신뢰도)를 복원한다. 간주 중 bleed 버스트가 다음 줄로 표시를 밀어내지 못하게 하기 위해서다.
- **전역 재탐색**
  - 시작 조건: confidence < 0.35가 2 s 넘게 이어지고, 마지막 anchor 이후 6 s가 지났을 때.
  - 탐색: 최근 6 s를 0.32 s hop / 0.64 s Hann으로 pooling하고, subsequence DTW(기울기 0.5–2)로 곡 전체에서 후보를 찾는다.
  - 채택 조건: 후보 비용 × (1 + 0.5·거리/60 s)가 현재 위치 ±2.6 s 안의 최선 비용 × 0.75보다 작아야 한다.
  - 검증: 그 후보를 shadow OLTW로 3 s 재생해, 뒤쪽 1.5 s의 평균 거리가 기존 경로의 0.75배 미만일 때만 전환한다.

## 4. 튜닝 과정 (근거)

평균은 각 단계에서 쓴 pair 집합 기준이다. 집합이 바뀌면 숫자를 서로 비교할 수 없다. 스크립트와 로그는 `scratch/c-tracking/`에 있다.

1. **특징의 변별력이 핵심이었다.** 초기 설정(chroma log1p(100·c), MFCC c1..c12, anchor 10, CMN 없음)에서는 실제 정답 프레임과의 거리가 ±10 s 창 평균과 거의 같았다(예: synth_badside 0.02 vs 0.04). 그래서 비슷한 구간에 쉽게 끌려갔다. mix 7쌍 기준 결과:

   | 변경 | 평균 PCO@300 / @1000 |
   |---|---|
   | 초기 | 60.8 / 76.6 |
   | gamma 10 | 70.6 / 84.0 |
   | gamma 1 | 75.9 / 89.9 |
   | gamma 10 + CMN 5 s + anchor 2 | 75.6 / 91.0 |
   | gamma 1 + CMN 5 s + anchor 2 + w_mfcc 0.25 | 78.2 / 91.2 |
   | spectral flux 그룹, n_fft 4096, eps 0.005/0.1 | 개선 없음(±1 pt) |

2. **DP 가중치.**
   - 대칭 가중치 (2,1,1)에 길이 정규화를 쓰면 hold와 진행의 비용이 같다. 그래서 반복적이거나 모호한 구간에서 단조 argmin이 멈추고 0.5–2.5 s 뒤처진다(synth_feel).
   - ref-only 가중치를 올리면(wv 1.5/2) 빠른 템포를 쫓지 못해 크게 나빠졌다(60.8 → 53.6 / 47.2).
   - 창을 ±3 s로 줄이면 한 번 놓친 뒤 복구하지 못했다(29.3 / 36.3). ±5 s와 ±6 s는 차이가 없었고, max_adv 2도 나빠졌다.
   - 대각 bias와 약한 hold 패널티는 도움이 됐다. mix 9쌍 기준:

     | weights (diag, hold, ref-only) | 평균 PCO@300 / @1000 |
     |---|---|
     | (2, 1, 1) | 78.9 / 91.4 |
     | (1.8, 1, 1) | 79.5 / 91.7 |
     | (1.8, 1.1, 1) | 80.6 / 92.2 |
     | **(1.8, 1.2, 1)** | **82.9 / 94.0** |
     | (1.8, 1.3, 1) | 11쌍에서 86.0 / 94.8, 대신 kkotbat_s2r 62 → 57 |

   - vocal은 (2,1,1) 87.5 / 95.1에서 (1.8,1.2,1) 91.9 / 96.3으로 올랐다.
   - 대가가 있다. hold 패널티는 "ρ ≥ 1 쪽"의 tempo prior처럼 작동하므로, 아주 느린 live(kkotbat_s2r, ρ≈0.55)에서는 76 → 62–72로 떨어진다.
3. **MFCC 개수와 CMN.**
   - vocal 모드에서 n_mfcc를 13에서 20으로 늘리자 91.8 / 98.0이 됐다(실제 edelweiss 62/73 → 73/92). 30은 비슷했다.
   - CMN을 끄면 78.5 / 90.9로 크게 나빠진다. 마이크와 stem의 음색 차이를 제거하는 것이 중요하다. CMN 10 s는 5 s와 비슷했다.
   - w_mfcc는 0.5–1이 적당했고, 2에서는 나빠졌다(86.4).
   - mix 모드는 n_mfcc 20, w_mfcc 0.5에서 85.8 / 95.8로 가장 좋았다(11쌍).
4. **vocal 모드의 구조 결함을 고쳤다.**
   - 문제: ref stem 활동 임계(p99−35 dB)가 bleed를 onset으로 잡았다(embers 11.5 s). 그래서 coasting이 엉뚱한 곳에서 멈췄고, 다시 잡을 때도 20 s 떨어진 곳에 고정돼 끝까지 lost였다(PCO 0%).
   - 처방: coasting의 상한을 줄 시작(`line_starts`)으로 바꾸고, ref 활동 임계를 p99−25 dB로 올렸다. 전역 재탐색도 추가했다.
   - 결과: vocal 평균이 33.9 / 48.2에서 87.5 / 95.1로 올랐다.
5. **전역 재탐색에서 false jump를 막았다.**
   - 첫 버전은 반복 후렴으로 잘못 점프했다(synth_feel 100 s → 183 s, kkotbat 시작 직후 → 107 s).
   - coarse 비용만으로는 반복 구간을 구분할 수 없었다(실측: 실제 위치 0.182 vs 반복 후렴 0.141). shadow 검증도 완벽한 반복은 걸러내지 못한다.
   - 그래서 이웃 기준(현 위치 ±2.6 s의 최선 비용), 거리 패널티 0.5/60 s, anchor 직후 6 s 금지, 판정 ratio 0.75를 넣었다.
   - 결과: false jump는 0회가 됐고 skip과 repeat은 각 1회로 정확히 복구했다. 거리 패널티 0.25에서는 synth_feel의 false jump가 남았다(71/87 → 0.5에서 84/97). tie 1.15는 정상 복구까지 막았다.
6. **출력 평활(smooth 0.3)과 lag 보정은 효과가 작았다.**
   - 평활은 ±1 pt 안이라 기본값을 raw로 둔다.
   - `lag_comp_s` 0.04–0.12는 평균을 +0.3–1.9 pt 올렸지만 느린 쌍을 해쳐서 기본값을 0으로 뒀다.
   - 발행 위치의 bias는 합성에서 −50…−100 ms이다. 1024 블록 발행 간격(평균 32 ms)과 onset에서의 DTW 지연 때문이다. 운영자는 `latency_s`로 보정하거나, UI 리드 타임을 쓰면 된다.

7. **리뷰 수정: vocal 상태기계(재현 스크립트 `scratch/review-C-tracking/r2_*.py`, ablation `scratch/c-tracking/fix/`).**
   - **줄 시작에서 start/jump/hold 해제**: `_coast_begin`이 현재 위치의 onset을 건너뛰어(`side="right"`) 목표 줄을 그대로 지나갔다. 3 s 무음 뒤 오차가 +3.6…+8.9 s, PCO@300 0–1%였다(합성). 이제 현재 위치 −0.5프레임 이상인 첫 onset을 limit로 쓴다. 같은 시나리오가 100%이고, 실제 Edelweiss에서 줄 시작에 start한 경우 진입 위치는 13.12 / 19.28 / 26.00 s(정답 12.99 / 19.27 / 25.85, 이전 16.72 / 23.24 / 29.56)이다. 줄 19.27에서는 PCO@1000이 53.8에서 100%가 됐다.
   - **relock이 뒤로 못 가는 문제**: `anchor`가 pos=c로 시작하고 단조 clamp가 있어 prior의 −2 s가 쓰이지 않았다. 줄 안 쉼표를 ref보다 1.0 / 1.5 / 2.0 s 길게 쉬면 PCO@300이 55 / 37 / 0%였다. 이제 relock 뒤 free 단계에서 뒤로 움직일 수 있고(앞으로는 clamp 유지. 앞으로까지 풀면 ref-only 연쇄가 prior 밖 31 s까지 달려갔다) back을 3 s로 늘렸다. 결과는 99 / 99 / 99%이다.
   - **고려했지만 기본값에서 뺀 것**(근거: 11쌍 vocal 평균 PCO@300, 수정된 GT, 이전 코드 92.2):

     | 변형 | PCO@300 / @1000 |
     |---|---|
     | 이전 코드(수정된 GT) | 92.2 / 97.3 |
     | grace 동안 DP 동결(리뷰 제안), 다른 수정 없음 | 90.4 / 97.6 |
     | grace 동안 발행 위치만 coast 속도로 제한, 다른 수정 없음 | 90.7 / 97.3 |
     | grace 그대로 + 줄 시작 수정 + free relock(back 2 s) | 92.1 / 97.4 |
     | 위 + ref 활동 기반 줄 안 쉼표 onset(대기 무제한 / 1 s) | 90.8 / 97.0 · 91.0 / 97.1 |
     | 위(쉼표 onset 없이) + back 3 s | 92.0 / 97.3 |
     | **최종: + 바닥(멈춘 자리 −0.5 s) + 잠정 relock 0.3 s** | **91.6 / 97.4** |

     - grace 동결이나 제한이 손해인 이유: ref 쉼표를 지나는 DP의 "질주"가 쉼표 끝, 즉 가수가 다시 들어오는 자리를 정확히 가리킨다. 이를 막으면 재진입이 0.3–0.9 s 늦어진다.
     - 줄 안 쉼표 onset이 손해인 이유: 라이브 게이트가 놓치는 약한 재진입과, bleed에 가려진 pickup 때문이다. ref에서는 활동으로 잡혀 coasting이 거기 멈춰 기다린다(synth_edelweiss 34.92 s: 대비 2 dB, −0.75 / −1.03 s. synth_badside 33.68 s: 대비 25 dB인데도 −0.90 s). level 대비로는 걸러지지 않았다. 그래서 `onset_rest_max_s > 0`일 때만 켠다. 깨끗한 근접 마이크용이다.
   - **간주 중 bleed 버스트**(30 s 간주, 0.12–0.28 s 버스트가 1–2 s마다 오는 테스트 `test_vocal_interlude_with_bleed_bursts_does_not_ratchet`): 이전 코드는 버스트마다 relock해 곡 끝까지 달려갔다(6 seed 모두, 다음 줄 PCO@300 0%). free relock만 넣으면 ref 활성 프레임 +0.3 비용 때문에 버스트가 이전 줄 끝으로 표시를 끌어당겼다. 바닥(멈춘 자리 −0.5 s)과 잠정 relock을 넣자 6 seed 모두 다음 줄 앞에서 기다렸다가 따라간다(PCO@300 100%). 잠정 relock의 비용은 0.2 s에서 −0.4 pt, 0.3 s에서 −0.6 pt, 0.5 s에서 −1.8 pt이다. limit에서 표시를 묶어두면 0.3 s에서도 −1.2 pt라서 limit을 coasting 속도로 넘을 수 있게 했다.
   - **기타**: latency 보정과 보고 템포가 coast limit을 넘던 문제(+0.30 s, tempo 1.0 → 이제 limit 그대로, tempo 0), 손상되거나 잘린 캐시에서의 예외와 동시 쓰기 경합(고유 임시 파일 + `os.replace`), NaN/inf 샘플 한 개가 CMN 상태를 영구 오염시키던 문제(입력 정리), 합성 GT lead-in 역변환, `--trk/--feat`의 Optional 필드 파싱과 명시한 `tuning=0`이 무시되던 문제를 고쳤다. `set_transpose`는 chroma group이 없는 reference에서 ValueError를 낸다.

## 5. 알려진 실패 모드

- **조옮김**: 키 preset도 자동 키 검출도 없으면 실제 Edelweiss(+2반음) mix 결과가 PCO@1000 35%로 무너진다. 곡 전체 평균 chroma로 키를 정하는 방식(OTI)은 후보 사이 차이가 작아(0.969 vs 0.960) 쓰지 않았다. 대신 키마다 shadow 트래커를 돌려 경로의 local distance를 비교하는 자동 키 검출을 기본으로 켠다(§7).
  - mix에서는 안정적이다(실제 Edelweiss mix: +2로 전환, PCO@300 73.1%).
  - 보컬 단독은 이웃한 키 사이에서 애매할 수 있다. 실제 Edelweiss 보컬은 전환하지 않았고, 라이브 36 s에야 +2를 제안했다(PCO@300 38.5%, preset을 주면 76.9%). 운영자가 키를 미리 정하거나 콘솔 제안을 확인해 적용해야 한다.
  - 전환 전(mix는 처음 6–10 s)에는 원래 키로 추적하므로 초반 위치가 틀릴 수 있다. 키 검색의 첫 판정(근거 6 s) 전에는 전역 재탐색도 미룬다(키가 틀리면 "놓침"처럼 보이기 때문).
- **극단적 템포비**: ρ가 국소적으로 3을 넘으면(kkotbat_r2s: 렌더 대비 실연이 1.8–5배 느림) max_adv 3을 넘어서 계속 뒤처진다(PCO@1000 71.5%). 실제 공연과 원곡 사이 템포 차이(±25%)에서는 생기지 않는다. 너무 느린 live(ρ≈0.55)는 hold 패널티 때문에 다소 손해를 본다.
- **모호하거나 반복적인 반주 구간**: 반주가 반복적이면 mix 모드에서 0.5–1.5 s 지연이 국소적으로 생긴다(synth_feel 84%@300). confidence가 떨어지는 구간과 대체로 겹친다(AUROC 0.82).
- **반복 구조**: 동일한 후렴이나 절이 반복되면 전역 재탐색은 거리 prior에 의존한다. 1분 넘게 떨어진 진짜 점프(곡 순서를 바꾸는 경우)는 비용 우위가 커야만 채택된다. 운영자 `jump`가 최종 수단이다.
- **vocal 모드**
  - 줄 시작(`line_starts`)이 없으면 stem 활동의 rising edge를 쓴다. bleed가 많은 stem에서는 잘못된 onset이 생길 수 있다. LiveSession은 `Timeline.line_start(i)` 목록을 넘겨야 한다(가사 수정 시 `set_line_starts`).
  - 실제 Edelweiss vocal은 76.9%@300, 92.3%@1000이다. 다른 가수의 음색과 장음 구간 때문이고, confidence 평균이 0.39로 낮다. 줄 25.85 s부터 start하는 경우는 진입은 맞지만(26.00 s) 이어지는 토큰을 잘 못 따라간다(PCO@1000 28.6%). 이전 코드도 같았다.
  - 가수가 ref보다 쉼표를 길게 쉬면 coasting이 줄의 다음 부분으로 넘어간다. relock 때 되돌리지만, 그 사이 최대 3 s 동안 표시가 앞서 있다(3 s를 넘으면 global relocation에 맡긴다). 줄 안 쉼표 onset(`onset_rest_max_s`)을 켜면 그 자리에서 최대 1 s 기다린다.
  - 0.3 s(`relock_confirm_s`)보다 긴 bleed 활동은 목소리로 취급한다. 간주 중 긴 bleed가 이어지면 표시가 다음 줄로 넘어갈 수 있다. level 게이트만으로는 구분할 수 없다.
  - relock은 목소리가 멈춘 자리 0.5 s 앞까지만 되돌린다. 가수가 앞 구절을 반복하면 운영자 `jump`가 필요하다.
- **신뢰도**: 비율 d(pos)/mean(d)의 EMA(1.5 s)로 계산한다. 순위 정보는 쓸 만하지만(평균 AUROC 0.78) 절대값 보정은 하지 않았다. 실제 쌍에서 평균이 0.4–0.7이므로, UI 임계(0.3/0.6)는 "dim, 경고" 정도로만 쓰는 것이 안전하다.
- **mix 모드 시작**: live에 노이즈 lead-in(1 s)이 있어도 합성 결과는 정상이다. 다만 start 시점에 ref와 전혀 다른 긴 인트로나 MC가 있으면 초반 위치가 앞서 나간다. 이 경우 운영자 `start(line)` / `jump`를 쓴다.

## 6. 다른 모듈과의 연결 메모

- `FeatureFrames`에 선택 필드 `groups`(name, start, stop)를 추가했다. 트래커의 `set_transpose`가 이것으로 chroma 열을 찾는다.
- `TrackerState`에 `active`(라이브 활동)와 `transpose`를 추가했다.
- `TrackerState.tempo`는 **발행 위치가 실제로 움직이는 속도**다. tracking 중에는 템포 추정값, coasting 중에는 coasting 속도이고, coast limit에서 기다릴 때와 hold/waiting/stopped에서는 0이다. 프롬프터의 `ref_time + tempo·(now − t)` 외삽이 다음 줄로 넘어가지 않게 하기 위해서다. latency 보정(`rate × latency`)도 coast limit에서 멈춘다.
- `set_transpose(k)`는 reference에 chroma group(`FeatureFrames.groups`)이 없으면 k % 12 ≠ 0에서 `ValueError`를 낸다. 예전에는 조용히 무시했다.
- `LiveTracker.set_line_starts(line_starts)`를 추가했다. `LiveSession.reload_lines`가 이미 이 이름이 있으면 호출한다.
- `TrackerParams`에 `lag_comp_s`를 추가했다.
- `LiveTracker`
  - 생성자는 계약대로 `(ref, fps, params, ref_active=None, line_starts=None)`을 받는다.
  - `set_latency(s)`와 `relocations` 카운터를 추가로 제공한다.
  - `jump`, `nudge`, `hold`는 frame 격자로 반올림하지 않은 정확한 시각을 발행한다.
- vocal 모드 `LiveSession`의 권장 구성
  - reference는 `load_ref_features(song_dir, "vocals", FeatureConfig.for_mode("vocal", meta.tuning))`로 만든다.
  - `line_starts`에는 `[Timeline(lines).line_start(i) for i ...]`를 넘긴다.

## 7. 자동 키(조옮김) 검출 (통합 단계에서 추가)

라이브가 원곡과 다른 키로 불리면 chroma가 어긋나 추적이 무너진다. 키 preset 없이 실제 Edelweiss를 mix로 추적하면 PCO@300이 11.5%였다.
`TrackerParams.auto_key=True`(기본값)로 두면 `start()` 직후 **나머지 11개 키 각각에 대해 shadow FrameSyncOLTW**를 돌린다(reference chroma를 roll한 것).

- **같은 입력, 같은 anchor**: 모든 shadow가 main과 같은 프레임을 받고, relock·점프·재탐색 때도 같은 방식으로 anchor된다.
- **판정 근거**: 검색을 시작한 뒤 쌓인 **누적 평균 local distance**. chroma가 키에 따라 달라지는 유일한 그룹이므로, 거리가 가장 낮은 키가 라이브의 키다.
  - mix 모드는 밴드가 들어오기 전의 방 소음·박수도 추적하므로, 라이브 활동 게이트가 `key_onset_s`(1 s) 동안 끊김 없이 켜진 뒤부터 근거를 다시 센다. vocal 모드는 목소리가 있을 때만 추적하므로 모든 추적 프레임이 근거다.
  - 검증된 전역 재탐색이 일어나면 근거를 처음부터 다시 쌓는다(틀린 자리에서 모은 근거이므로).
- **전환 조건**: 1위 키가 현재 키를 포함한 **다른 모든 후보보다 10% 넘게 낮고**(`key_ratio=0.9`), 이것이 **3회 연속**(1 s 간격, 최소 6 s 추적 후) 유지되어야 한다. 조건을 만족하면 그 shadow의 DP 상태를 그대로 넘겨받는다.
- **검색 종료**: `key_max_s`(40 s) 동안 추적한 뒤 shadow를 해제한다.
- **결정이 애매할 때**: 이웃한 두 키가 비슷한 경우(반주 없는 목소리)에는 전환하지 않는다. 대신 `TrackerState.key_suggest`로 콘솔에 후보 키를 보여준다(클릭하면 적용).
- **운영자 키 지정**: 운영자가 키를 직접 정하면(`set_transpose`) 자동 검출은 꺼진다. 콘솔 **Auto** 버튼(`set_auto_key`)으로 다시 켤 수 있다.
- **CPU 비용**: shadow 11개를 돌리는 처음 40 s 동안 트래커 p99가 0.5–1.3 ms이다.
- **재탐색과의 관계**: 키가 틀리면 신뢰도가 낮아 "놓침"처럼 보이므로, 키 검색의 첫 판정(근거 6 s) 전에는 전역 재탐색을 하지 않는다.
- **인터페이스**: `LiveTracker.set_auto_key(on)`, `TrackerState.key_auto`(키를 자동 검출이 정하는 중인지), `TrackerState.key_suggest`(제안 키 또는 None), `key_switches` 카운터. 서버는 `POST /api/session/autokey`, `SessionStart.auto_key`로 노출한다([API_CONTRACTS.md](../API_CONTRACTS.md)).

키 preset 없이 측정한 결과(`--no-key-preset`, `docs/eval/track_results_autokey_nopreset.json`; 최종 리뷰에서 음악 시작 전 잡음·박수를 근거에서 빼는 수정이 들어간 뒤의 값. 실제 Edelweiss mix는 그 전 80.8 / 96.2였다. 26개 이벤트뿐이라 이벤트 2개 차이다. 대신 잡음·박수 lead-in이 있는 39개 실험에서 잘못된 전환이 0회가 됐다):

| pair | mode | 자동 검출 키 | PCO@300 / @1000 (자동) | PCO@300 preset 없음·자동 끔 | PCO@300 preset(정답 키) |
|---|---|---|---|---|---|
| edelweiss (실제, +2) | mix | **+2** | **73.1** / 88.5 | 11.5 | 84.6 |
| synth_edelweiss_t+1 | mix | +1 | 97.2 / 100.0 | 79.6 | 96.3 |
| synth_kkotbat_t-1 | mix | −1 | 97.2 / 100.0 | 76.4 | 97.9 |
| synth_edelweiss_t+1 | vocal | +1 | 94.4 / 99.1 | 87.0 | 93.5 |
| synth_kkotbat_t-1 | vocal | −1 | 78.5 / 92.4 | 36.8 | 79.2 |
| edelweiss (실제, +2) | vocal | 전환 안 함(+2·+3 애매 → 36 s에 +2 제안) | 38.5 / 61.5 | 38.5 | 76.9 |

- **오검출 없음**: 조옮김이 없는 나머지 21쌍(기본 preset으로 평가)은 키 전환 0회였고 결과가 예전과 같다(`scratch/autokey/all_final.json`).
- **실제 보컬 단독 + 조옮김 공연**은 목소리만으로 이웃 키를 확실히 가리기 어렵다. 이 경우 운영자가 키를 미리 정하거나, 콘솔에 뜬 제안을 확인해 적용하는 것을 권장한다.

## 8. 추적 스타일: classic / pop (2026-10-13 피드백)

사용자 피드백: "지금 시스템은 박자가 계속 바뀌는 클래식에 특화돼 DTW로 매 음절을 정확히 따라가려 한다. 밴드·록을 포함한 pop은 박자가 일정해 템포 비율이 더 중요하다. 특정 음절을 놓쳤다고 다음 가사로 넘어가길 기다리면 안 된다."
그래서 곡마다 고르는 **추적 스타일** `style = "classic" | "pop"`을 추가했다(UI: "Classical — flexible tempo" / "Pop & band — steady tempo"). classic은 지금까지의 트래커 그대로이고, pop은 아래 설계다.
원자료: `docs/eval/track_results_styles.json`(두 스타일 × 전체 쌍, 최종 기본값). 스크립트·로그·sweep: `scratch/pop/`(`cache.py` 특징 캐시, `run.py` 이벤트 구분 지표, `sweep.py`, `diag.py` 시간축 진단, `leadin.py`, `unit/exp_*.py`).

```
uv run lyricprompter eval-track --styles classic,pop                      # 두 스타일로 모든 쌍
uv run lyricprompter eval-track --pairs pop_hila,steady_hila --styles pop --modes vocal
uv run lyricprompter simulate --song <id> --file live.wav --style pop     # 기본: 곡에 저장된 스타일
uv run lyricprompter prep ... --style pop                                 # 재준비는 --style 없으면 기존 스타일 유지
```

### 8.1 classic이 pop에서 실패하는 방식 (pop 평가 쌍에서 먼저 확인)

classic은 발행 위치를 OLTW argmin이 앞으로 갈 때만 움직인다(`smooth` 1.0). 그래서 DTW가 맞출 것이 없으면(관객이 부르는 줄, 애드리브, 가수가 빠뜨린 음절) 표시가 멈춘다. vocal 모드에서는 coasting이 다음 줄 시작(coast limit)에서 목소리를 기다린다.
아래 pop 쌍(§8.3)에서 classic을 먼저 재 이 실패를 확인했다. vocal 모드의 빠진 줄 안 이벤트는 PCO@300 13.6%였고, 표시가 정답보다 늦은 정도(latency 중앙값)는 +3.9 s였다. 즉 다음 줄 시작에서 멈춰 기다렸다. 그 뒤 두 줄도 35.7%였다. 애드리브 줄(다른 줄의 목소리)에서는 전역 재탐색이 그 다른 줄로 점프했다가 돌아왔고, pop_ 6곡 vocal 평균은 66.3 / 83.1(@300 / @1000), lost 12%였다.

### 8.2 pop 설계 (`tracking/tracker.py`, `TrackerParams.pop_*`)

DP(OLTW)는 그대로 돌리고, 그 위치를 **위상(phase)의 잡음 섞인 측정값**으로만 쓴다. 템포가 강한 단서다.

- **고정 템포 추정**: OLTW 국소 템포(6 s 경로 기울기)를 confidence 가중 running mean으로 모으고, 가중치 합이 1/`pop_tempo_alpha`(100)에 닿으면 EMA가 된다. 초기값 1.0의 가중치는 `pop_tempo_prior` 15다. 범위는 0.7–1.4다.
  - lock(아래) 뒤에만, lock 이후 경로로만 잰다. 이전 버전은 방 소음 구간에서 멈춘 DP의 기울기 때문에 시작 직후 템포가 0.74로 떨어져 20 s 동안 회복하지 못했다.
  - DP가 표시와 0.5 s 넘게 다르면 표본을 버린다. 빠진 줄에서 멈춘 DP가 밴드 템포를 끌어내리지 않게 하기 위해서다.
  - 추정값과의 차이가 0.1을 넘는 표본은 Huber 가중으로 영향이 제한된다.
  - 추정 정확도(30 s 이후, 곡 전체 평균 템포비 대비. drift ±0.5% 포함): pop_·steady 쌍에서 mix 평균 오차 0.1–0.6%, vocal 0.7–1.3%(p90 ≤ 2.4%)다(`scratch/pop/tempo_check.py`).
- **발행 위치 = 등속 예측 + 제한된 보정**: 매 프레임 `out += tempo` 하고, DP 위치 쪽으로 `pop_gain`(0.08)×오차만큼 당긴다. 다만 속도 변화는 `pop_rate`(0.5)×tempo×confidence 이하로 제한한다.
  - **DP 속도 게이트**: 최근 0.5 s 동안 DP가 tempo의 ±35%(`pop_vel_gate`) 안에서 움직였을 때만 보정한다. 멈추거나 내달리는 DP는 위상에 대한 증거가 아니다. 이 게이트가 mix pop_ 학습 4곡을 83.7%에서 91.4%로 올렸다(빠진 줄 66 → 92, 애드리브 52 → 91).
  - **snap**: DP가 1.5 s 넘게 앞서고 confidence ≥ 0.7인 상태가 2 s 이어질 때만 DP로 건너뛴다. 그 밖에는 단조(monotonic)다.
  - **re-anchor**: DP가 예측에서 3 s 넘게 벗어나고 confidence < 0.5인 상태가 2 s 이어지면 DP를 예측 위치 ±2 s soft prior로 다시 anchor한다(`pop_stuck_*`). 평가에서 효과는 ±1 pt로 작았지만, 길을 잃은 DP의 안전장치로 남겼다.
- **lock**: free-run(템포로 계속 진행)은 DP가 음악을 확실히 따라간 뒤에 시작한다(confidence ≥ `pop_lock_conf` 0.5).
  - mix: DP가 마지막 start/jump/hold 해제 위치에서 `pop_lock_s`(1 s) 이상 나아간 뒤에 시작한다. 그 전에는 classic처럼 DP를 따라가 같은 위치를 발행한다(테스트로 고정). 8 s 방 소음 lead-in에서 이전 버전은 밴드가 오기 전에 8 s를 앞서 나갔다. 지금은 음악 시작 시 표시가 0.04 / 0.16 s로 classic과 같다.
  - vocal: 목소리를 confidence ≥ 0.5로 따라간 뒤에 시작한다. 인트로의 반주 bleed에 relock돼도(confidence ≈ 0) lock되지 않는다. 그 전과, 무음 중 jump/hold 해제 뒤에는 classic처럼 목표 줄에서 목소리를 기다린다.
- **vocal, lock 이후**: coast limit이 없다. 쉼표와 빠진 줄을 템포로 coasting한다. relock prior는 예측 위치 ±2 s(`pop_prior_s`)이고 ref 무음 프레임 비용은 그대로다. relock replay가 coasting으로 이미 진행한 프레임을 두 번 더하지 않게 했다. 잠정 relock이 취소돼도 표시는 물러나지 않는다.
- **전역 재탐색**: lock 뒤에는 **같은 목표가 5 s 뒤 다시 검증될 때만**(`pop_reloc_confirm_s`, 위치는 처음 위치 + 경과 × tempo ±1.5 s, 같은 키) 채택한다. 한 줄짜리 애드리브나 잘못된 자리에서 부른 줄은 한 줄로 끝나지만, 밴드의 건너뛰기·반복은 계속되기 때문이다. lock 전에는 classic과 같다(시작 위치가 틀린 경우).
  - 학습 4곡 vocal pop_에서 3 s 확인은 오검출 2회(AAE 800 ms)였고, 5 s에서 0회(AAE 180 ms)가 됐다. hila의 다른 줄 애드리브는 6 s 길이였다.
  - 단위 실험(`unit/exp_reloc.py`, `exp_skip.py`, mix, 4 seed): 4 s 동안 110 s 지점의 다른 부분을 부른 경우, classic은 재탐색 2회에 최대 오차 68 s였고 pop은 0회에 최대 0.05–0.21 s였다. 실제 40 s 건너뛰기에서는 classic이 3.6–4.6 s 뒤, pop이 8.8–9.8 s 뒤 계속 정답에 머문다(5 s 확인의 대가).
- **보고 `tempo`**는 고정 템포 추정값이다. coasting 중에도 같다. 프롬프터의 외삽 `ref_time + tempo·Δt`가 줄 경계에서 멈추지 않는다. 자동 키 검출과 latency 보정은 classic과 같다.
- **classic 경로는 바뀌지 않았다.** pop 코드는 `style == "pop"`일 때만 실행된다. `pop_*` 값을 모두 바꿔도 classic 출력은 비트 단위로 같다(`test_classic_ignores_the_pop_fields`).
  - 기존 24개 쌍(mix 13 + vocal 11)의 classic 결과는 이번 작업 전 코드 스냅샷(`scratch/baseline-20261013`, `scratch/pop/classic_baseline_all.json`)과 모든 지표에서 같다(AUROC, conf_mean, coast_frac 포함, 차이 0).
  - 단 `docs/eval/track_results.json`(§2 표)은 그 스냅샷 이전에 만든 것이다. 실제 쌍(edelweiss)은 같지만 합성 19개 행은 이미 조금 달랐다(예: synth_kkotbat vocal 88.2 → 86.1, synth_hila vocal 95.7 → 96.6 @300). 현재 classic 수치는 `track_results_styles.json`의 classic 행이다.
- **실시간 전환**: `LiveTracker.set_style()`은 preset 차이 필드(현재는 `style` 하나)만 바꾼다. latency와 다른 override는 유지된다. 위치는 그 자리에서 이어진다. pop으로 바꾸면 현재 템포를 고정 템포 추정의 시작값으로 쓰고, lock돼 있으면 coast limit을 해제한다. classic으로 바꾸면 coast limit이 다시 다음 onset이 된다.

### 8.3 pop 평가 쌍 (`eval-track`의 `steady_<곡>` / `pop_<곡>`)

- **곡**: JamendoLyrics pop 4곡(embers, feel, hila, badside. htdemucs stem, 단어 시작 GT)과 `songs/`의 요루시카 2곡(replicant = `06-レプリカント`, rokugatsu = `07-六月は雨上がりの街を書く`)이다. 요루시카 곡은 읽기만 한다. BS-RoFormer stem을 쓰고, GT는 정렬된 음절 시작 486 / 396개다.
- **튜닝 / hold-out**: embers, feel, hila, replicant로 튜닝했다. **badside와 rokugatsu는 마지막 확인에만 썼다.**
- **템포**: 곡 전체에 하나의 템포비를 준다(0.92–1.08 log-uniform, ref s / live s). 여기에 곡에 걸친 선형 drift ≤ 1%를 더한다(`pop_time_map`, 2 s 구간).
- **라이브 mix**: 분리된 stem으로 다시 만든다. 반주(instrumental)와 보컬을 같은 time map으로 phase-vocoder warp하고, **보컬에만** 공연 변형을 준다(`perform_vocal`). 그 위에 SynthSpec 열화를 더한다(1 s 노이즈 lead-in, EQ tilt, 잔향, pink noise 20 dB, 게인).
  - vocal 모드: 변형된 보컬에 반주 bleed −20 dB를 섞는다.
  - **GT**: 밴드의 time map을 따른다. 빠진 줄도 마찬가지다. 프롬프터는 빠진 줄도 제때 보여줘야 한다.
- `steady_<곡>`: 변형 없음. 고정 템포만 있다.
- `pop_<곡>`: 같은 seed라 time map과 열화가 steady와 같고, 보컬 변형만 다르다.
  - **빠진 줄**: 1–3줄짜리 구간 2개를 무음으로 둔다. 관객 합창(지연 복사 4개, 2.5 kHz 저역통과, 1.5 s 잔향)이 mix에 −10 dB, vocal 모드 마이크에 −20 dB로 들어간다. 첫 줄은 빼지 않는다.
  - **앞당기거나 늦게 부른 줄**: 줄의 50%를 ±0.2–0.4 s 옮긴다. 이웃 줄과 겹치지 않게 했다.
  - **애드리브**: 2줄. 그 줄을 ±3–5반음 pitch shift하거나, 다른 줄의 목소리를 그 길이에 맞춰 넣는다.
- 지표는 §1과 같다. `scratch/pop/run.py`는 pop_ 이벤트를 빠진 줄 / 그다음 2줄 / 애드리브 줄 / 옮긴 줄 / 그 밖으로 나눠서도 잰다.

### 8.4 결과 (최종 기본값, `track_results_styles.json`)

| 집합 | mode | 쌍 | classic @300 / @500 / @1000 | classic AAE ms / lost | pop @300 / @500 / @1000 | pop AAE ms / lost |
|---|---|---|---|---|---|---|
| steady (학습 4곡) | mix | 4 | 97.5 / 99.0 / 99.7 | 81 / 0.000 | **99.9** / 100.0 / 100.0 | 67 / 0.000 |
| steady (학습 4곡) | vocal | 4 | 96.3 / 97.4 / 98.6 | 142 / 0.007 | **99.2** / 99.2 / 99.2 | 131 / 0.008 |
| pop (학습 4곡) | mix | 4 | 79.2 / 89.4 / 93.5 | 281 / 0.031 | **91.0** / 98.3 / 100.0 | 123 / 0.000 |
| pop (학습 4곡) | vocal | 4 | 70.3 / 81.0 / 85.2 | 1616 / 0.102 | **87.0** / 97.6 / 99.2 | 220 / 0.008 |
| steady (**hold-out** 2곡) | mix | 2 | 99.2 / 99.9 / 100.0 | 72 / 0.000 | **100.0** / 100.0 / 100.0 | 68 / 0.000 |
| steady (**hold-out** 2곡) | vocal | 2 | 95.8 / 97.9 / 98.5 | 98 / 0.007 | **99.7** / 100.0 / 100.0 | 67 / 0.000 |
| pop (**hold-out** 2곡) | mix | 2 | 80.3 / 91.2 / 94.7 | 264 / 0.026 | **94.0** / 98.8 / 100.0 | 109 / 0.000 |
| pop (**hold-out** 2곡) | vocal | 2 | 58.3 / 75.6 / 79.0 | 3485 / 0.169 | **83.9** / 99.1 / 100.0 | 149 / 0.000 |
| 기존 쌍 (classical·구간별 템포, §1.1) | mix | 13 | **87.9** / 91.8 / 96.5 | 299 / 0.015 | 72.7 / 78.7 / 84.6 | 3759 / 0.134 |
| 기존 쌍, kkotbat 실제 2쌍 제외 | mix | 11 | **94.7** / – / 98.8 | | 85.0 / – / 97.9 | |
| 기존 쌍 (classical·구간별 템포, §1.1) | vocal | 11 | **91.7** / 94.0 / 97.4 | 275 / 0.011 | 81.8 / 88.4 / 94.4 | 457 / 0.018 |

pop_ 6곡의 이벤트를 구분해서 본 결과(PCO@300 / @1000, 괄호는 latency 중앙값. +는 표시가 늦음):

| 이벤트 | mode | classic | pop |
|---|---|---|---|
| 빠진 줄 안 (관객이 부름) | mix | 56.3 / 84.4 (+228 ms) | 90.8 / 100.0 (+154 ms) |
| 빠진 줄 안 (관객이 부름) | vocal | 13.6 / 32.8 (**+3854 ms**: 다음 줄에서 대기) | **84.5 / 100.0 (+81 ms)** |
| 빠진 줄 다음 2줄 | mix | 83.2 / 96.6 | 92.4 / 100.0 |
| 빠진 줄 다음 2줄 | vocal | 35.7 / 44.1 | 90.6 / 100.0 |
| 애드리브 줄 | mix | 39.4 / 63.4 (+1769 ms) | 94.0 / 100.0 |
| 애드리브 줄 | vocal | 30.3 / 57.9 | 86.9 / 100.0 |
| 앞당기거나 늦게 부른 줄 | mix | 69.3 / 95.9 | 85.8 / 100.0 |
| 앞당기거나 늦게 부른 줄 | vocal | 62.0 / 94.9 | 74.0 / 100.0 |
| 그 밖 | mix | 96.2 / 98.5 | 97.4 / 100.0 |
| 그 밖 | vocal | 90.1 / 94.3 | 94.0 / 99.0 |

| pair | mode | classic @300 / @1000 (재탐색) | pop @300 / @1000 (재탐색) |
|---|---|---|---|
| pop_embers | mix / vocal | 79.4 / 93.7 (1) · 72.5 / 85.7 (3) | 93.1 / 100.0 (0) · 91.0 / 97.9 (1) |
| pop_feel | mix / vocal | 64.5 / 88.5 (0) · 56.1 / 76.6 (4) | 80.0 / 100.0 (0) · 71.3 / 100.0 (0) |
| pop_hila | mix / vocal | 80.4 / 96.0 (0) · 74.8 / 91.9 (2) | 91.0 / 100.0 (0) · 87.3 / 100.0 (0) |
| pop_replicant | mix / vocal | 92.4 / 95.9 (0) · 78.0 / 86.4 (1) | 100.0 / 100.0 (0) · 98.4 / 99.0 (1) |
| pop_badside (hold-out) | mix / vocal | 86.1 / 96.1 (0) · 69.5 / 82.3 (2) | 92.7 / 100.0 (0) · 88.0 / 100.0 (0) |
| pop_rokugatsu (hold-out) | mix / vocal | 74.5 / 93.2 (0) · 47.0 / 75.8 (2) | 95.2 / 100.0 (0) · 79.8 / 100.0 (0) |

- **pop 스타일의 목적**: 밴드가 계속 가는 동안 표시가 멈추지 않는 것. 이 목적은 hold-out 곡에서도 그대로 확인됐다. vocal 모드의 빠진 줄은 classic이 다음 줄에서 평균 3.9 s 기다렸지만 pop은 +0.08 s로 지나간다. pop_ 쌍에서 lost(|오차| > 2 s)는 0–0.8%이고 PCO@1000은 97.9–100%다. 남은 vocal 재탐색 2회(embers, replicant)는 lock 전, 인트로 bleed에서 classic과 똑같이 일어난다.
- **@300이 vocal에서 84–87%에 머무는 이유**는 대부분 줄 단위 ±0.2–0.4 s 프레이징 변형이다. 목소리만 듣는 vocal 모드에서 밴드의 위상은 여러 줄의 평균으로만 알 수 있다. pop은 가수가 앞당기거나 늦게 부른 줄에서도 밴드 쪽에 가깝게 머문다(74%, classic 62%). @500에서는 97–99%다.
- **trade-off: 템포가 바뀌는 곡에서는 classic이 낫다.** 기존 쌍(구간별 0.8–1.25배, 실제 다른 연주)에서 pop은 mix −15 pt, vocal −10 pt(@300)다. 극단적인 템포비(kkotbat r2s/s2r, 0.55배·1.8–5배)는 템포 범위 0.7–1.4 밖이라 무너진다(6.9 / 2.8%). 실제 Edelweiss는 mix에서 classic 84.6 → pop 46.2(@1000은 둘 다 100), vocal에서 76.9 → 88.5다. 그래서 **기본값은 classic**이고, 스타일은 곡마다 정한다.
- **CPU**: pop의 트래커 step은 평균 ≤ 0.22 ms/frame이다. p99 중앙값은 0.41 ms로 classic과 같고, 최댓값은 2.56 ms(kkotbat, re-anchor가 든 블록)다. 40 ms 예산의 7% 이하다.

### 8.5 튜닝 과정 (학습 4곡 pop_ 평균 PCO@300, mix / vocal)

| 단계 | mix | vocal | 메모 |
|---|---|---|---|
| classic | 79.2 | 70.3 | vocal 빠진 줄 15%, lost 10% |
| v1: 등속 예측 + conf 가중 gain 0.04, snap(conf ≥ 0.5, 1 s) | 79.8 | 80.6 | AAE 1173: 다른 줄 애드리브로 재탐색 오검출. pitch 애드리브에서 snap |
| + 속도 변화 제한 `pop_rate`, 템포 표본 게이트·Huber | 80.2 | 79.1 | mix lost 0 |
| + 재탐색 확인(sweep 3 s → 6 s, 최종 5 s) | 80.2 | 78.7 → 79.9 | vocal AAE 814 → 199, 오검출 2 → 0회 |
| + DP 속도 게이트 | 91.4 | 82.6 | mix 빠진 줄 66 → 92 |
| PLL식 적분 템포 항(ki 2e-5–2e-4) | ≤ 94.2 | ≤ 83.6 | 같은 gain의 ki = 0보다 낫지 않거나(mix) 최선 설정 대비 +1 pt 안쪽(vocal), 큰 ki는 손해: 넣지 않음 |
| + mix lock, lock 뒤에만 템포 추정 | 91.3 | 82.8 | 8 s lead-in 문제 해결. lock만 넣었을 때 steady mix 96.7 → 템포 수정 뒤 100 |
| + 템포 prior 15, alpha 0.01 | 91.8 | 86.4 | |
| + vocal lock에 confidence ≥ 0.5, 재탐색 확인은 lock 뒤에만 | 91.8 | 87.8 | 인트로 bleed lock 해결 |
| **최종**: rate 0.5, gain 0.08, 속도 게이트 0.35 | 91.0 | 87.0 | 구간별 템포 쌍(synth_hila_skip·feel·kkotbat) mix 76.5 → 84.8, vocal 66.7 → 84.6 |

- 마지막 단계는 pop_에서 손해가 1 pt 안쪽이다. 대신 실제 밴드의 템포 흔들림(구간별 템포 쌍을 대리 지표로 봤다)에 훨씬 강해서 이 값을 골랐다.
- 다른 sweep에서는 차이가 ±1 pt였다. 그 대상은 re-anchor 임계, relock prior ±1 / 2 s, snap 조건이다. 기록은 `scratch/pop/logs/sweep*.log`에 있다.
- 학습 곡의 결과(91.0 / 87.0)와 hold-out 곡(94.0 / 83.9)이 비슷해 과적합 신호는 없다. 다만 hold-out은 2곡뿐이다.

### 8.6 알려진 한계 (pop)

- **템포가 정말 바뀌는 곡**(rubato 인트로, 브리지의 템포 변화, 끝의 ritardando)은 pop이 몇 초 늦게 따라간다. 이런 곡은 classic을 쓴다. 공연 중에도 콘솔에서 바꿀 수 있다.
- **실제 구조 변경**(건너뛰기, 반복)의 복구가 classic보다 약 5 s 늦다(재탐색 확인).
- **vocal 모드의 첫 lock 전**은 classic과 같다. 인트로 bleed로 DP가 앞서 나가는 classic의 약점도 그대로다(steady embers·replicant vocal 재탐색 1회).
- **가수가 줄마다 일정하게 앞당기거나 늦게** 부르면 pop은 밴드 쪽에 머문다. 가수를 그대로 따라가는 표시를 원하면 classic이다.
- 평가 데이터는 합성이다. stem을 다시 합치고 보컬만 변형했다. 실제 라이브의 편곡 차이, 관객 소음, 마이크 bleed는 Edelweiss 실제 쌍 정도로만 확인했다. 실제 pop 라이브 녹음으로 확인하는 것이 다음 과제다.
