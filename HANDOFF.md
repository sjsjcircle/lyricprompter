# 작업 인수인계

> 다음 세션에서 이 파일부터 읽고 이어서 진행하세요. 맨 아래에 붙여 넣을 재개 프롬프트가 있습니다.

## 최신 상태 (2026-10-13, 3번째 세션: 사용자 테스트 피드백 반영)

사용자가 실제 J-pop 두 곡(`songs/06-レプリカント`, `songs/07-六月は雨上がりの街を書く`)으로 테스트한 피드백을 반영했다. 설계 명세는 `.omc/plans/2026-10-13-feedback.md`, 이번 라운드 전 코드 스냅샷은 `scratch/baseline-20261013/`에 있다(git 없음, `diff -ru`로 비교).

- **타이밍 편집**(피드백 1): 원인은 줄을 이웃 음절과 겹치지 않는 범위에서만 통째로 옮기던 `move_line`이었다(07의 30·31번 줄이 잠겨 Set to playhead가 거절됨). 이제 이웃 줄이 압축되거나(꼬리 자르기 / 비율 압축) 밀려나고(`pushed` 표시), 0초 앞이나 곡 끝 너머만 거절한다. Set start / Set end, Undo(Cmd+Z, 서버 메모리 100단계), Tap 모드(T 키)를 추가했다. API: `PUT /api/songs/{id}/lines/{idx}`에 `end`, 응답에 `pushed`, `POST /api/songs/{id}/undo`.
- **classic / pop 추적 스타일**(피드백 2): `SongMeta.style`(기본 classic), `prep --style`, 콘솔 Session의 Style(곡에 저장하고 공연 중이면 바로 전환), `POST /api/session/style`, `PUT /api/songs/{id}/style`, `simulate --style`, `eval-track --styles`. classic 경로는 기존 24쌍에서 지표가 하나도 바뀌지 않았다. pop은 템포 예측기에 DTW를 약하게 보정으로 쓴다. 새 합성 밴드 공연(`steady_*`, `pop_*`: 템포 일정 + 줄 누락·애드리브·박자 밀림)에서 PCO@300이 mix 79.6 → 92.0, vocal 66.3 → 85.9이다. 대신 템포가 바뀌는 기존 쌍에서는 나빠진다(mix 87.9 → 72.7). 자세한 것은 `docs/eval/TRACKING.md` §8, `docs/eval/track_results_styles.json`.
- **프롬프터 보기**(피드백 3·4): display 설정 `layout`(line | page), `page_lines`(1–8, 기본 4), `wipe`. Multiple lines는 문단(빈 줄, 섹션 머리말) 단위 페이지이고, 긴 문단은 고르게 나눈다(`common.js` `paginate`). 프롬프터 단축키 L / W를 추가했다.
- **질문 5(일본어 노래의 한국어 발음 표시)**: 실험만 했다(`scratch/q5/REPORT.md`). 결론은 두 가지다.
  - 지금은 언어를 Korean으로 두고 한국어 발음 가사를 넣으면 된다. 음절 onset의 96–99.6%가 일본어 원문 정렬과 같은 프레임에 놓였다.
  - 원문과 발음을 함께 보여주려면 "일본어 줄 뒤 한글 줄 = 발음(유사도로 확인) + 자동 생성 발음" 형식이 좋다. 엄격한 번갈아 입력은 줄 하나만 빠져도 전부 어긋난다. 구현 추정은 1.5–2일이다.
  - 실험 중 OpenJTalk 오독을 찾았다(06 `真面`→まほ, 07 `結局君だけ`의 君→くん, `i`→イ). text_review 플래그가 붙지 않았으므로 후리가나로 지정해야 한다.
- **테스트**: fast 806개 통과, ruff clean. 변경과 관련된 slow 테스트 14개도 통과했다.
- **남은 과제(선택)**
  - pop을 실제 밴드 라이브 녹음(원곡과 쌍)으로 검증하기. 지금은 합성 공연만 썼다.
  - Tap 모드의 반응 시간 보정(옵션).
  - 질문 5의 (2)안 구현(사용자 결정 필요).
  - `docs/eval/track_results.json`과 TRACKING.md §2 표는 이번 라운드 전부터 합성 19행이 현재 코드와 조금 달랐다. 최신 값은 `track_results_styles.json`에 있다.
  - OpenJTalk 오독 검출(text_review) 보강.

---
(아래는 2번째 세션 종료 시점 기록)

## 2번째 세션 종료 시점 상태 (2026-10-13)

**구현·통합·최종 리뷰 완료.** 사용법은 [README.md](README.md)를 보세요.

- **테스트**: 789개 통과(fast + slow), ruff clean.
- **최종 교차 리뷰**: 4개 관점(자동 키, 처음 쓰는 사용자 E2E, 모듈 간 통합, 실시간 견고성)에서 확인된 결함은 수정했고, 반박되거나 사소한 항목은 제외했다. 주요 수정:
  - 음악 시작 전 잡음·박수로 키가 잘못 바뀌던 문제
  - DC 오프셋 입력에서 vocal 모드가 고장 나던 문제
  - inf 샘플 하나로 모든 프롬프터가 멈추던 문제
  - 복사·이름 변경한 곡 패키지의 id 불일치
  - 다른 곡의 Go 버튼이 현재 세션 위치를 옮기던 문제
  - 서버 재시작 후 표시 설정이 사라지던 문제
  - Hold 때 프롬프터가 뒤로 튀던 문제
  - vocal 모드에서 구절을 건너뛴 뒤 재탐색이 느리던 문제(최대 32 s → 약 10 s)
- **추적 평가**(`docs/eval/TRACKING.md`): mix PCO@300 87.9% / @1000 96.5%, vocal 91.7% / 97.4%. 자동 키는 오검출 0회이고, 실제 Edelweiss mix(+2)에서 preset 없이 73.1% / 88.5%다(자동 키를 끄면 11.5%).
- **준비된 곡**: `songs/{edelweiss,kkotbat,tts-ja}`
- **남은 과제(선택)**
  - 일본어 실제 노래로 정렬 검증
  - 실제 오디오 인터페이스·마이크 테스트(TCC 권한 필요)
  - 보컬 단독 모드의 키 판정 개선(라이브 튜닝 추정 등)
  - 영어가 아닌 라틴 문자 가사(이탈리아어 등)에 대한 prep 경고
  - git 저장소 초기화와 커밋(아직 안 함)

---
(아래는 1번째 세션 종료 시점 기록)

## 1. 목표 (사용자 요구사항)

[laurenceyoon/live-lyrics-tracking](https://github.com/laurenceyoon/live-lyrics-tracking)을 참고해 **자동 가사 프롬프터**를 새로 만든다. 레퍼런스는 음절 분할과 타이밍을 사람이 직접 준비해야 한다.

- **입력**: 전체 가사 텍스트(주로 한국어·일본어·영어, 한 곡 안에서 섞일 수 있음)와 원본 음원.
- **오프라인 준비(자동)**: 보컬 분리 → 음절/모라 분할 → 강제정렬 → 줄·단어·음절 타이밍 생성. 수작업 타이밍은 없어야 한다.
- **실시간 추적**: reference와 라이브 오디오를 비교해 위치를 자동으로 따라간다. 운영자가 두 모드 중 하나를 고른다.
  - (a) 라이브 = 반주 포함 믹스 ↔ reference = 원본 전체 음원
  - (b) 라이브 = 보컬 단독(보컬 마이크/채널) ↔ reference = 분리한 보컬 stem
- **실행 환경**: Apple M5 Pro, 24GB, macOS 26.3, Homebrew, ffmpeg 8.1, node 26, uv 0.13(설치 완료).

## 2. 진행 상황

| 단계 | 상태 |
|---|---|
| 레퍼런스 repo 분석 | ✅ 완료 (`_reference/live-lyrics-tracking`, 오디오 포함 클론) |
| 도구 조사 6개 주제 (조사 + 반박검증) | ✅ 5개 완료, ⚠️ "가사 강제정렬"은 조사 대부분 완료·검증 전 중단 |
| 설계 확정 | ⏳ 초안만 있음 (아래 4절) |
| 구현 / 테스트 / 평가 | ⏳ 시작 전 |

## 3. 저장된 자료 위치

| 경로 | 내용 |
|---|---|
| `docs/research/research_<topic>.json` | 주제별 조사 보고서: 권고안, 대안, 핵심 사실(출처 포함), 코드, 주의점. 주제는 separation, syllables, tracking, environment, reference-repo |
| `docs/research/verify_<topic>.json` | 위 보고서를 반박 검증한 결과. **보고서와 다르면 검증 쪽을 우선** |
| `docs/research/research_alignment_PARTIAL_notes.md` | 강제정렬 조사의 실험 출력(중단 전에 건진 것) |
| `docs/research/artifacts/` | 조사 에이전트가 만든 프로토타입과 메모. 주요 파일은 아래 |
| ├ `environment/proj/pyproject.toml`, `uv.lock` | 이 Mac에서 설치와 import까지 검증한 의존성 세트 |
| ├ `syllables/scripts/lyricsprep.py` | 한·일·영 가사 → Song/Line/Word/Syllable/AlignUnit 프로토타입(454줄, span 왕복 테스트 통과) |
| ├ `alignment/scripts/` | MMS_FA 정렬, Qwen3-ForcedAligner 정렬, Jamendo 평가 스크립트 |
| ├ `reference-repo/scripts/oltw_clean.py` | 레퍼런스 OLTW 버그를 고친 클린 스케치 |
| └ `tracking/proto/` | 벡터화 OLTW 프로토타입 |
| `_reference/live-lyrics-tracking/` | 레퍼런스 repo 클론. **비공개·라이선스 없음 → 코드와 .pt 모델 복사 금지, 로컬 참고와 평가 데이터로만 사용** |
| `data/research-testsets/jamendo/` | JamendoLyrics 영어 4곡: mp3, 단어 단위 정답, htdemucs 분리 결과 |
| `data/research-testsets/tts-multilingual/` | macOS TTS로 만든 한·일·영 혼합 테스트 음성(줄 단위 정답) |
| `.cache/models/` | 이미 받은 모델 4.2GB: Qwen3-ForcedAligner-0.6B, MMS_FA(torch hub), htdemucs, kresnik 한국어 CTC. 쓰려면 `HF_HOME=.cache/models/hf`, `TORCH_HOME=.cache/models/torch` |

## 4. 조사 결론 요약 (설계 초안)

### 4.1 환경
- **Python 3.12, uv 프로젝트 venv.** miniconda base(3.14)는 쓰지 않는다.
- 검증된 조합(`artifacts/environment/proj`): torch 2.14.1 + torchaudio 2.11.0 + torchcodec 0.17, audio-separator 0.47.0, transformers 5.19, fastapi 0.143.
- **필수 패키징 수정 3가지**
  1. `[tool.uv] override-dependencies = ["samplerate>=0.2.4"]` — 0.1.0의 dylib는 x86 전용이다.
  2. `audioread`를 명시적으로 추가한다. audio-separator가 선언 없이 import한다.
  3. ctc-forced-aligner는 PyPI가 아니라 **git**에서 설치한다(PyPI 이름 충돌).
- **librosa 버전 결정 필요**
  - 1.0을 쓰면 audioread fallback이 없어 m4a/AAC를 못 읽는다. **입력은 항상 ffmpeg으로 WAV/FLAC로 먼저 변환**해야 한다.
  - synctoolbox가 필요하면 librosa 0.11, numpy 2.4.x, numba 0.65.x로 고정해야 한다.
  - 권장: synctoolbox 없이 간다(평가 GT는 이미 CSV로 있다).
- **실시간 경로는 CPU, 오프라인은 MPS.** 실시간 resampling은 `soxr` LQ/MQ 품질(지연 고려).

### 4.2 보컬 분리
- **도구**: `audio-separator` 0.47.0 (MIT), Python API.
  - `AUDIO_SEPARATOR_MODEL_DIR`를 영구 경로로 지정한다. 기본값 `/tmp`는 재부팅 때 지워진다.
  - `use_soundfile=True`로 쓴다.
- **기본 모델**: `bs_roformer_vocals_resurrection_unwa.ckpt` (204MB, bleed가 적음). 라이선스가 확실한 대안은 `vocals_mel_band_roformer.ckpt` (Kim, MIT).
- **리드/백보컬 분리(옵션)**: `bs_roformer_karaoke_frazer_becruily.ckpt`. 보컬 단독 모드의 reference로는 karaoke 리드 stem(de-reverb 없음)이 유력하다.
- **빠른 fallback**: demucs-mlx / htdemucs.
- **라이선스 주의**: diffq(CC BY-NC)가 macOS 필수 의존성이다(Demucs 경로에서만 import). de-reverb anvuew 모델은 GPL.

### 4.3 가사 텍스트 처리 (음절)
- **데이터 모델**: `Song → Line → Word → Syllable(표시 단위) → AlignUnit(정렬 단위)`. 각 노드가 원문 char span을 가진다. AlignUnit은 `reading`과 `roman`(MMS_FA 어휘 [a-z'])을 갖고, 시간은 index로 역매핑한다.
- **일본어**: `pyopenjtalk-plus`(MIT). 가사 읽기 정확도가 가장 좋았다.
  - span은 `g2p_mapping()`으로 잡는다.
  - 동형이음어(今日, 何, 明日)는 `run_mecab_nbest_features()`로 후보를 만들고 CTC 점수로 고른다.
  - 宇宙=そら 같은 의훈은 자동 추론이 안 되므로 사용자 후리가나 입력(`漢字(かんじ)`)을 지원하고 저신뢰 단어로 표시한다.
- **한국어**: 한글 한 글자 = 한 음절 = 한 AlignUnit. `uroman`을 음절별로 적용한다(MMS 학습과 같은 방식). g2pk2(연음)는 기본에서 빼고 옵션으로 둔다.
- **영어**: 음절 수는 CMUdict, 표기 분할은 pyphen(GPL/LGPL 주의).
- **처리할 것**: 숫자·기호 정규화, `(oh oh)` 같은 백보컬 괄호, `[Chorus]` 섹션 표시, 빈 줄(연 구분), `x2` 같은 반복 표기.

### 4.4 강제정렬 (부분 결과, 검증 전)
- **torchaudio 2.11의 `MMS_FA`와 `forced_align`은 아직 존재하고**, torch 2.14.1에서 동작한다.
  - `forced_align`은 **CPU 전용**이다(MPS 미구현).
  - emission은 MPS에서 4분 곡 기준 2.8초가 걸린다. CPU로는 14초.
- **MMS_FA 전역 정렬 + 보컬 활동 gating이 1차 후보.**
  - gating이 핵심이다. Jamendo "Embers" 곡에서 평균 오차(AAE)가 14.7초 → 2.0초로 줄었다.
  - 영어 4곡 단어 시작 오차 중앙값은 0.06–0.24초.
- **Qwen3-ForcedAligner-0.6B** (Apache-2.0, 1.8GB, ko/ja/en 지원)
  - 30–60초 창 단위에서는 매우 정확하다(Cortez 곡 AAE 0.07초).
  - 곡 전체를 한 번에 넣으면 크게 드리프트한다(240초에서 최대 36초 오차).
  - 따라서 **창 단위 보정**에 쓴다.
- **MMS와 Qwen의 불일치(>0.5초)를 신뢰도 신호로** 쓰면, MMS 오차 >1초 14건 중 13건을 잡았다. 이 하이브리드 평가는 도중에 중단됐다.
- 한국어 네이티브 CTC(kresnik)는 MMS_FA + uroman보다 낫지 않았다.
- **남은 일**: 하이브리드 평가 완료, 한국어·일본어 노래로 검증(꽃밭에서 GT, TTS), 저신뢰 줄 검토 UI.

### 4.5 실시간 추적
- **직접 구현한다.** pymatchmaker 0.3.0은 numpy<2와 Python<3.13을 요구해 같은 환경에 둘 수 없다.
  - 대안: pymatchmaker **develop 브랜치**의 `matchmaker/dp/oltw_dixon.py`(Apache-2.0, 레퍼런스 저자가 작성, Java MATCH로 검증)를 고지문과 함께 vendoring.
- **핵심 알고리즘**: 라이브 프레임마다 한 열씩 갱신하는 OLTW. min-plus prefix scan으로 벡터화하고(numpy로 충분), 가중치 (1,1,2), i+j로 정규화, ±10초 창.
- **특징 baseline (라이선스 문제 없음)**: Chroma + MFCC(5). 논문 기준 AAE 453ms, 중앙값 144ms.
  - chroma 필터뱅크는 곡마다 고정 tuning으로 한 번만 만들고 ref와 live에 같은 것을 쓴다. 프레임마다 tuning을 다시 추정하면 흔들린다.
- **ref 특징**은 live와 **똑같은 스트리밍 코드 경로**로 계산한다.
- **MVP 범위**: OLTW, 모드 (b)에서 보컬 활동 gating 기반 coasting(간주 구간), 줄 단위 UI + 리드 타임, 운영자 조작(다음/이전 줄, N번째 줄로 재동기, 홀드).
- **2단계**: 전역 재탐색(coarse subsequence DTW), 신뢰도, 음성학적 특징(MMS emission / PPG), 조옮김(chroma shift).
- **레퍼런스 OLTW 버그 목록**: `research_reference-repo.json`에 있다. 0.98 대각 곱, `<=` off-by-one, `ref_pointer` 출력, 처음 5초 고정 등.

### 4.6 평가 데이터 (레퍼런스 repo)
- **영어**: `resources/lyrics/Edelweiss.lrc`(단어 단위) ↔ `resources/audio/Edelweiss.wav`.
- **한국어**: `꽃밭에서`. score CSV의 음절 시각을 `alignments/꽃밭에서_offline.csv` 경로로 `_ref.wav` 시간에 매핑하면 실연 음절 GT가 된다. `_score.wav`(합성)와 `_ref.wav`(실연)는 추적 평가 쌍으로도 쓸 수 있다.
- **일본어**: GT가 없다. TTS 합성으로 만들거나 실제 곡이 필요하다.
- **지표**: 허용오차 200/300/500/750/1000ms 이내 비율, AAE, 중앙값.
- **추적 합성 평가**: 시간 늘이기, 피치 시프트, 노이즈를 넣어 GT warp를 알고 있는 "가짜 라이브"를 만든다.

## 5. 다음 단계 (권장 순서)

1. **설계 확정**: 4절 초안을 바탕으로 정리한다. 아래 항목은 근거와 함께 기본값으로 정하고 진행한다.
   - librosa 1.0 vs 0.11
   - 단일 venv vs 분리 전용 venv
   - 프론트엔드 스택(빌드 없는 vanilla JS vs Vite)
2. **환경 구성**: `uv init` → `artifacts/environment/proj/pyproject.toml`을 기준으로 의존성 설치.
3. **구현**: 오프라인 준비 CLI(`prep`) → 실시간 추적 엔진 → FastAPI + WebSocket 서버 → 프롬프터·운영자 콘솔 UI → 평가 CLI.
4. **평가와 반박형 리뷰**: Edelweiss, 꽃밭에서, Jamendo, TTS로 수치를 확인한다.
5. **선택**: 레퍼런스 저자(Jiyun Park, KAIST MACLab)에게 코드·모델 사용 허락이나 LICENSE를 요청한다. 사용자가 collaborator다.

## 6. 재개 프롬프트 (새 세션에 붙여 넣기)

```
HANDOFF.md를 읽고 자동 가사 프롬프터 작업을 이어서 진행해줘.
docs/research/의 조사·검증 결과(검증 결과 우선)를 바탕으로 설계를 확정하고,
강제정렬 조사의 남은 부분(MMS+Qwen 하이브리드 평가, 한국어/일본어 검증)을 마무리한 뒤 구현을 시작해줘.
```
