# lyricprompter — 설계 (v1)

자동 가사 프롬프터. 가사 텍스트(ko/ja/en, 혼합 가능)와 원곡 음원만 넣으면 다음을 자동으로 수행한다.

1. 오프라인 준비: 보컬 분리 → 음절/모라 분할 → 강제정렬 → 줄·단어·음절 타이밍 생성
2. 라이브: reference와 라이브 오디오를 실시간으로 정렬(OLTW)해서 현재 가사 위치를 프롬프터에 표시

조사 근거는 `docs/research/`에 있고, 보고서와 검증 결과가 다르면 `verify_*.json`이 우선한다. 레퍼런스 repo(`live-lyrics-tracking/`)는 **비공개이고 라이선스가 없다.** 코드와 모델 파일은 복사하지 않고, 평가 데이터와 설계 참고로만 쓴다.

## 0. 원칙

- **환경**: Python 3.12, uv venv (`.venv`)
  - 오프라인 작업(분리, 정렬)은 MPS에서 돌린다.
  - 실시간 경로는 CPU와 numpy만 쓴다(torch 없음).
- **기준 시간축**: 모든 시간은 초(float)로 다루고, **reference 시간축(원곡 mix.wav 기준)**을 쓴다. stem들은 mix와 샘플 단위로 정렬되어 있어야 한다(분리는 길이를 보존한다).
- **분석 표준**: 16 kHz mono, hop 640(40 ms, 25 fps). 실시간 추적과 reference 특징은 **같은 스트리밍 코드 경로**로 계산한다.
- **모델 캐시**: 프로젝트 `.cache/models/` 아래에 둔다. `HF_HOME`, `TORCH_HOME`, `AUDIO_SEPARATOR_MODEL_DIR`은 `config.py`가 `setdefault`로 지정한다.
- **입력 변환**: 오디오는 항상 ffmpeg으로 canonical WAV(44.1 kHz stereo, float/PCM24)로 먼저 변환한다. librosa 1.0은 m4a/AAC를 읽지 못한다.

## 1. 패키지 구조 (`src/lyricprompter/`)

| 모듈 | 책임 | 담당 |
|---|---|---|
| `config.py` | 경로, 상수, 캐시 env | 공용(완료) |
| `songfile.py` | `songs/<id>/` 패키지 load/save, song.json 스키마 | 공용(완료) |
| `text/model.py` | 데이터 모델 dataclass와 (de)serialize | 공용(완료) |
| `text/` 나머지 | 가사 파싱, 음절/모라 분할, 로마자화 | A |
| `audio_io.py` | ffmpeg 변환, mono 로드/리샘플, 저장 | B1 |
| `prep/separate.py` | audio-separator 래퍼 | B1 |
| `prep/pipeline.py` | 오프라인 준비 전체 오케스트레이션과 `prep` CLI | B1 |
| `prep/vad.py` | stem 기반 보컬 활동(gate) | B2 |
| `prep/align.py` | MMS_FA gated 전역 정렬과 신뢰도 | B2 |
| `prep/refine.py` | Qwen3-ForcedAligner 창 단위 보정(하이브리드) | B2 |
| `eval/datasets.py`, `eval/align_eval.py` | 정렬 평가 | B2 |
| `features/extractor.py` | 스트리밍 특징(chroma+MFCC+RMS) | C |
| `tracking/oltw.py` | 벡터화 frame-sync OLTW | C |
| `tracking/tracker.py` | LiveTracker(상태기계, coasting, 템포, 신뢰도, 조작) | C |
| `timeline.py` | ref 시간 → 줄/음절 커서 | C |
| `eval/track_eval.py` | 추적 평가(합성 라이브와 실제 쌍) | C |
| `live/audio_input.py` | sounddevice 입력, 파일 시뮬레이션 | D |
| `live/session.py` | 입력 → 특징 → 트래커 스레드, 상태 스냅샷 | D |
| `server/app.py`, `server/static/*` | FastAPI REST + WS, 콘솔과 프롬프터 UI | D |
| `cli.py` | 서브커맨드 디스패치(각 모듈의 `cli(argv)` 호출) | 공용(완료) |

테스트는 `tests/test_<module>.py`에 둔다. 모델 다운로드나 실제 음원이 필요한 테스트는 `@pytest.mark.slow`를 붙인다.

## 2. 데이터 모델 (`text/model.py`, 구현 완료 — 수정 시 전 모듈 영향)

```
Song(meta: SongMeta, lines: list[Line])
Line(idx, text, raw, stanza, section, backing, words, start, end, confidence, needs_review, flags)
Word(text, span, lang, kind, reading, syllables, needs_review)       # kind: lex|num|sym|backing
Syllable(text, span, units, start, end)                               # 표시 단위 (프롬프터에서 칠해지는 단위)
AlignUnit(reading, roman, start, end, score)                          # 음향 단위 (ko 음절 / ja 모라 / en 음절)
```

- span은 `Line.text` 기준 `[start, end)` 문자 인덱스다.
- `roman`은 MMS_FA 어휘 `[a-z']`만 쓴다. 빈 문자열이면 음향 토큰이 없다는 뜻이고, 시간은 이웃에서 보간한다.
- 시간 필드는 정렬 전에는 `None`이다.
- `xN` 반복 표기는 **파싱 단계에서 실제로 부르는 순서대로 줄을 복제해 펼친다.** 원본 줄 인덱스는 `flags`에 `repeat_of:<i>`로 남긴다.

## 3. 곡 패키지 (`songs/<song_id>/`)

```
song.json            # schema=1: {"schema":1,"meta":{...},"lines":[...]}
lyrics.txt           # 사용자가 입력한 원문
audio/mix.wav        # canonical 44.1k stereo
audio/vocals.wav     # 분리 보컬 (mix와 동일 길이)
audio/instrumental.wav
audio/lead.wav       # (옵션) karaoke 모델 리드 보컬
cache/feat_<hash>.npz  # reference 특징 캐시 (FeatureConfig 해시 키)
prep_report.json     # 단계별 소요시간, 모델, 줄별 신뢰도, 플래그
```

`SongMeta`의 필드는 다음과 같다.

- 기본 정보: `id`, `title`, `artist`, `default_lang`, `duration`
- `tuning`: `librosa.estimate_tuning`으로 구한 mix의 튜닝(semitone 소수)
- `stems`: 상대경로 목록
- `prep`: 사용한 모델, 버전, 옵션

## 4. 오프라인 준비 파이프라인

1. **ffmpeg 변환** → `audio/mix.wav`
2. **보컬 분리**: `audio-separator`
   - 기본 모델: `bs_roformer_vocals_resurrection_unwa.ckpt`
   - 라이선스가 확실한 대안: `vocals_mel_band_roformer.ckpt`
   - `use_soundfile=True`, 출력은 vocals와 instrumental
   - 옵션 `--lead`: `bs_roformer_karaoke_frazer_becruily.ckpt`를 mix에 돌려 리드 보컬을 따로 만든다.
3. **가사 파싱**: `text.parse_lyrics(raw, default_lang)`
4. **정렬 (B2)**
   - **emission**: vocals 16k를 MMS_FA(with_star)에 넣어 MPS에서 계산한다. 30초 창에 앞뒤 2초 문맥을 붙인다.
   - **gating**: vocal stem RMS가 p99−35 dB보다 낮은(비활성) 프레임은 실제 토큰의 log-prob에서 15를 뺀다. 판정은 0.3초 dilate한다.
   - **star 토큰**: 곡 앞뒤와 줄 사이에 `*`를 넣는다.
   - **강제정렬**: `torchaudio.functional.forced_align`은 CPU 전용이다. TokenSpan을 AlignUnit → Syllable → Word → Line 순서로 집계한다.
   - **줄 신뢰도**: 토큰 점수(prob) 평균. 0.10 미만이면 `needs_review`.
   - **refine** (옵션, 기본값은 평가 결과로 결정): 저신뢰 줄은 Qwen3-ForcedAligner-0.6B로 앞뒤 신뢰 줄 사이 창(≤30초)에서 다시 정렬하고, 신뢰 범위 안에 들어오면 채택한다. MMS와 Qwen이 0.5초 넘게 어긋나면 flag를 단다.
   - **시간 정리**: 음향 토큰이 없는 unit은 보간하고, 단조성을 보장하며, 겹침을 정리한다.
5. **튜닝 추정과 저장**: song.json과 prep_report.json을 쓴다.
6. **진행률**: stdout에 `PROGRESS <stage> <0-100> <msg>` 줄로 낸다. 서버가 이것을 파싱한다.

**구현 후 데이터로 정한 정렬 기본값** (`AlignOptions` / `RefineOptions`, 근거는 [eval/ALIGNMENT.md](eval/ALIGNMENT.md))
- **gate 임계값은 적응형**이다. 잡음 바닥(프레임 dB의 10번째 백분위수)에 20 dB를 더하고, [loud−55, loud−35] dB 범위로 자른다. 위의 고정값 p99−35 dB는 쓰지 않는다.
- **임계값이 둘로 나뉜다.** `needs_review`는 confidence < 0.15일 때, Qwen 보정 대상은 < 0.10일 때다.
- **Qwen 결과 채택 조건**: MMS와 다르고, Qwen confidence가 0.15 이상일 때만 채택한다.
  - 두 결과가 일치하면 검토 표시를 지운다.
  - Qwen 출력이 퇴화했으면(한 줄이 한 시각으로 뭉침, 창 끝에 고정, 보컬 밖) 채택하지 않고 표시도 지우지 않는다.
- **refine="auto"는 best effort**다. Qwen이 실패해도 MMS 결과는 유지한다.
- **음절 끝**은 loud−20 dB 지속 마스크까지 늘린다. **줄의 첫 onset**은 loud−25 dB 마스크에서 최대 0.5 s 앞으로 당긴다.

## 5. 특징과 추적 (C)

### FeatureExtractor
- **설정**: `FeatureConfig(sr=16000, hop=640, n_fft=2048, kind="mix"|"vocal", tuning=0.0, transpose=0, n_mfcc=13, w_chroma, w_mfcc)`
- **chroma**: 필터뱅크(`librosa.filters.chroma(..., tuning=T)`)는 **한 번만 만든다.** 인과(causal) 창 rfft를 쓰고 log 압축 후 L2 정규화한다.
- **MFCC**: c1..c12, L2 정규화.
- **그룹 결합**: 그룹별로 L2 정규화한 뒤 가중치를 곱해 이어 붙인다(전체 cosine 거리).
- **rms_db**: 프레임별로 출력한다(활동 판정용).
- **같은 코드 경로**: `push(samples) -> FeatureFrames(feats[n,D], rms_db[n])`. `compute_offline(y, cfg)`도 내부적으로 `push`를 블록 단위로 호출한다.

### FrameSyncOLTW (`tracking/oltw.py`)
- 라이브 프레임마다 ref 창 열 하나를 갱신한다.
  - `D_t[i] = min(D_{t-1}[i-1]+2d, D_{t-1}[i]+d, D_t[i-1]+d)`
  - min-plus prefix scan으로 벡터화한다(`docs/research/artifacts/tracking/proto/`).
- 정규화는 `D/(|i-i0|+t-t0+1)`, 창은 ±10초, 위치는 단조로 움직이고 프레임당 최대 3칸 전진한다.
- `anchor(i0, prior=None)`로 재시작하고, `backward_path`로 템포 ρ를 추정한다.

### LiveTracker (`tracking/tracker.py`)
- **상태**: `WAITING → TRACKING ↔ COASTING`, `HOLD`, `MANUAL`, `STOPPED`
- **mix 모드**: 매 프레임 OLTW를 갱신한다.
- **vocal 모드**
  - 라이브가 0.3초 넘게 비활성이면 **COASTING**으로 바뀐다. OLTW 갱신을 멈추고, 위치를 템포 ρ로 외삽한다. 다만 **다음 ref 보컬 활동 시작점(다음 줄 시작)을 넘지 않는다.**
  - 라이브 보컬이 돌아오면 예측 위치 주변 ±3초에 soft prior를 걸고 `anchor`해 다시 잡는다.
- **신뢰도**: 정규화 비용을 이동 기준선과 비교하는 0..1 값.
- **조작**: `jump(ref_time)`, `hold(bool)`, `nudge(dt)`, `set_transpose(k)`(ref chroma roll), `latency_comp`(출력 ref_time에 ρ·latency를 더한다)
- **출력**: `TrackerState(status, ref_time, tempo, confidence, live_time, frame_idx)`
- **전역 재탐색**(구현됨): 신뢰도가 2 s 넘게 낮으면 최근 6 s를 reference 전체와 coarse subsequence DTW로 비교한다. 후보는 shadow 트래커 replay로 검증한 뒤에만 옮긴다.
- **자동 키 검출**(통합 단계에서 추가, 기본값 켜짐)
  - `start()` 직후 나머지 11개 키마다 chroma를 roll한 shadow OLTW를 돌린다. 모두 같은 프레임을 받고 같은 방식으로 anchor된다.
  - 누적 평균 local distance가 다른 모든 후보보다 10% 넘게 낮은 상태가 3회 연속이면 그 키로 바꾼다.
  - 애매하면 바꾸지 않고 `key_suggest`로 제안만 한다.
  - 운영자가 키를 지정하면 꺼진다. 근거: [eval/TRACKING.md](eval/TRACKING.md) §7(평가 결과는 §2.1).

### Timeline (`timeline.py`)
- `Timeline(lines).locate(ref_time) -> Cursor(line_idx, syl_idx, syl_progress, next_line_idx, in_gap)`
- 프론트엔드 JS도 같은 규칙으로 구현한다(서버는 ref_time과 tempo만 보낸다).

## 6. 라이브 (D)

- **입력**: `DeviceInput(device, channels=[1-based], samplerate=None→장치 기본)`
  - sounddevice 콜백은 raw 블록을 큐에 넣기만 한다(처리 금지).
  - 워커 스레드가 선택 채널을 다운믹스하고, `soxr.ResampleStream`(quality 'MQ')으로 16k로 바꾼 뒤 extractor → tracker 순으로 넘긴다.
- **파일 입력**: `FileInput(path, realtime=True, speed=1.0, monitor=False)`. 실시간 속도로 블록을 내보내 시뮬레이션과 데모에 쓴다. `monitor`를 켜면 스피커로도 재생한다.
- **LiveSession**
  - 구성: 곡 패키지 로드 → 모드별 reference stem 선택(mix → `mix.wav`, vocal → `vocals.wav` 또는 `lead.wav`) → reference 특징(캐시) → LiveTracker
  - `snapshot()`은 dict를 돌려준다.
- **서버(FastAPI)**
  - REST: `/api/songs`, `/api/songs/{id}`, `/api/songs/{id}/audio/{stem}`, `POST /api/songs`(업로드 → prep 작업을 subprocess로 실행), `/api/jobs/{id}`, `/api/devices`, `/api/session/*`(start/stop/cue/hold/resume/jump/nudge/transpose/autokey/latency), `PUT /api/songs/{id}/lines/{idx}`(타이밍 수동 보정)
  - WS `/ws`: 20 Hz로 `{"type":"state",...}`를 push하고, 곡이 바뀌면 `{"type":"song","id":...}`를 보낸다.
- **UI**: 빌드 단계 없는 vanilla JS/CSS(`server/static/`)
  - `/`(콘솔): 곡 라이브러리와 업로드/prep 진행률, 세션 제어(모드, 입력 장치·채널, 시작/정지, 홀드, 줄 점프, nudge, 키, 지연, 리드 타임), 신뢰도·템포·입력 레벨, 줄별 신뢰도 검토(해당 위치부터 재생, 시작 시각 보정)
  - `/prompter`(가수용 전체화면): 현재 줄을 크게 표시하고 음절 단위 karaoke wipe를 한다. 다음 줄은 미리보기, 이전 줄은 흐리게 보인다. 화면 쪽에서 ref_time을 `ref_time + tempo·(now−t)`로 rAF 보간한다. 거울 모드, 글자 크기, 리드 타임을 설정할 수 있다.

## 7. 평가

- **정렬 정답 데이터**
  - Edelweiss.lrc: 영어, 단어 단위
  - 꽃밭에서: score.csv의 `offset`×0.5 → offline 경로 → ref 시간. 한국어, 음절 단위
  - Jamendo 4곡: 영어, `data/research-testsets/jamendo`
  - TTS: ko/ja/en, `data/research-testsets/tts-multilingual`
- **추적 평가**
  - (1) 합성: 구간별 time-stretch(0.8–1.25), 노이즈, 잔향, EQ를 넣어 정답 warp를 아는 가짜 라이브를 만든다.
  - (2) 실제 쌍: Edelweiss.wav ↔ Edelweiss_short.wav(두 번째 연주, 양쪽 GT 있음), 꽃밭에서 ref ↔ score
  - mix와 vocal 두 모드를 모두 평가한다(vocal은 양쪽 모두 분리 stem 사용).
- **지표**: AAE, MedAE, PCO@{200,300,500,1000}ms, 표시 기준 지연(latency), lost_frac. 참조 구현은 `docs/research/artifacts/reference-repo/scripts/gt_and_metrics.py`.
- **경로 탐색**: 레퍼런스 데이터는 `config.ref_repo_dir()`로 찾는다. 순서는 `LLT_REF_REPO` → `./live-lyrics-tracking` → `./_reference/live-lyrics-tracking`.
