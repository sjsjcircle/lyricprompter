# lyricprompter_oscsender — 설계

공연용 OSC 큐 전송기. 레퍼런스 음원과 큐 파일(레퍼런스 시간축의 시각 + OSC 메시지)을 받아, 라이브 공연을 lyricprompter의 트래커로 따라가면서 공연이 큐 시각에 닿을 때 OSC 메시지를 UDP로 보낸다. 사용법은 [README](../README.md), 함수 시그니처와 정확한 동작 계약은 [SPEC.md](SPEC.md)에 있다. 이 문서는 구조와 그 이유를 요약한다.

## 0. 원칙

- **기준 시간축**: 모든 시간은 초(float)이고, **reference 시간축(`audio/mix.wav` 기준)**을 쓴다. 큐 시각, 트래커 위치, 로그의 `ref_time`이 모두 같은 축이다. vocal stem은 mix와 같은 길이로 맞춰 저장한다(자르거나 무음을 붙임).
- **분석 격자는 lyricprompter와 같다**: 16 kHz mono, hop 640(40 ms, 25 fps). reference 특징과 라이브 특징은 같은 스트리밍 코드(`FeatureExtractor.push`)로 계산하고, reference 특징은 `projects/<id>/cache/`에 캐시한다.
- **실시간 경로는 CPU와 numpy만** 쓴다. torch가 없다. 오프라인 준비는 ffmpeg 변환(44.1 kHz stereo PCM24)과 튜닝 추정(librosa, 블록 단위 스트리밍)뿐이다.
- **트래커는 복사하고 고치지 않는다**: `features/`, `tracking/`, `live/audio_input.py`, `audio_io.py`는 lyricprompter의 파일과 같다. 유일한 차이는 `tracker.py`의 `STYLES` import 한 줄이다. `tests/test_vendored.py`가 그 한 줄을 되돌린 뒤 상위 checkout의 파일과 텍스트 전체를 비교한다(상위 checkout이 옆에 없으면 건너뜀). 추적 동작을 바꾸려면 상위에서 고치고 다시 복사한다.
- **큐 파일이 원본이다**: 사람이 편집기로 고칠 수 있는 텍스트이고, 콘솔 편집은 바뀐 토큰만 다시 쓴다. 서버는 파일을 감시해 바깥 편집도 받아들인다.
- **전송 경로에서 기다리지 않는다**: 큐를 보내는 스레드는 DNS 조회, 파일 쓰기, 콘솔 전송, 다른 잠금을 기다리지 않는다. 그런 일은 모두 다른 스레드나 잠금 밖에서 한다.

## 1. 패키지 구조 (`src/lyricprompter_oscsender/`)

| 모듈 | 책임 |
|---|---|
| `config.py` | 경로(`OSCSENDER_PROJECTS`), 분석 격자, `STYLES`, `MODES` |
| `cli.py` | 하위 명령 디스패치(각 모듈의 `cli(argv) -> int`) |
| `songfile.py` | stem 파일 이름 |
| `audio_io.py`, `features/`, `tracking/`, `live/audio_input.py` | lyricprompter에서 복사(변환, 특징, OLTW, LiveTracker, 입력) |
| `osc.py` | OSC 1.0 코덱, `Target`, `OscSender`(이름 붙은 대상으로 UDP 송신, 주소 패밀리별 소켓 하나) |
| `cues.py` | 큐 파일 파싱·형식·편집, `check` |
| `settings.py` | `.settings.json` 로드(손상 복구)·검증·원자적 저장, `@target` 병합과 충돌 경고 |
| `project.py` | 프로젝트 저장소, 튜닝 추정, 파형 peak 캐시, 휴지통, `add` / `list` |
| `synth.py` | 합성 라이브 take(lyricprompter `eval/track_eval.py` 복사), 테스트용 합성 음악 |
| `scheduler.py` | `CueScheduler`(순수 함수형 코어), `CueRunner`(타이머 스레드, 송신, 로그) |
| `live/session.py` | `OscSession`: 입력 → 특징 → 트래커 → `CueRunner`, 운영자 조작, show log |
| `live/simulate.py` | 가상 시계 오프라인 실행, `simulate` / `eval` |
| `latency.py` | echo 측정·응답기, 오디오 loopback 측정, `echo` / `latency` |
| `server/app.py` | FastAPI REST + WebSocket, Hub, 큐 편집·Undo, 파일 감시, 세션, `serve` |
| `server/static/` | 콘솔(vanilla ES modules, 빌드·CDN 없음) |

테스트는 `tests/test_<module>.py`에 있다. 실제 장치나 20초 넘는 실시간 실행이 필요한 것은 가짜 객체를 쓰거나 `@pytest.mark.slow`를 붙인다.

## 2. 큐 파일 모델 (`cues.py`)

```
CueSheet(text, cues, targets, issues)
Cue(id, time, address, args, target, message, note)    # id = 1부터 세는 줄 번호
TargetDecl(name, host, port, latency_ms, line)
CueIssue(line, message, severity)                      # "error" (줄 무시) | "warning"
```

- `cues`는 `(time, id)` 순서다. 시각이 같은 큐는 파일의 줄 순서로 나간다.
- `id`는 그 텍스트 한 버전에서만 유효하다. 줄을 넣거나 지우면 뒤의 id가 바뀌므로, 클라이언트는 편집할 때 `sheet_rev(text)`(sha1 앞 12자)를 함께 보내고 서버는 다르면 409와 현재 시트를 돌려준다.
- 다시 파싱해도 같은 큐를 알아보는 기준은 `Cue.key() = (round(time, 4), target, address, args)`다. 실행 중인 세션은 이 키로 보낸 상태를 넘겨받는다.
- **파싱은 예외를 내지 않는다.** 잘못된 줄은 `CueIssue`로 보고하고 건너뛴다. 서버는 여기에 알 수 없는 대상, 설정과 다른 `@target`, 레퍼런스 끝 뒤의 큐 경고를 더한다.
- 토크나이저는 각 토큰의 줄 안 위치(span)와 따옴표 여부를 기록한다(shlex는 이것을 알려 주지 않는다). 편집 함수(`set_cue_time`, `shift_cue`, `set_cue_message`, `add_cue`, `delete_cue`)는 텍스트 전체를 받아 돌려주며, 바뀐 글자만 다시 쓴다.
  - 시각은 원래 토큰의 형식(`TimeStyle`: 초 / 분:초 / 시:분:초, 0 채움, 괄호, 쉼표)으로, 밀리초 3자리로 쓴다. 뒤에 2칸 이상 공백이 있으면 길이 차이만큼 공백을 조절해 다음 칸 정렬을 지킨다.
  - BOM, 줄 끝(`\n`, `\r\n`, `\r`), 주석, 빈 줄은 그대로 둔다.
  - `add_cue`는 시각상 바로 앞 큐 줄 다음에 넣고, 그 줄의 들여쓰기·시각 형식·칸 맞춤을 따른다.
- `format_arg` / `format_args`는 `parse_message`로 다시 읽으면 같은 `(tag, value)`가 되는 정규 형식을 만든다.

## 3. 스케줄러 모델 (`scheduler.py`)

### 3.1 순수 코어 `CueScheduler`

트래커 갱신은 `TrackPoint(ref_time, rate, status, t, hold_from, settling)`로 들어온다. 모든 시간은 주입한 시계 하나(`now`) 위에 있다.

- **위치 외삽**: 마지막 점 `p` 이후 `P(now) = p.ref_time + v · clip(now − p.t, 0, max_extrap_s)`, `v = p.rate`(status가 `tracking` / `coasting`일 때, 아니면 0). `max_extrap_s = 0.1`. 큐를 40 ms 프레임 경계가 아니라 그 사이의 정확한 시각에 보내면서도, 멈춘 트래커가 위치를 0.1초 넘게 끌고 가지 않게 한다.
- **지연 앞당김**: 대상 지연 `L(c)`(초)인 큐는 `P(now) + v · L(c) ≥ c.time − 1e-9`일 때 나간다. reference 축의 앞당김이 `rate × latency`이므로 실제 시간으로 L만큼 일찍 나간다. `v = 0`이면 앞당기지 않는다.
- **due 시각의 역함수**: `t = p.t + (c.time − p.ref_time − v · L) / v`. 이 값이 `p.t + max_extrap_s`보다 늦으면 다음 갱신 전에는 나가지 않는다(+∞). `v = 0`이면 이미 닿았으면 −∞, 아니면 +∞. `due(now)`는 정확히 due 시각 ≤ now인 큐를 고른다.
- **레인**: 아직 처리하지 않은 큐를 지연 값(µs로 반올림)별 레인에 정렬된 인덱스로 둔다. 한 레인 안에서는 due 시각이 큐 시각과 함께 커지므로 `due`와 `next_due_time`은 각 레인의 머리만 본다: O(레인 수 + k).
- **점프 판정**: 갱신마다 이전 점을 새 갱신 시각으로 외삽한 `E`와 비교한다. `d = pt.ref_time − E`.
  - `d > catchup_s`(기본 1.0): 앞쪽 점프. `(E, new]`의 미처리 큐(그리고 vocal 가드가 붙잡고 있던 큐)를 건너뛴 큐 정책으로 넘긴다.
  - `d ≤ −rearm_back_s`(1.0): 뒤쪽 점프. 새 위치 뒤의 큐를 다시 무장한다(새 통과).
  - 그 밖: 특별한 일이 없다. 지나친 큐는 `due`로 나가며, 이상적인 due 시각이 갱신보다 한 프레임(40 ms) 넘게 이르면 `catchup`, 아니면 `track`이다.
- **건너뛴 큐 정책**: `skip`(모두 `skipped`), `latest`(`(target 또는 기본 대상, address)`마다 마지막 하나만 `chase`), `all`(모두 `chase`). 보내는 기록이 먼저, 각 묶음은 시각순이다.
- **복원 가능한 큐**: 트래커의 뒤쪽 점프는 잘못된 재탐색일 수 있다. 그 점프가 다시 무장한 큐(이번 통과에서 이미 처리한 큐)를 그 통과가 닿았던 위치(reach)와 함께 기억한다. 이후 `rearm_back_s` 이상의 앞쪽 이동이 reach에서 `rearm_back_s` 안쪽이나 그 너머에 내리면 그 큐들을 조용히(기록 없이) 처리한 상태로 되돌린다. 더 뒤에 내리거나 더 작은 이동이면 내린 위치까지의 큐만 되돌린다. 뒤쪽 점프 뒤에 같은 `(target, address)`로 다른 큐를 보냈다면 장비 상태가 바뀌었으므로 그 키의 큐는 복원하지 않는다. 운영자 점프와 `start`는 이 집합을 비운다.
- **settling과 coasting**: 음악을 따라가지 않는 점(status `coasting`, vocal 가드 `hold_from`, `settling`)이 이어지는 동안과 그 뒤 첫 정상 점은, 뒤쪽 이동을 그 구간이 시작된 위치(마지막으로 따라간 위치)에서 잰다. 목소리가 돌아와 relock이 coasting으로 지나친 위치를 되짚어도 아무것도 다시 나가지 않는다.
- **vocal 가드**: `hold_from`이 있으면 `time ≥ hold_from`인 큐는 나가지 않고 기다린다.
- **시작·점프 경계**: `start(pt, boundary)`와 `jump(pt, boundary)`의 `boundary`는 요청한 시각이다. 트래커는 그 시각에 입력 지연 앞당김을 더한 위치를 내므로, 경계 앞의 큐는 정책(`start` / `chase` / `skipped`)으로, 경계부터 공개 위치까지의 큐는 곧바로 `track`으로 나간다. 음수 입력 지연으로 공개 위치가 경계 뒤에 있으면 경계로 올린다. 그래서 0:00의 큐나 Go로 고른 큐는 입력 지연과 상관없이 바로 나가고, 다음 갱신이 거짓 점프로 보이지 않는다. 운영자 점프는 경계 이후의 큐를 모두 다시 무장한다.
- **큐 교체**(`set_cues`): 보낸 상태와 복원 상태를 `Cue.key()`로 넘긴다. 새 큐가 현재 위치보다 뒤(이미 지난 시각)면 보내지 않은 채 처리한 것으로 표시한다. 늦게 보내지 않는다.

### 3.2 실시간 실행기 `CueRunner`

- 잠금 하나(`threading.Condition(RLock)`)가 스케줄러를 지킨다. 잠금 안에서는 무엇을 보낼지만 정하고, 송신은 잠금 밖에서 한 번에 한 스레드가 순서대로 한다(대기열 `_pending`). 느린 `send`가 `feed`나 화면 조회를 막지 않는다.
- **타이머 스레드**는 다음 due 시각까지 Condition에서 기다린다. macOS의 timed wait는 길이의 1/4 정도까지 늦게 깨므로 남은 시간의 0.75씩 줄여 가며 기다리고(최대 50 ms), 1 ms 미만은 나누지 않는다. 송신은 이상적인 시각보다 약 0.1 ms 늦다.
- `feed(pt)`(오디오 스레드)는 먼저 이전 점이 갱신 시각까지 due로 만든 큐를 처리하고(타이머가 아직 깨지 않았을 수 있다), 갱신을 적용한 뒤, 지금 due인 큐를 처리한다. 점프 기록이 시간 순서를 어기지 않는다.
- `armed`가 꺼져 있으면 송신하지 않고 `dry_run` 기록만 남긴다. `fire_manual`은 `armed`와 상관없이 보내고(`manual`), 통과 상태는 바꾸지 않는다.
- 기록(`FireRecord`)은 크기 제한 deque(500)에 쌓고 `on_fire`로 넘긴다. `on_fire`는 모든 잠금 밖에서 seq 순서대로 한 스레드가 호출한다.

## 4. 세션 (`live/session.py`)

- **구성**(`build_tracking`): `project.json`의 튜닝·스타일 → 모드의 reference stem(mix → `mix.wav`, vocal → `vocals.wav`, 없으면 `StemMissingError`) → `FeatureConfig.for_mode` → `load_ref_features`(캐시) → `TrackerParams.for_mode` + 스타일 프리셋 → vocal 모드에서는 reference 보컬 활동 → `LiveTracker(line_starts=None)`. 가사가 없으므로 vocal 모드의 구절 시작(onset)은 reference 보컬 활동의 상승 지점이다. `simulate`도 같은 함수를 쓴다.
- **스레드와 잠금 순서**: 입력 소스의 워커 스레드가 `_on_audio`를 부른다. 순서는 항상 **오디오 스레드 → 세션 잠금(RLock) → runner 잠금**이고 그 반대는 없다.
  - 오디오 스레드는 세션 잠금을 쥔 채 `runner.feed`를 부른다. 그래서 운영자 점프를 오래된 트래커 점이 앞지르지 못한다.
  - runner가 부르는 콜백(`send`, `live_time`, `on_fire`)은 세션 잠금을 잡지 않는다. `send`는 현재 설정 객체만 읽고, `live_time`은 오디오 스레드가 교체하는 튜플 하나를 읽고, `on_fire`는 show log 대기열에 넣기만 한다.
  - 송신은 runner 잠금 밖에서 한다(3.2).
- **프레임 끝 시각 찍기**: `tracker.push` 직후의 `clock()`이 아니라, 블록 안에서 가장 새 프레임이 끝난 순간을 쓴다: `t = clock() − excess / sr`, `excess` = 그 프레임 끝 뒤에 남은 샘플 수. 블록 크기(리허설 파일은 20 ms)가 지연으로 더해지지 않는다.
- **vocal 가드**: vocal 모드이고, 위치가 자유 진행 중이 아니며(pop이 잡은 뒤가 아님), 트래커가 coasting 중이거나 라이브 입력이 비활성일 때 `hold_from = (onsets[k] + 1) / fps − 0.1`(k = 현재 위치 이후 첫 onset). 보컬이 들어오는 자리에 놓인 큐는 coasting 위치가 아니라 목소리가 실제로 돌아올 때 나간다. `settling`은 vocal 모드에서 relock이 잠정 상태(`tracker._prov`)이거나 OLTW가 아직 뒤로 물러날 수 있을 때(`oltw.free > 0`) 참이다.
- **운영자 조작**: `jump_time` / `jump_cue`는 `tracker.jump` 뒤에 가드 없는 점과 요청 시각(경계)으로 `runner.jump`를 부른다. Hold, nudge, 키, 자동 키, 스타일, 입력 지연은 트래커를 바꾼 뒤 새 상태를 보통 갱신으로 넘긴다(작은 앞쪽 이동은 늦게 보내고, 작은 뒤쪽 이동은 아무것도 다시 보내지 않는다).
- **시작**: show log를 만들고, extractor를 초기화하고, `tracker.start(start_time)`, `runner.start(point, boundary=start_time)` 뒤에 소스를 시작한다. 리허설 pre-roll(`align_source_start(2.0)`)은 파일 시작과 트래커 기준점을 함께 2초 앞으로 옮긴다.
- **show log**: `ShowLog`는 `SimpleQueue`와 쓰기 스레드 하나로 CSV에 한 줄씩 쓰고 바로 flush한다. 큐를 보내는 경로는 넣기만 한다. 쓰기 실패는 상태의 `warning`으로 보인다.
- **snapshot**: 20 Hz로 JSON-safe dict(NaN 없음)를 만든다. 위치, 템포, 신뢰도, 레벨, 키, 다음 큐 3개와 ETA, 최근 기록 30개, 개수, 대상별 송신 통계.

## 5. 서버 (`server/app.py`)

- **프로세스 하나, Hub 하나**: Hub가 설정, `OscSender` 하나(세션, Fire, Test, 미리듣기, loopback이 함께 씀), 세션 하나, 프로젝트별 Undo 기록과 broadcast rev, 감시하는 큐 파일, WebSocket 클라이언트를 가진다. asyncio 잠금은 `op_lock`(세션 시작·정지, 프로젝트 삭제), `edit_lock`(큐 파일과 `project.json` 쓰기), `settings_lock`, `loopback_lock`이다.
- **설정과 호스트 조회**: 송신기는 `HostBook.cached`로만 주소를 얻으므로, 쇼 도중 설정을 바꿔도 잠금 안에서 DNS를 기다리지 않는다(모르는 `.local` 이름은 5초 걸린다).
  - 조회는 모든 잠금 밖의 워커 스레드에서, host:port마다 하나씩 한다. 바뀌지 않은 호스트는 다시 조회하지 않는다.
  - 설정 PUT, 대상을 추가하는 큐 편집, 프로젝트 생성은 새로 생긴 호스트의 첫 조회를 기다린 뒤 응답한다.
  - 주소가 없는 활성 대상은 5초마다(그리고 Test와 수동 Fire 직전에), 이름으로 찾은 주소는 60초마다(DHCP) 다시 조회한다. 실패한 조회는 동작하던 주소를 지우지 않는다.
- **큐 편집**: 파일 읽기 → rev 확인(다르면 409 + 현재 시트) → `cues.py` 편집 함수 → 원자적 저장 → 감시 기준선 갱신 → Undo 기록(편집 전후 텍스트) → 프로젝트 rev 증가(broadcast) → 실행 중인 세션에 새 큐 → `@target` 병합. 응답은 새 시트 전체다.
  - Undo는 프로젝트마다 100개, 메모리에만 있다. 디스크의 파일이 마지막 기록의 "편집 후"와 다르면(바깥 편집) 거절하고 기록을 지운다. 기록 사슬이 끊긴 편집(이전 기록의 "후"와 이번 "전"이 다름)도 그 전 기록을 지운다.
- **파일 감시**: 별도 태스크가 0.5초마다 라이브 프로젝트와 콘솔이 마지막으로 연 프로젝트의 `cues.txt`를 stat한다. 변경은 파일 서명(mtime, size, inode, ctime)으로 알고, 5초 안에 쓴 파일은 텍스트도 비교한다(FAT32·HFS+의 mtime 해상도는 1–2초라 같은 크기로 저장하면 서명이 같을 수 있다). 바뀌면 rev를 올리고 세션에 넘기고 `@target`을 병합한다. 수동 Fire와 큐 id로 하는 Go는 감시기가 아직 보지 못한 디스크 편집을 먼저 세션에 넘긴다.
- **세션 시작은 make before break**: 큐 파일을 읽고(`start_cue`는 rev 확인), 팩토리로 `DeviceInput` / `FileInput` + `OscSession(log_dir)`을 만들고, pre-roll·입력 지연·키를 적용해 시작한 뒤에야 이전 세션을 멈춘다. 새 장치 입력이 열리지 않고 이전 장치 세션이 인터페이스를 쥐고 있으면, 이전 세션을 멈추고 새로 만들어 한 번 더 시도한다. 만드는 동안 저장된 설정, 큐 편집, 스타일은 시작 후 적용한다.
- **WebSocket**: 연결하면 `hello`, 이후 20 Hz `state`(세션 snapshot 또는 idle), 프로젝트가 바뀌면 `project {id, rev, sheet_rev}`, 설정이 바뀌면 `settings {rev, armed}`. 클라이언트마다 종류별 최신 메시지 하나만 남기는 outbox를 두어 느린 콘솔이 broadcaster를 막지 않고, 2초 동안 받지 않는 연결은 끊는다.
- **OriginGuard**(lyricprompter에서 복사): Host 헤더가 loopback, IP, 이 컴퓨터 이름, `--allow-host`가 아니면 421. 상태를 바꾸는 요청과 WebSocket은 Origin(없으면 Referer)이 같은 출처가 아니면 403. 인증은 없다.

## 6. 콘솔 (`server/static/`)

- **미리듣기 엔진**: 페이지마다 AudioContext 하나. 프로젝트의 mix를 한 번 디코드하고 `AudioBufferSourceNode`를 `startCtx`에 `offset`부터 재생한다.
  - 문맥 위치 = `offset + (currentTime − startCtx)`, 들리는 위치 = 그 값 − `outLat`(`outputLatency || baseLatency || 0`). 화면의 재생 위치, Set, Tap, Add cue at playhead, 일시정지는 들리는 위치를 쓴다.
  - **클릭**: 25 ms 주기 look-ahead 루프가 0.12초 앞까지 같은 문맥 시간축에 oscillator(1.5 kHz × 대상별 음정, 30 ms 엔벨로프)를 예약한다. 음악과 샘플 단위로 맞는다. 늦은 클릭(백그라운드 탭)은 버린다.
  - **미리듣기 OSC**: 큐마다 `cueCtx + outLat − 대상 latency`에 `setTimeout`으로 `POST .../fire {"source": "preview"}`를 보낸다. 큐 키(시각·대상·메시지)로 통과마다 한 번만 보낸다. 정지 후 재생과 위치 이동은 새 통과, 일시정지 후 재개는 이어서. 큐 편집이나 토글은 예약을 다시 하고(이미 due였던 것은 바로 보냄), 라이브 세션이 시작되면 끈다. 서버는 미리듣기 송신을 세션 로그에 남기지 않는다.
- **편집 대기열**: 모든 큐 편집과 Undo는 promise 사슬 하나로 순서대로 보낸다. 각 편집은 그때 시트의 rev를 싣고, 줄 배치가 바뀐 뒤(추가·삭제·Undo·다시 읽기) 실행되는 id 편집은 버리고 알린다. 응답의 시트로 다시 그리고, 409면 응답에 실린 현재 시트를 적용한다. Tap으로 대기 중인 큐와 Start at의 큐는 (시각, 대상, 메시지)나 같은 줄로 새 id를 따라간다.
- **라이브 화면**: `RefClock`이 state 메시지의 위치를 템포로 최대 0.35초 외삽해 그린다. Fired 로그는 매 state에 실린 최근 30개를 seq로 중복 제거하고, seq가 줄거나 프로젝트가 바뀌면 새 실행으로 본다.

## 7. 지연 측정 (`latency.py`)

- **Network echo**(`measure_echo`): UDP 소켓 하나로 순차 ping. 기본은 `/oscsender/ping i:seq d:send_time`을 보내고 첫 인자가 그 seq인 `/oscsender/pong`만 받는다(늦은 답장, 다른 주소, OSC가 아닌 패킷은 무시). 보내기 전마다 소켓에 쌓인 것을 비운다. 다음 ping은 간격(기본 50 ms) 뒤에, 그러나 답장이나 timeout(기본 500 ms) 전에는 보내지 않는다. `message`를 주면 그것을 보내고 처음 돌아온 패킷을 답장으로 본다. 추정값 = 왕복 중앙값 / 2. `serve_echo`는 ping에 pong(같은 인자)으로 답하고, `echo_all`이면 모든 메시지를 그대로 돌려준다.
- **Audio loopback**(`measure_loopback`): `AudioCapture`가 콜백마다 블록 첫 샘플의 ADC 시각을 찍는다: `clock − (currentTime − inputBufferAdcTime)`(PortAudio가 주고 1초보다 오래되지 않을 때), 아니면 `clock − stream.latency − frames / sr`. 15초 링 버퍼.
  - 검출: 1 ms RMS 엔벨로프(dB). 바닥 = 보내기 전 0.3초의 중앙값. onset = 보낸 뒤 1.0초 안에서 `max(바닥 + threshold, −70 dBFS)`를 처음 넘는 프레임, 그 프레임(또는 앞 프레임)에서 임계값에 닿는 첫 샘플로 다듬는다.
  - 다음 시도는 소리가 바닥 근처(6 dB 안)로 돌아오거나 간격(1.5초)이 지나면 시작한다. 놓친 시도 뒤에는 간격을 다 기다린다(늦게 온 소리가 다음 시도에 섞이지 않게).
  - 추정값 = 보낸 순간부터 소리까지의 중앙값.
- 측정 쪽 시간은 모두 `OscSender`가 송신 시각을 찍는 시계(`time.perf_counter`)와 같은 시계다. `LatencyReport`의 통계(min, median, p95, max, IQR)는 성공한 시도만으로 낸다.

## 8. 평가 방법 (`live/simulate.py`, `synth.py`)

- **오프라인 실행**(`simulate_cues`): 세션과 같은 `build_tracking`으로 트래커를 만들고, 라이브 오디오를 320샘플(20 ms) 블록으로 넣는다. **가상 시계 = 라이브 시간**이다. 프레임을 완성한 블록은 블록 끝 시각의 갱신이 되고(점은 `min(live_time, t_k)`로 찍음), 다음 블록 끝까지 `next_due_time()`이 닿는 큐를 그 정확한 시각에 `due(max(nd, t_k))`로 보낸다. 스레드 지터와 오디오 입력 지연이 없다(입력 지연 보정 기본 0).
- **합성 take**(`eval`): `synth.make_synthetic`으로 레퍼런스를 변형한다. 구간(2–8초)마다 log-uniform 템포(기본 0.8–1.25), phase vocoder 워프, 1초 잡음 리드인, 핑크 잡음(SNR 20 dB), 잔향(RT60 0.6 s), EQ 기울기(±6 dB), 게인. 라이브 ↔ reference 시간 대응을 정확히 안다(`ref_to_live`).
- **지표**: 큐마다 첫 송신의 라이브 시각 − 정답 라이브 시각. 모든 대상 지연 0, 입력 지연 0. 시작 위치(0초)의 큐와 레퍼런스 끝 뒤의 큐는 채점하지 않는다. take별·전체의 놓친 수, 다시 나간 수, |오차| 중앙값·p90·최댓값, 부호 있는 중앙값(bias), 50 / 100 / 200 ms 이내 비율.
- **테스트용 합성 음악**: `make_score` / `render_song`은 화음, 선율, 타악 잡음, 쉼표가 있는 결정적 합성 곡을 만든다. `tempo_map`으로 다시 렌더한 "라이브"는 phase vocoder 없이 정답을 정확히 안다. `tests/test_simulate.py`가 이것으로 mix 모드 두 스타일 모두 모든 큐 정확히 한 번, |오차| 중앙값 ≤ 60 ms, p90 ≤ 150 ms를 확인하고, `tests/test_integration.py`는 실제 서버·세션·UDP 수신기로 합성 곡 리허설을 끝까지 돌린다(모든 큐가 순서대로 한 번, pre-roll과 시작 시 정책).
- 측정 결과는 [README의 정확도](../README.md#6-정확도-합성-평가)에 있다.
