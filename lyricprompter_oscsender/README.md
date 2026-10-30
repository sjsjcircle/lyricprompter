# lyricprompter_oscsender — 공연용 OSC 큐 전송기

**레퍼런스 음원과 큐 파일만 넣으면** 공연 중 라이브 밴드·가수를 실시간으로 따라가다가, 공연이 각 큐의 시각에 닿을 때 그 OSC 메시지를 보냅니다. 큐 파일은 레퍼런스 음원의 시간축에 적은 시각과 OSC 메시지 목록입니다. 조명 콘솔, QLab, 영상 서버, 샘플러처럼 OSC를 받는 장비를 연주에 맞춰 움직일 때 씁니다.

추적은 [lyricprompter](../README.md)의 트래커를 **바꾸지 않고 복사해** 씁니다: OLTW 정렬, classic / pop 스타일, mix / vocal 모드, 자동 키 검출, 전역 재탐색. 복사한 파일이 원본과 같은지는 `tests/test_vendored.py`가 확인합니다. 가사 분석과 프롬프터 화면은 들어 있지 않습니다.

```
레퍼런스 음원 + 큐 파일 ──▶ [준비] ffmpeg 변환 · 튜닝 추정 · 큐 파싱 ──▶ projects/<id>/
라이브 오디오 ──▶ [추적] reference와 실시간 정렬(OLTW) + 자동 키 검출 ──▶ reference 위치
reference 위치 ──▶ [전송] 큐 시각에 닿으면 (대상의 지연만큼 미리) ──▶ OSC 메시지 (UDP)
```

| 라이브 입력 (운영자가 선택) | 비교 대상 reference |
|---|---|
| **Full band mix**: 반주가 포함된 오디오(믹서 출력 등) | 레퍼런스 음원 `audio/mix.wav` |
| **Vocal mic only**: 보컬 단독(보컬 마이크 채널) | 같은 녹음에서 분리한 보컬 `audio/vocals.wav` (직접 넣어야 함) |

## 설치 (macOS, Apple Silicon)

이 폴더는 상위 lyricprompter와 **따로 설치하는 uv 프로젝트**입니다. 보컬 분리와 가사 정렬 모델이 없어 torch를 설치하지 않으므로 환경이 가볍고, 모델을 내려받지도 않습니다. `uv.lock`도 따로 있고, Apple Silicon Mac(`arm64`)의 Python 3.12를 기준으로 합니다.

### 1) 공통 도구

Homebrew, `ffmpeg`, `uv`가 필요합니다. 아직 없다면 상위 README의 [설치 1)–2) 단계](../README.md#설치-macos-apple-silicon)를 따라 설치하세요. `ffmpeg`는 레퍼런스 음원과 보컬 stem을 변환할 때, `uv`는 Python과 패키지를 관리할 때 씁니다.

### 2) 패키지 설치

저장소 폴더(`lyricprompter`)에서 이 프로젝트 폴더로 이동합니다.

```bash
cd lyricprompter_oscsender
```

Python 3.12가 없다면 먼저 설치하고, 저장소에 고정된 버전의 패키지를 설치합니다.

```bash
uv python install 3.12
```

```bash
uv sync --locked
```

`uv sync --locked`는 이 폴더 안에 별도의 `.venv`를 만듭니다. 상위 프로젝트의 `.venv`와는 섞이지 않습니다. 이후 명령은 모두 이 `lyricprompter_oscsender` 폴더에서 실행하세요.

### 3) 설치 확인

```bash
uv run oscsender --help
```

`serve`, `add`, `list`, `check`, `simulate`, `eval`, `echo`, `latency`, `devices` 명령 목록이 나오면 됩니다.

### 마이크와 오디오 인터페이스

실시간 입력을 사용할 때는 **시스템 설정 › 개인정보 보호 및 보안 › 마이크**에서 서버를 실행한 앱(터미널, iTerm 등)의 권한을 허용하세요. 오디오 인터페이스를 연결한 뒤 다음 명령으로 입력 장치를 확인할 수 있습니다.

```bash
uv run oscsender devices
```

이 명령의 도움말과 안내문에는 `lyricprompter`라는 이름이 나옵니다. 상위 프로젝트의 모듈을 그대로 복사했기 때문이며, 장치 목록은 같습니다. 콘솔에서는 **Input › Audio interface**의 장치와 입력 채널을 고릅니다. 장치를 새로 연결했다면 세션을 멈춘 뒤 **Rescan devices**를 누르세요.

### 설치 중 막혔을 때

| 증상 | 확인할 내용 |
|---|---|
| `error: Failed to spawn: oscsender` | 상위 `lyricprompter` 폴더에서 실행한 경우입니다. `cd lyricprompter_oscsender`로 이 폴더에 들어온 뒤 `uv sync --locked`와 명령을 실행하세요. |
| `ffmpeg not found on PATH` | `brew install ffmpeg`를 완료한 뒤 새 터미널에서 `ffmpeg -version`을 확인하세요. |
| `Address already in use` | 기존 서버를 Ctrl+C로 종료하거나 `uv run oscsender serve --port 8801`로 시작하세요. 브라우저 주소의 포트도 `8801`로 바꿉니다. 상위 lyricprompter(기본 8000)와는 포트가 달라 함께 실행할 수 있습니다. |
| 장치가 없거나 입력 신호가 없음 | 위의 마이크 권한, 장치 연결, 입력 채널을 확인한 뒤 세션을 멈추고 **Rescan devices**를 누르세요. |

## 빠른 시작

```bash
uv run oscsender serve
```

터미널에 주소가 나오면 브라우저에서 [콘솔](http://127.0.0.1:8800/)을 엽니다. 사용하는 동안 터미널을 열어 두고, 종료할 때는 **Ctrl+C**를 누릅니다.

1. **프로젝트 만들기**: 콘솔 왼쪽 **New project**에 다음을 넣고 **Create project**를 누릅니다. 음원 변환과 튜닝 추정이 끝나면 목록에 나타납니다.
   - `Title`
   - `Reference audio`: 레퍼런스 녹음. ffmpeg가 읽는 형식이면 됩니다(WAV, MP3, M4A, FLAC 등).
   - `Cue file`(UTF-8 텍스트), 또는 `… or paste the cues` 칸에 붙여넣은 큐. 비워 두고 나중에 콘솔에서 큐를 추가해도 됩니다.
   - `Vocal stem (optional, for vocal mode)`: 보컬 마이크만 따라갈 때 필요합니다([Listen to](#listen-to)).
   - `Tracking style`: [classic / pop](#style).

   명령줄로 만들 수도 있습니다.

   ```bash
   uv run oscsender add --audio song.wav --cues cues.txt --title "꽃밭에서"
   ```

2. **큐 확인**: 프로젝트를 고르면 오른쪽에 파형과 큐 목록이 나옵니다. **Play**(Space)로 레퍼런스를 들으며 큐마다 나는 클릭 소리로 시각을 확인하고 고칩니다([큐 타이밍 확인](#2-큐-타이밍-확인)).
3. **OSC 대상 설정**: 가운데 **OSC targets**에 받는 장비의 Host와 Port를 넣고 **Test**로 메시지가 도착하는지 확인합니다. 큐 파일의 `@target` 줄은 여기에 자동으로 추가됩니다. 지연 보정은 [OSC 지연 측정과 보정](#5-osc-지연-측정과-보정)을 참고하세요.
4. **리허설**: **Input**을 `Rehearsal`로 두고 `This project's mix`(또는 리허설 녹음 파일)를 고른 뒤 **Start**를 누릅니다. 녹음을 실시간 속도로 재생하면서 공연처럼 큐를 보냅니다. 오른쪽 위 **OSC LIVE** 버튼을 눌러 `SAFE — dry run`으로 두면 큐가 로그에만 남고 전송되지 않습니다.
5. **공연**: **Input**을 `Audio interface`로 바꾸고 장치와 채널을 고른 뒤 곡이 시작되기 직전에 **Start**를 누릅니다.

## 1. 큐 파일 형식

UTF-8 텍스트 파일 하나가 프로젝트의 큐 목록입니다(`projects/<id>/cues.txt`). 한 줄에 큐 하나를 씁니다.

```
# 공연: 예시 곡                          (#으로 시작하는 줄은 주석, 빈 줄은 무시)
@target lights 192.168.0.50:8000
@target video  127.0.0.1:7000  latency=80ms
@target qlab   [::1]:53000     latency=12

0:00.000                 /show/start
0:12.480     lights      /cue/go 12                # verse 1 wash
1:05.2       video       /clip/play "intro.mp4" 0.5
75.300                   /mixer/fader f:0.75 i:3 true
[01:20.00]               /scene "two words" nil
1:32,5       qlab        /cue/7/start
0:01:40.250  lights      /preset s:123 d:0.1 h:5 'it\'s'
1:45.000     lights      /cue/go 13 # chorus
```

큐 한 줄은 `<시각> [<대상>] <주소> [<인자> ...] [# 메모]`입니다. 토큰은 공백이나 탭으로 구분하고, 칸을 맞추는 공백 수는 자유입니다.

### 시각

| 쓰는 법 | 예 | 의미 |
|---|---|---|
| `S[.fff]` | `75.3`, `75.300` | 초 |
| `M:SS[.fff]` | `1:05.2`, `01:05.200`, `75:02.0` | 분:초 (분은 60 이상도 됩니다) |
| `H:MM:SS[.fff]` | `1:01:05.2`, `0:01:40.250` | 시:분:초 |
| `[ ]` | `[01:20.00]` | LRC 타임태그처럼 괄호로 감싸도 됩니다. |
| `,` | `1:32,5` | 소수점 대신 쉼표를 써도 됩니다. |

- 첫 칸 뒤의 분·초는 60 미만이어야 합니다(`1:60`은 오류).
- 음수, 100:00:00.000보다 늦은 시각, `HH:MM:SS:FF` 같은 프레임 타임코드는 오류입니다(`frames are not supported: write HH:MM:SS.fff`).
- 줄 순서는 상관없습니다. 큐는 시각순으로 정렬하고, 시각이 같으면 파일의 줄 순서대로 보냅니다.
- 콘솔에서 시각을 고치면 그 줄이 쓰던 형식(초 / 분:초 / 시:분:초, 0 채움, 괄호, 쉼표)을 유지하고 밀리초 3자리로 씁니다.

### 대상 (생략 가능)

주소 앞의 토큰이 `/`로 시작하지 않으면 대상 이름입니다. 영문자나 `_`로 시작하고 영문자·숫자·`_ . -`만 씁니다. 생략하면 콘솔의 **Default target**으로 보냅니다. OSC targets에도 없고 `@target`으로 선언하지도 않은 이름은 경고가 뜨고, 그 큐는 보낼 수 없습니다.

### 주소

`/`로 시작하는 첫 토큰입니다. 공백, `#`, `,`(`{a,b}` 안은 허용), 제어 문자, ASCII가 아닌 글자는 쓸 수 없습니다. 패턴 문자 `* ? [ ] { }`는 그대로 보냅니다.

### 인자

토큰 하나가 OSC 인자 하나입니다.

| 토큰 | OSC 타입 | 예 |
|---|---|---|
| `"..."` 또는 `'...'` | `s` 문자열. 안에서 `\`는 다음 글자를 그대로 씁니다. `""`는 빈 문자열입니다. | `"two words"`, `'it\'s'` |
| `true` / `false` | `T` / `F` (대소문자 무관) | `true`, `False` |
| `nil` / `null` | `N` (대소문자 무관) | `nil` |
| 정수 | `i`(int32). int32를 넘으면 `h`(int64), int64도 넘으면 오류 | `12`, `-3`, `3000000000` |
| 실수 | `f`(float32). float32 범위를 넘으면 오류(`write d:...`) | `0.5`, `-.5`, `1e3`, `2.` |
| 타입 접두어 `i:` `h:` `f:` `d:` `s:` | 지정한 타입. 값이 타입에 맞지 않으면 오류입니다. `s:`는 토큰의 나머지(또는 따옴표 문자열) | `f:1`, `d:0.1`, `h:5`, `s:123`, `s:"a b"` |
| 그 밖의 단어 | `s` 문자열 | `intro`, `GO` |

blob(`b`)과 impulse(`I`)는 큐 파일에 쓸 수 없습니다. 한 큐는 OSC 메시지 하나(번들 아님)로 바로 보냅니다.

### 메모 (`#`)

- 앞에 공백이 있는 `#`부터 줄 끝까지가 메모입니다. 콘솔의 Note 칸에 보이고, 전송하지 않습니다.
- 따옴표 안의 `#`은 글자 그대로입니다.
- **공백 없이 붙은 `#`은 메모가 아닙니다.** `/cue/go 12# verse 2`는 `"12#"`, `"verse"`, `2` 세 인자를 보내고 경고(`write a space before # to start a note, or quote the string`)가 뜹니다. 따옴표나 `s:`로 쓴 인자에는 경고가 없습니다.

### `@target` 선언

`@target <이름> <호스트>:<포트> [latency=<n>ms]`로 받는 장비를 선언합니다.

- IPv6 주소는 `[::1]:9000`처럼 괄호로 감쌉니다.
- `latency`는 ms 단위이고 `ms`는 생략해도 됩니다. 0–5000 범위입니다([대상 지연](#5-osc-지연-측정과-보정)).
- 같은 이름을 두 번 선언하면 첫 줄을 씁니다(경고). `@target` 외의 `@...` 줄은 경고 후 무시합니다.

### 오류와 경고

- **오류가 있는 줄은 건너뛰고 나머지는 계속 읽습니다.** 콘솔의 큐 목록 위에 `줄 번호: 메시지` 목록이 나오고(클릭하면 그 줄로 이동), 프로젝트 목록에 오류 수가 표시됩니다.
- **경고**(큐는 그대로 씀): 다른 줄과 시각·대상·주소·인자가 모두 같은 큐, 두 번 선언한 대상, `#`이 붙은 인자, 대상을 알 수 없는 큐, 레퍼런스 끝보다 늦은 큐, OSC targets와 다른 `@target` 선언, 일부 편집기가 줄바꿈으로 보이는 문자(`\v`, `\f`, U+2028 등: 공백으로 읽음).
- 줄 끝은 `\n`, `\r\n`, `\r`(옛 Mac, Excel 내보내기)을 모두 읽고, 고칠 때 원래 형식을 유지합니다. 맨 앞의 BOM은 무시합니다.

서버 없이 파일만 검사할 수도 있습니다. 큐를 시각순으로 출력하고, 오류가 있으면 종료 상태 1을 돌려줍니다(`--json`: JSON 출력).

```bash
uv run oscsender check cues.txt
```

### `@target`과 콘솔의 OSC targets

OSC targets(`projects/.settings.json`)는 모든 프로젝트가 함께 쓰는 설정입니다. 큐 파일의 `@target` 선언은 여기에 **아직 없는 이름만** 추가됩니다.

- 추가되는 때: 프로젝트를 만들 때(콘솔, `oscsender add`), 콘솔에서 프로젝트를 열 때, 세션을 시작할 때, 큐 파일을 고칠 때(콘솔 편집, Edit file, Replace file, Undo, 텍스트 편집기에서 저장).
- **이미 있는 대상은 바꾸지 않습니다. 설정이 이깁니다.** host, port, 선언한 latency가 설정과 다르면 그 `@target` 줄에 경고(`... the OSC targets setting wins ...`)가 뜹니다. 선언대로 보내려면 OSC targets에서 고치세요.
- 콘솔에서 지운 대상이라도 큐 파일에 `@target`이 남아 있으면 그 프로젝트를 다시 열거나 고칠 때 다시 추가됩니다.

## 2. 큐 타이밍 확인

lyricprompter의 타이밍 검토 화면을 본뜬 수동 편집기입니다. 자동 정렬은 하지 않습니다. 레퍼런스를 들으며 손으로 맞춥니다.

### 파형과 확대 띠

- **파형**(곡 전체): 큐 마커가 대상별 색으로 보이고, 미리듣기 재생 위치(파랑)와 라이브 세션 위치(주황)가 표시됩니다. 클릭하면 그 위치로 이동합니다(마커 근처를 클릭하면 그 큐의 시각으로). 마우스를 올리면 큐 내용이 나옵니다.
- **확대 띠**(재생 위치 ±4초): 마커에 라벨이 붙습니다. **마커를 좌우로 끌면 그 큐의 시각이 옮겨집니다**(놓을 때 저장, 1 ms 단위, Esc로 취소). 끌지 않고 클릭하면 그 위치로 이동합니다.
- **Follow live**를 켜 두면 이 프로젝트가 라이브 중일 때 큐 목록이 세션이 마지막으로 보낸 큐로 스크롤하고(Tap 모드와 칸 편집 중에는 제외), 미리듣기가 멈춰 있으면 확대 띠가 라이브 위치를 따라갑니다.

### 미리듣기

- **Play / Pause**(Space): 레퍼런스(mix)를 브라우저에서 재생합니다. 큐 줄의 **Play**는 그 큐 1초 앞부터 재생합니다.
- **Click at cues**: 큐마다 짧은 클릭(1.5 kHz, 대상별로 음높이가 다름)을 냅니다. 클릭은 음악과 같은 AudioContext 시간축에 샘플 단위로 맞춰 예약하므로, 클릭이 음악과 어긋나 들리면 큐 시각이 어긋난 것입니다. 옆 슬라이더는 클릭 볼륨입니다.
- 화면의 재생 위치, **Set**, **Tap**, **Add cue at playhead**, 일시정지 위치는 브라우저가 알려 주는 출력 지연(`outputLatency`)을 뺀 **들리는 위치**를 씁니다.
- **Send OSC while previewing**: 미리듣기 중 큐마다 실제 OSC 메시지를 보냅니다.
  - 대상의 latency만큼 미리 보내므로, 효과가 음악에 맞게 나오는지 눈과 귀로 확인할 수 있습니다([보정](#눈과-귀로-보정)).
  - SAFE일 때도 보냅니다. 라이브 세션이 돌고 있으면 확인 창을 거쳐야 켜지고, 세션이 시작되면 자동으로 꺼집니다.
  - 재생 한 번에 큐마다 한 번만 보냅니다. 정지 상태에서 재생하거나 위치를 옮기면 처음부터, 일시정지 후 재개하면 이어서 셉니다.

### 큐 목록

열은 `Go | # | Time | Target | Message | Note`와 버튼들입니다. `#`은 `cues.txt`의 줄 번호입니다. 미리듣기 위치보다 앞의 큐는 흐리게, 라이브 세션이 마지막으로 보낸 큐와 다음 큐는 강조해 보여줍니다. 오류나 경고가 있는 줄은 표시되고, 마우스를 올리면 내용이 나옵니다.

| 조작 | 동작 |
|---|---|
| **Play** | 그 큐 1초 앞부터 미리듣기 |
| **−.1 / +.1 / −.01 / +.01** | 큐를 0.1초 / 10 ms씩 옮깁니다. |
| **Set** | 큐 시각을 미리듣기의 들리는 위치로 맞춥니다. |
| Time 칸 클릭 | 시각을 직접 입력합니다(큐 파일의 시각 문법: `75.3`, `1:05.200` …). Enter 저장, Esc 취소 |
| Message 칸 더블클릭 | `[대상] /주소 인자 [# 메모]`를 직접 입력합니다. Enter 저장, Esc 취소 |
| **Fire** | 그 큐를 지금 보냅니다(SAFE여도). 이 프로젝트가 라이브 중이면 Fired 로그와 show log에 `manual`로 남습니다. |
| **✕** | 그 큐 줄을 지웁니다(확인 후, Undo로 복구). |
| **Go** | 이 프로젝트가 라이브 중이면 그 큐로 점프합니다(큐가 바로 나감). 세션이 없으면 **Start at**을 그 큐로 정합니다. |

### 도구 모음

- **Undo**(⌘Z / Ctrl+Z): 프로젝트마다 최근 100번까지 되돌립니다. 기록은 서버 메모리에만 있어 서버를 다시 시작하면 사라집니다. 텍스트 편집기로 파일을 바꾼 뒤에는 그 전의 편집을 되돌릴 수 없습니다(기록을 지움).
- **Tap**(여러 큐를 빠르게 맞출 때): 켜면 미리듣기 위치 다음의 첫 큐가 파랗게 대기합니다.
  - 재생하면서 큐 순간마다 **T**를 누르면 그 큐 시각이 들리는 위치(1 ms 반올림)로 정해지고 다음 큐가 대기합니다.
  - `#` 칸을 클릭하면 그 큐부터 다시 탭합니다. **Esc**로 끝냅니다.
  - 사람의 반응 시간은 보정하지 않으므로, 필요하면 −.01 / −.1로 다듬으세요.
- **Add cue at playhead**: Time(재생 위치가 미리 채워짐), Target(`Default` 또는 대상 이름), Message를 넣고 **Add**를 누릅니다. 새 줄은 시각상 바로 앞 큐 줄 다음에, 그 줄의 시각 형식과 칸 맞춤으로 들어갑니다.
- **Edit file**: 파일 전체를 편집 창에서 고칩니다. **Save**는 오류가 있어도 저장하고(오류 줄은 무시) 문제 목록을 보여줍니다. 그 사이 다른 곳에서 파일이 바뀌었으면 저장하지 않고 알립니다(다시 Save하면 덮어씀).
- **Download cues.txt**, **Replace file**(업로드한 파일로 교체, Undo 가능).

### 파일은 고친 부분만 바뀝니다

- `cues.txt`가 원본입니다. 콘솔 편집은 그 줄의 고친 부분만 다시 씁니다. 시각을 고치면 시각 토큰만, 메시지를 고치면 시각 뒤 부분만 바뀌고 주석, 빈 줄, 칸 맞춤, 줄 끝 형식은 그대로입니다.
- **텍스트 편집기로 직접 고쳐도 됩니다.** 서버는 라이브 중인 프로젝트와 콘솔에서 연 프로젝트의 파일을 0.5초마다 확인합니다. 바뀌면 콘솔이 다시 읽고(`The cue file changed elsewhere and was reloaded.`), 실행 중인 세션에도 바로 반영합니다.
- 여러 콘솔에서 고쳐도 엇갈리지 않습니다. 편집은 파일 버전과 함께 보내고, 그 사이 파일이 바뀌었으면 적용하지 않고 목록을 다시 읽어 알립니다.

### 단축키

| 키 | 동작 |
|---|---|
| Space | 미리듣기 재생 / 일시정지 |
| T | Tap 모드에서 대기 중인 큐의 시각 지정 |
| ⌘Z / Ctrl+Z | 큐 편집 되돌리기 |
| H | Hold / Resume |
| `[` / `]` | −0.5 s / +0.5 s 미세 조정 |
| Esc | 마커 끌기 취소, Add cue 닫기, Tap 모드 끝내기 |

입력 칸에 글을 쓰는 중이거나 대화창이 열려 있으면 단축키는 동작하지 않습니다.

## 3. 공연 (실시간)

가운데 Session 칸에서 고르고 **Start**를 누릅니다. 세션은 한 번에 하나이며, 세션이 도는 중에 Start를 누르면 확인 후 새 세션으로 바꿉니다. 입력 칸에서 Enter를 눌러도 시작하지 않습니다.

### Listen to

- **Full band mix**: 믹서 출력처럼 반주가 들어간 오디오를 레퍼런스 `mix.wav`와 비교합니다.
- **Vocal mic only**: 보컬 마이크만 받아 같은 녹음의 분리 보컬 `vocals.wav`와 비교합니다. 프로젝트에 보컬 stem이 없으면 고를 수 없습니다.
  - **보컬 stem 구하기**: 상위 lyricprompter에서 같은 음원으로 곡을 준비하면(콘솔의 **Add song** 또는 `uv run lyricprompter prep`, 가사 필요) 분리한 보컬이 `songs/<id>/audio/vocals.wav`에 생깁니다(`--lead`로 준비했다면 리드 보컬 `lead.wav`도 있음). 다른 분리 도구의 결과도 같은 녹음에서 분리했고 같은 순간에 시작하면 됩니다.
  - New project의 `Vocal stem` 칸이나 `oscsender add --vocals`로 넣습니다. 레퍼런스와 길이가 다르면 잘라 내거나 무음을 붙여 맞추고, 0.5초 넘게 다르면 경고합니다.
  - 이미 만든 프로젝트에는 **Listen to**의 **Add vocal stem…**(있으면 **Replace vocal stem…**) 버튼으로 넣거나 교체합니다. 변환이 끝나면 **Vocal mic only**를 고를 수 있습니다.

### Style

프로젝트마다 고릅니다. New project의 `Tracking style`, Session 칸의 `Style`, `oscsender add --style`에서 정합니다. Session 칸에서 바꾸면 프로젝트에 저장되고, 그 프로젝트가 라이브 중이면 바로 적용됩니다.

| 스타일 | 맞는 곡 | 동작 |
|---|---|---|
| **Classical — flexible tempo** (`classic`, 기본값) | 클래식, 루바토, 템포가 자주 바뀌는 곡 | DTW로 음악을 프레임 단위로 따라가 템포 변화와 루바토를 따릅니다. vocal 모드에서 가수가 쉬면 다음 보컬 시작에서 기다립니다. |
| **Pop & band — steady tempo** (`pop`) | 밴드, 록, 대부분의 팝 | 밴드 템포로 계속 진행하고 DTW 결과로는 조금씩만 보정합니다. 애드리브나 놓친 부분이 있어도 멈추지 않습니다. |

- 템포가 실제로 바뀌는 부분(자유로운 인트로, 템포가 바뀌는 브리지, 끝의 ritardando)은 pop이 몇 초 늦게 따라갑니다. 이런 곡은 classic이 낫습니다.
- 곡 구조가 바뀌면(절 건너뛰기, 후렴 반복) pop은 classic보다 복구가 늦습니다. 급하면 **Go**로 옮기세요.
- 자세한 설명은 상위 README의 [추적 스타일](../README.md#추적-스타일-classic--pop)을 참고하세요.

### Input

- **Audio interface**: 장치(`Device`)와 채널을 고릅니다. 채널은 1부터 세고, 여러 개를 고르면 다운믹스합니다. 장치를 새로 연결했다면 세션을 멈춘 뒤 **Rescan devices**를 누르세요.
- **Rehearsal**: 녹음 파일을 실시간 속도로 재생해 공연처럼 따라갑니다.
  - `Play`: `This project's mix`, `This project's vocals`, `Another file…`(서버 컴퓨터의 파일 경로. 상대 경로는 이 프로젝트 폴더 기준).
  - `Play through speakers`를 켜면 소리도 함께 냅니다.

### Start at

- `Beginning`, `Cue N · 시각 메시지`, `Time…`(`1:05.200` 같은 시각) 중에서 고릅니다.
- **시작 위치에 있는 큐는 Start를 누르는 순간 나갑니다**(예: `0:00.000`의 큐, 큐 N에서 시작할 때의 큐 N).
- 시작 위치보다 앞의 큐는 건너뛴 큐로 처리합니다(**Skipped cues** 정책, 기록 이유 `start` / `skipped`). 기본값(`Send latest per address`)이면 (대상, 주소)마다 마지막 큐를 시작할 때 보내, 장비 상태를 그 지점에 맞춥니다.
- **리허설 pre-roll**: Rehearsal에서 이 프로젝트의 mix나 vocals를 큐나 시각부터 시작하면 그 2초 앞에서 재생과 추적을 시작합니다. 그래서 고른 큐와 그 앞 2초 안의 큐는 재생이 닿을 때 나갑니다.

### 세션 조작

- **Stop**, **Hold**(H) / **Resume**, **−0.5 s / +0.5 s**(`[` / `]`), 큐 목록의 **Go**.
- **Key**: −/+ 로 반음 단위 키를 정하고, **Auto**(기본 켜짐)는 처음 몇 초 동안 라이브의 키를 자동으로 찾습니다. 확신이 없으면 `+2?` 같은 제안 버튼만 띄웁니다(클릭하면 적용). 키를 손으로 정하면 Auto는 꺼집니다. 키는 프로젝트마다 따로입니다.
- **Input latency (ms)**: 오디오 입력과 처리 지연 보정입니다([OSC 지연 측정과 보정](#5-osc-지연-측정과-보정)).

### OSC ARM

오른쪽 위 큰 버튼이 전송 스위치입니다.

- **OSC LIVE**(빨강): 큐를 대상에 보냅니다.
- **SAFE — dry run**(회색): 큐를 보내지 않고 Fired 로그에 `dry`로만 남깁니다. 전송 규칙은 그대로라 리허설에서 어떤 큐가 언제 나갈지 확인할 수 있습니다.
- 설정에 저장되어 서버를 다시 켜도 유지됩니다. **Fire**, **Test**, 미리듣기 OSC는 SAFE에서도 보냅니다.

### 라이브 화면

- **상태**: `Tracking`, `Coasting`(vocal 모드에서 가수가 쉬는 동안 템포로 진행), `On hold`, `Waiting for audio`, `No audio`(1초 넘게 오디오가 오지 않음), `Stopped`, `Disconnected`.
- reference 시각(크게), Live time, Tempo(%), Confidence, Input 레벨, Key.
- **Next cue**: 다음 큐와 남은 시간, 그 뒤 두 큐. 위치가 멈춰 있거나 vocal 모드에서 목소리를 기다리는 큐는 남은 시간이 `–`입니다.
- **Fired** 로그(최신이 위): Time(보낸 시각), #, Cue(큐 시각), Target, Message, Result, Late. 위에 sent · dry · skipped · errors 개수가 나옵니다. **Clear**는 이 콘솔 화면의 로그만 지웁니다.

| Result | 뜻 |
|---|---|
| `sent` | 보냄 |
| `dry` | SAFE라서 보내지 않음 |
| `skipped` | 건너뛴 큐라서 보내지 않음 |
| `error` | 보내지 못함(대상 이름 없음, 꺼진 대상, 주소를 찾지 못함 등. 마우스를 올리면 이유) |

Result 옆의 작은 글씨는 보낸 이유입니다. 표시가 없으면 `track`입니다.

| 이유 | 뜻 |
|---|---|
| `track` | 위치가 큐 시각에 닿아 제때 보냄 |
| `catchup` | 작은 앞쪽 보정으로 위치가 큐를 지나쳐 늦게 보냄 |
| `chase` | 큰 점프로 건너뛴 큐를 **Skipped cues** 정책에 따라 보냄 |
| `start` | 시작 위치 앞의 큐를 정책에 따라 시작할 때 보냄 |
| `skipped` | 건너뛰고 보내지 않음 |
| `manual` | **Fire** 버튼 |

**Late**는 보낸 순간의 위치가 큐 시각을 지난 정도입니다(reference ms, 대상 지연의 앞당김 포함).

### Show log

세션을 시작할 때마다 `projects/<id>/logs/show-YYYYmmdd-HHMMSS.csv`를 만들고(같은 이름이 있으면 `-2`, `-3`), 보낸 큐를 한 줄씩 바로 기록합니다. 도중에 꺼져도 그때까지의 기록은 남습니다. 콘솔 왼쪽 **Show logs**에서 내려받을 수 있습니다.

| 열 | 내용 |
|---|---|
| `wall_iso` | 보낸 시각(ISO 8601, ms) |
| `live_time` | 세션 시작부터의 라이브 시간(s) |
| `ref_time` | 그때의 reference 위치(s) |
| `cue_id`, `cue_time` | 큐 줄 번호, 큐 시각(s) |
| `target`, `address`, `message` | 대상, 주소, 파일에 적힌 메시지 |
| `reason` | `track`, `catchup`, `chase`, `skipped`, `start`, `manual` |
| `sent`, `dry_run` | `true` / `false` |
| `error` | 보내지 못한 이유 |
| `ref_late_ms` | 큐 시각을 지난 정도(ms) |

## 4. 큐 전송 규칙

트래커는 40 ms 프레임마다 reference 위치와 템포를 냅니다. 그 사이는 템포로 외삽해 프레임 사이 시각에도 큐를 보냅니다. 외삽은 마지막 갱신에서 최대 0.1초까지라, 오디오가 끊기면 위치는 그 이상 가지 않습니다.

- **대상 지연만큼 미리**: latency가 L인 대상의 큐는 실제 시간으로 L만큼 일찍 보냅니다. 위치가 움직이지 않을 때(Hold, 목소리를 기다리는 동안)는 미리 보내지 않습니다.
- **한 번 지나갈 때 한 번**: 보낸(또는 건너뛴) 큐는 다시 무장되기 전까지 다시 나가지 않습니다.

| 상황 | 동작 |
|---|---|
| 앞쪽 작은 보정 (**Catch-up window** 이하, 기본 1.0 s) | 지나친 큐를 모두 늦게 보냅니다(`catchup`). |
| 앞쪽 큰 점프 (Catch-up window 초과: 재탐색, 건너뛴 절) | 건너뛴 큐에 **Skipped cues** 정책을 적용합니다. |
| 뒤로 1초 이상 (반복한 후렴, 되돌아간 연주) | 새 위치 뒤의 큐를 다시 무장합니다. 연주가 다시 닿으면 다시 나갑니다. |
| 뒤로 1초 미만 (추적 흔들림) | 아무것도 다시 보내지 않습니다. |
| 트래커가 잘못 뒤로 갔다가 돌아옴 | 이미 보낸 큐를 다시 보내지 않습니다. |
| vocal 모드, 가수가 쉬는 동안 | 다음 보컬 시작 0.1초 앞부터의 큐는 목소리가 돌아올 때까지 기다립니다. 반주만 있는 구간의 큐는 위치가 템포로 진행하며 보냅니다. |
| **Go** (큐로 점프) | 그 큐를 바로 보냅니다. 앞으로 건너뛴 큐는 정책을 따르고, 그 큐부터 뒤는 다시 무장합니다. |
| **Hold** / **Resume** | Hold 동안 위치가 멈춰 큐가 나가지 않습니다. Resume하면 트래커가 멈췄던 위치 근처에서 다시 따라잡고, 그 사이의 큐는 위치가 지나갈 때 늦게 나갑니다. 한 번에 Catch-up window보다 크게 움직이면 정책을 따릅니다. 밴드 위치를 알면 Go가 빠릅니다. |
| **+0.5 s / −0.5 s** | +0.5는 지나친 큐를 늦게 보내고, −0.5는 아무것도 다시 보내지 않습니다. |
| 공연 중 큐 파일 수정 | 그대로인 큐는 보낸 상태를 유지합니다. 새로 생기거나 시각·대상·주소·인자가 바뀐 큐는 새 큐로 봅니다. 이미 지난 위치면 보내지 않고, 아직 오지 않았으면 그때 보냅니다. |

- vocal 모드의 기다림은 classic 스타일, 그리고 pop 스타일이 아직 목소리를 잡기 전에 적용됩니다. pop이 잡은 뒤에는 쉬는 동안에도 밴드 템포로 계속 진행합니다.
- 뒤로 가는 기준(1초)은 고정입니다. Catch-up window는 OSC targets 아래에서 0.1–30초로 바꿀 수 있습니다.

**Skipped cues** (OSC targets 아래, 모든 대상 공통)

| 정책 | 동작 |
|---|---|
| `Skip` | 건너뛴 큐를 하나도 보내지 않습니다(모두 `skipped`). |
| `Send latest per address` (기본값) | (대상, 주소)마다 가장 늦은 큐 하나만 보내고(`chase`), 나머지는 `skipped`입니다. 같은 주소의 마지막 상태만 맞추면 되는 조명·믹서 큐에 맞습니다. |
| `Send all` | 건너뛴 큐를 모두 시각순으로 바로 보냅니다. |

## 5. OSC 지연 측정과 보정

지연 보정은 두 가지입니다.

| 설정 | 위치 | 보정하는 것 | 효과 |
|---|---|---|---|
| **Input latency (ms)** | Session 칸, 모든 대상 공통 | 오디오 입력(인터페이스, 드라이버)과 처리 지연 | 트래커 위치를 템포 × 지연만큼 앞당겨 모든 큐가 그만큼 일찍 나갑니다. 음수면 늦게 나갑니다. −500–2000 |
| 대상별 **Lat. ms** | OSC targets, 대상마다 | 받는 장비가 메시지를 받아 효과를 내기까지(네트워크, 장비 반응, 출력) | 그 대상의 큐만 그만큼 일찍 보냅니다. 0–5000 |

Audio interface 세션 중에는 Input latency 아래에 인터페이스가 알려 주는 입력 지연(`The audio interface reports N ms input latency.`)이 나옵니다. 출발점으로 쓰세요. 두 값 모두 설정에 저장되고, 실행 중인 세션에 바로 적용됩니다.

### 측정 (Measure)

OSC targets 각 대상의 **Measure**를 누르면 측정 창이 열립니다.

- **Network echo**: 네트워크 왕복 시간을 잽니다.
  - `oscsender echo on the receiver`: 받는 컴퓨터에도 이 프로젝트를 설치하고 다음을 실행한 뒤, `Echo port`(기본 9001), `Pings`(기본 20), `Timeout (ms)`(기본 500)를 정하고 **Run**을 누릅니다.

    ```bash
    uv run oscsender echo
    ```

  - `The receiver answers a message`: 질의에 답장을 보내는 장비(예: QLab)는 `Reply message`에 답장을 받을 메시지를 넣습니다. 정해진 포트로 답하는 장비는 `Listen port`를 넣습니다.
  - 추정값 = 왕복 시간 중앙값 / 2. 한 방향 네트워크 지연이며, 장비의 반응 시간은 들어가지 않습니다.
  - 명령줄: `uv run oscsender latency --host 192.168.0.50` (`--port` 기본 9001, `--message "/version"`처럼 답장을 받을 메시지도 가능).
- **Audio loopback**: 소리를 내는 장비(샘플러, QLab 오디오, 클릭)용입니다. 대상의 Test message를 보내고, 오디오 입력에 그 소리가 들어오는 순간을 찾습니다.
  - `Input device`와 채널, `Test message`, `Trials`(기본 8), `Threshold (dB)`(기본 20: 바닥 잡음보다 이만큼 커지면 소리로 봄)를 정합니다. 시간은 대략 Trials × 1.5초입니다.
  - 장비의 소리가 입력에 들어와야 합니다(스피커 앞의 마이크, 또는 장비 출력을 인터페이스 입력에 연결).
  - 추정값 = 보낸 순간부터 소리까지의 중앙값. PortAudio가 주는 ADC 시각으로 재므로 우리 쪽 입력 지연은 빠지고, 네트워크·장비·출력(마이크라면 공기 경로까지)이 들어갑니다.
  - 오디오 인터페이스를 쓰는 세션이 도는 동안에는 할 수 없습니다.
- 결과: 추정값, Median, Min / p95 / max, Jitter (IQR), Received(받은 수 / 보낸 수). **Use as latency**는 추정값을 ms 정수로 반올림해 그 대상의 latency에 넣습니다(loopback이면 Test message도 저장). **Cancel**은 그대로 닫습니다.

### 눈과 귀로 보정

조명·영상처럼 소리가 없는 장비의 반응 시간은 echo로 잴 수 없습니다. 미리듣기로 맞춥니다.

1. 큐 목록 위의 **Send OSC while previewing**을 켜고 레퍼런스를 재생합니다.
2. 미리듣기는 각 큐를 `큐 시각 + 브라우저 출력 지연 − 대상 latency`에 보냅니다. 공연 때와 같은 대상 latency를 씁니다.
3. 효과가 음악보다 늦으면 그 대상의 **Lat. ms**를 늘리고, 빠르면 줄입니다.

미리듣기 전송은 브라우저 타이머와 서버 요청을 거치므로 공연 경로(서버의 타이머 스레드)보다 덜 정밀합니다. 몇 번 반복해 경향을 보세요.

### 직접 입력

**Lat. ms** 칸에 숫자를 넣으면 바로 저장되고, 실행 중인 세션에도 바로 적용됩니다.

### 테크 리허설 순서 (권장)

1. 대상마다 **Test**로 메시지가 도착하는지 확인합니다. 대상 줄의 상태 칸에 보낸 수나 오류(`unresolved` 등)가 나옵니다.
2. 대상마다 **Measure**: 소리를 내는 장비는 Audio loopback, 나머지는 Network echo.
3. 소리가 없는 장비는 미리듣기와 Send OSC while previewing으로 Lat. ms를 다듬습니다.
4. **Input latency**: 인터페이스가 알려 주는 값에서 시작합니다. 실제 입력 경로(밴드, 또는 PA로 튼 리허설 녹음을 마이크·믹서로 받음)로 Audio interface 세션을 돌려, 효과가 일찍 나오면 줄이고 늦게 나오면 늘립니다. 합성 평가에서는 큐가 30–50 ms 일찍 나가는 경향이 있었습니다([정확도](#6-정확도-합성-평가)). 리허설에서도 그렇다면 음수 값으로 보정하세요.
5. SAFE로 리허설 녹음을 한 번 더 돌려 Fired 로그와 show log에서 놓치거나 두 번 나가는 큐가 없는지 확인합니다.

## 6. 정확도 (합성 평가)

아래 값은 **45초짜리 합성 곡**(약 4초마다 큐)에서 **Full band mix** 모드로 측정한 결과입니다. 합성 take의 오차는 큐가 나간 라이브 시각과 정답 라이브 시각(합성할 때 아는 시간 대응)의 차이이고, [`oscsender eval`](#내-프로젝트로-측정하기)과 같이 대상 지연과 Input latency는 0으로 둡니다.

| 조건 | 결과 |
|---|---|
| 레퍼런스 자체로 리허설(라이브 = 레퍼런스) | 모든 큐가 한 번씩 나감. 라이브 시계 기준 이상적인 시각과 0–2.4 ms 차이. 실시간 실행에서 UDP로 받은 도착 시각의 퍼짐 14 ms(큐 9개) |
| 템포가 2–8초마다 0.8–1.25× 사이에서 바뀌고 잡음·잔향·EQ를 더한 take (3개, 큐 27개) | classic: 중앙값 76 ms, p90 166 ms, 200 ms 이내 93% / pop: 중앙값 131 ms, p90 276 ms |
| 템포가 거의 일정한 take (0.95–1.05×) | pop: 중앙값 33 ms, p90 97 ms, 200 ms 이내 100% / classic: 중앙값 88 ms, p90 201 ms |
| 모든 take | 놓친 큐, 두 번 나간 큐 없음 |
| 치우침 | 약 30–50 ms 일찍 나가는 경향 |

- 템포가 계속 바뀌는 곡은 classic, 템포가 일정한 곡은 pop이 정확합니다(상위 lyricprompter의 평가와 같은 경향).
- **실제 녹음은 합성 take와 다릅니다.** 실제 밴드, 방 울림, 마이크 누화, 가수의 해석, 곡 구조 변경은 합성에 없습니다. 실제 공연의 추적 정확도는 상위 README의 [성능](../README.md#3-성능-평가-데이터-기준)을 참고하고, 자기 곡은 아래 명령과 리허설로 확인하세요.

### 내 프로젝트로 측정하기

```bash
uv run oscsender eval --project <id>
```

프로젝트의 레퍼런스로 합성 take를 만들고(2–8초 구간별 템포 변화, SNR 20 dB의 핑크 잡음, 잔향, EQ, 1초 잡음 리드인) 큐 파일의 큐를 보내 봅니다. take마다, 그리고 전체의 큐 수(`cues`), 나간 수(`fired`), 놓친 수(`missed`), 오차 절댓값의 `median` / `p90` / `max`, `bias`(부호 있는 중앙값, 음수 = 일찍), 50 / 100 / 200 ms 이내 비율을 출력합니다. 시작 위치(0초)의 큐와 레퍼런스 끝 뒤의 큐는 채점하지 않습니다.

| 옵션 | 설명 |
|---|---|
| `--seeds N` | 합성 take 수(기본 3, seed 0..N-1) |
| `--mode mix\|vocal` | 기본 mix. vocal은 보컬 stem으로 합성합니다. |
| `--style classic\|pop` | 기본은 프로젝트의 스타일 |
| `--tempo LO,HI` | take의 템포 범위(reference s / live s, 기본 `0.8,1.25`). 템포가 일정한 곡이면 `0.95,1.05` |
| `--transpose K` | take를 K 반음 올립니다. |
| `--key-preset` | 키를 자동 검출하지 않고 `--transpose` 값을 알려 줍니다. |
| `--json` | JSON으로 출력 |

리허설 녹음처럼 실제 녹음이 있으면 `simulate`로 어떤 큐가 언제 나가는지 볼 수 있습니다.

```bash
uv run oscsender simulate --project <id> --file rehearsal.wav
```

```
live    0:03.980  cue    0:04.000  ref    0:03.980  track    fx           /cue/go 1
live    0:08.000  cue    0:08.000  ref    0:08.000  track    main         /cue/go 2
```

- 기본은 오프라인 실행입니다. 실시간보다 빠르게 돌고 아무것도 보내지 않으며, 큐마다 라이브 시각, 큐 시각, reference 위치, 이유, 대상, 메시지를 출력합니다. 콘솔에 저장된 대상 latency를 쓰므로 위 예의 `fx`(latency 20 ms) 큐는 20 ms 일찍 나갑니다. Input latency는 0입니다(`--latency-ms`로 지정).
- `--realtime`(또는 `--monitor`: 소리도 냄)은 실제 세션처럼 실시간으로 돌리고, `--send`를 더하면 콘솔의 대상에 실제로 보냅니다.
- 그 밖의 옵션: `--mode`, `--style`, `--start-time S`, `--transpose K`, `--auto-key / --no-auto-key`, `--speed`.

## CLI 명령

모두 `uv run oscsender <명령>`으로 실행하고, `--help`로 옵션을 볼 수 있습니다.

| 명령 | 설명 | 주요 옵션 |
|---|---|---|
| `serve` | Run the console web server | `--host`(기본 127.0.0.1, 다른 기기에서 열려면 `0.0.0.0`), `--port`(기본 8800), `--projects-dir`, `--log-level`, `--allow-host NAME` |
| `add` | Create a project from a reference recording + cue file | `--audio`, `--title`(필수), `--cues`(생략하면 빈 큐 파일), `--vocals`, `--style classic\|pop`, `--id`, `--projects-dir` |
| `list` | List projects | `--projects-dir`, `--json` |
| `check` | Validate a cue file and print its cues | `FILE`, `--json` |
| `simulate` | Track a recording as if it were live and print the cues it fires | [위](#내-프로젝트로-측정하기) |
| `eval` | Measure cue timing accuracy on synthetic live takes of a project | [위](#내-프로젝트로-측정하기) |
| `echo` | Answer OSC pings (run on a receiving machine) | `--host`, `--port`(기본 9001), `--echo-all`(모든 메시지를 그대로 돌려줌), `--quiet` |
| `latency` | Measure the OSC round trip to a receiver | `--host`(필수), `--port`(기본 9001), `--count`, `--interval-ms`, `--timeout-ms`, `--message`, `--listen-port`, `--json` |
| `devices` | List audio input devices | `--json` |

- `add`의 id는 `--id`를 생략하면 제목으로 만들고, 같은 id가 있으면 `-2`, `-3`을 붙입니다. 출력 마지막 줄이 프로젝트 폴더입니다.
- `serve --host 0.0.0.0`으로 실행하면 시작 메시지에 다른 기기(태블릿)에서 열 주소가 함께 나옵니다. `localhost`, IP 주소, 이 컴퓨터 이름(`<이름>.local` 포함) 외의 이름으로 열려면 `--allow-host NAME`을 더하세요.

## 알려진 한계

- **OSC는 UDP만** 보냅니다. TCP와 SLIP은 지원하지 않습니다. UDP는 도착을 확인하지 않으므로 Fired 로그의 `sent`는 이 컴퓨터가 보냈다는 뜻일 뿐입니다. 도착은 Test와 리허설로 확인하세요.
- **추적의 한계는 lyricprompter와 같습니다**([상위 README의 알려진 한계](../README.md#알려진-한계)).
  - 원곡의 3배 이상 빠르거나 2배 이상 느리면 정확도가 떨어집니다.
  - 곡 구조를 크게 바꾼 경우(건너뛰기, 반복)는 전역 재탐색으로 복구되지만 몇 초 걸립니다. 그동안의 큐는 늦게 나가거나 정책에 따라 처리됩니다. **Go**가 가장 확실합니다.
  - 보컬 단독 모드에서 조옮김한 공연은 키 검출이 늦거나 제안에 그칠 수 있습니다. 키를 알면 미리 정하세요.
- **vocal 모드에는 분리한 보컬 stem이 필요합니다.** 이 프로젝트는 보컬을 분리하지 않습니다([Listen to](#listen-to)).
- **미리듣기는 브라우저 탭이 앞에 있을 때만 정확합니다.** 백그라운드 탭에서는 브라우저가 타이머를 늦춰 클릭이 빠지고 미리듣기 OSC가 늦게 나갑니다.
- **Safari에는 `outputLatency`가 없어** 들리는 위치(Set, Tap, Add cue at playhead)와 미리듣기 OSC 시각에 출력 장치의 지연(블루투스 등)이 빠집니다. 타이밍을 맞출 때는 `outputLatency`를 지원하는 브라우저(Chrome 등)를 쓰세요.
- **보안**: 서버에는 인증이 없습니다. 콘솔을 여는 사람은 누구나 세션을 조작하고 OSC를 보낼 수 있습니다. `--host 0.0.0.0`은 믿을 수 있는 무대 네트워크에서만 쓰세요.
- **음원 변환**: 오디오 스트림에 시작 오프셋이 있는 파일(일부 m4a/mp4)은 변환한 레퍼런스가 그 오프셋만큼 어긋날 수 있습니다(lyricprompter에서 복사한 변환 코드의 동작). 다른 프로그램에서 잰 시각으로 큐를 적었다면 미리듣기로 큐가 맞게 들리는지 확인하세요. 가능하면 WAV나 FLAC 레퍼런스를 쓰세요.
- **Undo 기록은 메모리에만** 있습니다. 서버를 다시 시작하면 사라집니다.

## 구조

```
src/lyricprompter_oscsender/
  cues.py        큐 파일 파싱 · 형식 · 편집 (check)
  settings.py    콘솔 설정 (.settings.json)
  osc.py         OSC 1.0 인코딩 · UDP 송신
  scheduler.py   큐 전송 시점 (CueScheduler) · 실시간 실행기 (CueRunner)
  latency.py     네트워크 echo · echo 응답기 · 오디오 loopback (echo, latency)
  project.py     프로젝트 저장소 · 튜닝 추정 · 파형 (add, list)
  synth.py       합성 라이브 take · 테스트용 합성 음악
  live/          세션 (OscSession, show log) · simulate / eval · 오디오 입력
  features/  tracking/  audio_io.py    lyricprompter에서 그대로 복사 (특징 · OLTW · 트래커 · 변환)
  server/        FastAPI + WebSocket · 콘솔 UI (vanilla JS)
docs/            SPEC.md (내부 계약) · DESIGN.md (설계)
```

| 모듈 | 책임 |
|---|---|
| `config.py` | 경로와 상수(분석 격자, 스타일, 모드) |
| `cli.py` | `oscsender` 하위 명령 디스패치 |
| `songfile.py` | stem 파일 이름(`mix`, `vocals`) |
| `audio_io.py` | ffmpeg 변환, mono 로드 (복사) |
| `features/extractor.py` | 스트리밍 특징 chroma + MFCC (복사) |
| `tracking/oltw.py`, `relocate.py`, `tracker.py` | OLTW, 전역 재탐색, LiveTracker (복사, `STYLES` import만 다름) |
| `live/audio_input.py` | 오디오 장치·파일 입력, `devices` (복사) |
| `osc.py` | OSC 메시지 인코딩/디코딩, 이름 붙은 대상으로 UDP 송신 |
| `cues.py` | 큐 파일 파싱, 시각·인자 형식, 고친 토큰만 바꾸는 편집 함수, `check` |
| `settings.py` | OSC targets, ARM, 정책, Catch-up window, Input latency 저장과 검증, `@target` 병합 |
| `scheduler.py` | 위치 외삽, 지연 앞당김, 점프 판정, 건너뛴 큐 정책, 타이머 스레드 |
| `latency.py` | echo 측정과 응답기, 오디오 loopback 측정, `echo` / `latency` |
| `project.py` | 프로젝트 생성·삭제(휴지통), 튜닝 추정, 파형 캐시, `add` / `list` |
| `synth.py` | 합성 라이브 take(lyricprompter 평가 코드 복사)와 테스트용 합성 음악 |
| `live/session.py` | 입력 → 특징 → 트래커 → 큐 실행기, 운영자 조작, show log, 상태 스냅샷 |
| `live/simulate.py` | 오프라인 큐 전송, `simulate` / `eval` |
| `server/app.py` | REST + WebSocket, 큐 편집과 Undo, 파일 감시, 세션, `serve` |
| `server/static/` | 콘솔 `console.html` / `console.css` / `console.js`, `common.js` |

**데이터 위치**: 기본은 이 폴더의 `projects/`입니다(`--projects-dir` 또는 환경 변수 `OSCSENDER_PROJECTS`로 바꿈). `projects/`는 `.gitignore`로 Git에서 제외됩니다. 백업하려면 `projects/` 폴더 전체를 복사하세요.

```
projects/<id>/project.json      제목, 만든 시각, 길이, 튜닝, 스타일, 원본 파일 이름, stem 목록
projects/<id>/audio/mix.wav     레퍼런스 (44.1 kHz stereo PCM24로 변환)
projects/<id>/audio/vocals.wav  보컬 stem (선택, 레퍼런스 길이에 맞춤)
projects/<id>/cues.txt          큐 파일 (원본, 직접 편집 가능)
projects/<id>/cache/            reference 특징 캐시, 파형 캐시
projects/<id>/logs/             show log (CSV)
projects/.settings.json         OSC targets, ARM, 정책, Catch-up window, Input latency
projects/.trash/                지운 프로젝트 (<id>-<YYYYmmdd-HHMMSS>)
```

설정 파일이 손상되면 서버는 잘못된 값만 기본값으로 바꾸고(잘못된 대상은 끈 채로 남김) 시작하며, OSC targets 위에 무엇을 고쳤는지 보여줍니다. 원래 파일은 `.settings.json.bad-<hash>`로 남깁니다.

**테스트**:

```bash
uv run pytest -q
```

```bash
uv run ruff check
```

설계와 내부 동작은 [docs/DESIGN.md](docs/DESIGN.md), 정확한 계약은 [docs/SPEC.md](docs/SPEC.md)에 있습니다.
