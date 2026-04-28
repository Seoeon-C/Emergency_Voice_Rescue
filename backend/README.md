# SoundGuard

BEATs, Whisper API, Gemini 1.5 Pro, 고정 TTS를 사용하는  
음향 기반 위험구역 무단침입 및 위험 감지 시스템 MVP입니다.

## 적용 가정

출입 허가되지 않은 위험 구역에 누군가 무단으로 침입했을 때를 가정합니다.

예시 장소:

- 산
- 밭
- 폐공사장
- 폐건물
- 시골 외지 개인 사유 구역
- 저수지
- 갯벌
- 해안가 구역
- 군사지역

## 판단 상황

### 상황 0: 이상없음

감지 조건 미충족.

### 상황 1: 무단침입

발소리, 사람 음성 등으로 사람 존재 여부를 판단하고, 체류시간 기반으로 1차/2차 대응을 수행합니다.

- 5초 이상 체류: 1차 경고 방송
- 15초 이상 체류: 2차 경고 방송 + 상황실 전송

### 상황 2: 위험 감지

비명, 충격음, 유리 깨짐 또는 STT 기반 응급구조신호가 감지되면 위험 상황으로 판단합니다.

예:

- 도와주세요
- 살려주세요
- 119 불러줘
- 사람이 쓰러졌어요
- 불났어요

## 파일 구조

```text
main.py                    # 전체 실행
environmental_sound.py     # BEATs 환경음 분류
stt.py                     # Whisper API STT
decision.py                # Gemini 1.5 Pro 판단
output.py                  # 고정 TTS 출력, 로그 기록, 상황실 전송
config.py                  # 환경변수 설정
.env.example               # 환경변수 예시
requirements.txt           # 의존성
```

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## API 키 설정

`.env`에 입력하세요.

```env
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## BEATs 준비

프로젝트 루트 기준으로 아래처럼 배치합니다.

```text
beats/
└─ BEATs.py

checkpoints/
└─ BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
```

`.env` 경로:

```env
BEATS_PY_DIR=beats
BEATS_CHECKPOINT_PATH=checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
```

BEATs 파일이 없으면 fallback 모드로 실행됩니다.  
단, fallback은 실제 분류용이 아니라 실행 흐름 확인용입니다.

## 고정 TTS 음성 파일

`assets/tts/` 폴더에 아래 파일을 넣으면 실제 음성처럼 재생됩니다.

```text
assets/tts/INTRUSION_WARN_1.mp3
assets/tts/INTRUSION_WARN_2.mp3
assets/tts/EMERGENCY_GUIDE.mp3
assets/tts/EVACUATION_GUIDE.mp3
```

파일이 없으면 콘솔에 문구만 출력됩니다.

## 실행

```bash
python main.py
```

## 로그

이벤트 로그는 아래에 저장됩니다.

```text
outputs/logs/events_YYYYMMDD.jsonl
```

## 상황실 전송

`.env`의 `CONTROL_ROOM_WEBHOOK`에 URL을 넣으면 POST 방식으로 이벤트를 전송합니다.

```env
CONTROL_ROOM_WEBHOOK=https://example.com/webhook
```

없으면 로컬 로그만 저장됩니다.

## 허가 사용자 인증

기본 비밀번호:

```text
1234
```

`.env`에서 변경할 수 있습니다.

```env
AUTH_PASSWORD=1234
AUTH_DISABLE_SECONDS=10
```

인증 성공 시 10초 동안 감지 로직이 꺼집니다.
