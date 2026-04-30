# 0430 통합본 작업 내역

## 작업 개요

`_2` 파일들을 베이스로, `_3` 파일들의 신규 기능을 이식하여 `_4` 통합본을 생성.
원본 파일들은 모두 그대로 유지.

---

## 신규 생성 파일 목록

| 파일 | 베이스 | 추가 출처 |
|---|---|---|
| `backend/decision_4.py` | `decision_2.py` | `decision_3.py` |
| `backend/stt_4.py` | `stt_re.py` | - |
| `backend/output_4.py` | `output.py` | `output_3.py` |
| `backend/app_4.py` | `app_2.py` | `app_3.py` |
| `main_4.py` | `main_2.py` | - |

---

## 파일별 변경 내용

### `decision_4.py`

**`decision_2.py` 대비 변경점:**

- OpenAI 클라이언트 생성 시 API 키 없으면 `None`으로 처리
  ```python
  # 기존 (decision_2)
  client = OpenAI(api_key=settings.openai_api_key)

  # 변경 (decision_4)
  client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
  ```
- `_ask_gpt()` 함수 시작 시 `client is None` 조기 반환 추가 → API 키 없어도 오류 없이 룰 기반으로 폴백

**`decision_2.py`에서 유지된 것:**
- `dwell >= settings.intrusion_warn_1_seconds` (설정값 사용, 하드코딩 아님)
- `decide()` 마지막 emergency tts 보정 로직 (`situation=2`인데 tts_key가 침입 경고면 `EMERGENCY_GUIDE`로 자동 수정)

---

### `stt_4.py`

**`stt.py` 대비 변경점 (`stt_re.py` 내용 반영):**

- `_clean_transcript()` 로직 순서 변경: **응급 키워드 체크를 환각 필터보다 먼저 실행**
  - 기존: 환각 필터(부분 포함 `in`) → 응급 키워드
  - 변경: 응급 키워드 → 환각 필터(완전 일치 `==`)
  - 이유: 기존 방식은 응급 키워드가 포함된 문장이 환각으로 오인 차단될 수 있었음
- 환각 탐지 방식 완화: `in` (부분 포함) → `==` (완전 일치)
  - 이유: "감사합니다"가 포함된 정상 발화를 환각으로 잘못 처리하는 문제 방지

---

### `output_4.py`

**`output.py` 대비 변경점:**

- pygame 시작 배너 억제 추가
  ```python
  import os
  os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
  ```
- `DecisionResult` import를 `decision_4`에서 가져오도록 변경

---

### `app_4.py`

**`app_2.py` 대비 변경점 (`app_3.py` 기능 이식):**

#### 1. selfcheck 연동
- `from .selfcheck import run_self_check` 추가 (import 명 버그 수정: `self_check` → `selfcheck`)
- `_run_self_check()` 메서드 추가: 실행 중 자가진단 실행, `audio_lock`으로 녹음과 충돌 방지

#### 2. `AuthorizationManager` 확장
- `self_check_callback` 파라미터 추가
- `_control_active` 플래그 추가 (`is_control_active()`, `_set_control_active()`)
- CLI에 `c` 명령어 추가 → 실행 중 자가진단 트리거
- `_run_control_command()` 헬퍼 추가: 제어 명령 실행 시 감지 일시정지 → 명령 실행 → 감지 재개 패턴 통일

#### 3. `SoundGuardApp` 확장
- `audio_lock = threading.Lock()` 추가 → 녹음 중 자가진단 실행 방지
- `control_pause_notified` 플래그 추가 → 일시정지 메시지 중복 출력 방지
- `_record_audio()` 내부 `sd.rec()`/`sd.wait()`를 `audio_lock`으로 감쌈

#### 4. 메인 루프 변경
- `is_control_active()` 체크 추가 (루프 최상단): 제어 명령 처리 중에는 감지 즉시 정지
- 녹음 후 `is_disabled() or is_control_active()` 이중 체크로 엣지케이스 처리

**`app_2.py`에서 유지된 것 (app_3에서 빠져있던 기능들):**
- `_apply_emergency_lock()`: 응급 잠금 3분 유지, back-off 재안내 (즉시 → 30초 → 60초 간격)
- `warn1_issued`: `SoundGuardApp` 레벨 플래그 (DwellTracker reset에 영향 안 받음)
- `silence_cycles`: 30초 연속 이상없음 시 경고 상태 자동 초기화
- 에스컬레이션 로직: warn1 발령 후 재감지 시 자동으로 warn2 승격
- `from .decision_4 import` (decision_2 대신 decision_4 사용)

---

### `main_4.py`

```python
from backend.app_4 import SoundGuardApp

if __name__ == "__main__":
    SoundGuardApp().run()
```

---

## 주의사항

- `selfcheck/` 폴더는 `_3` 파일들과 함께 추가된 신규 모듈로, `app_4.py`가 직접 의존
- `app_3.py`에서 `from .self_check import ...`로 오타가 있었던 것을 `app_4.py`에서 `from .selfcheck import ...`로 수정
- `config`는 `_3`과 원본이 동일하여 `config_4.py` 별도 생성 없이 기존 `config.py` 그대로 사용
