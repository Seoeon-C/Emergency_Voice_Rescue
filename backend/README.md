# SoundGuard - GPT 판단 / STT 균형 버전

이 버전은 이전 버전에서 STT가 너무 보수적으로 막히던 문제를 수정했습니다.

## 핵심 변경

- `MIN_RMS_FOR_STT`: 0.012 → 0.006
- `MIN_PEAK_FOR_STT`: 0.080 → 0.050
- `ALLOW_UNKNOWN_STT=true` 추가
- BEATs fallback 상태에서 `unknown`이라도 음량이 충분하면 STT 수행
- 단, STT 결과가 없으면 `unknown`은 체류시간으로 누적하지 않음
- 자막형 환각 문구는 계속 필터링

## .env 예시

```env
OPENAI_API_KEY=sk-...
OPENAI_STT_MODEL=whisper-1
OPENAI_LLM_MODEL=gpt-4o-mini

BEATS_PY_DIR=beats
BEATS_CHECKPOINT_PATH=checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt

DEVICE=cpu
SAMPLE_RATE=16000
CHUNK_SECONDS=5

MIN_RMS_FOR_STT=0.006
MIN_PEAK_FOR_STT=0.050
ALLOW_UNKNOWN_STT=true

ZONE_NAME=위험구역 A
LOCATION_TEXT=폐공사장 A구역 입구
LATITUDE=37.000000
LONGITUDE=127.000000

CONTROL_ROOM_WEBHOOK=

AUTH_PASSWORD=1234
AUTH_DISABLE_SECONDS=10
```

## 기준 조정 팁

현재 로그가 아래처럼 나오면:

```text
rms=0.00954, peak=0.33096
```

이 값은 `MIN_RMS_FOR_STT=0.006`, `MIN_PEAK_FOR_STT=0.050`보다 크므로 STT를 시도합니다.

무음인데도 STT가 자주 돈다면:
- `MIN_RMS_FOR_STT=0.010`
- `MIN_PEAK_FOR_STT=0.080`

말했는데도 STT가 안 돈다면:
- `MIN_RMS_FOR_STT=0.003`
- `MIN_PEAK_FOR_STT=0.030`
