# SoundGuard - refined rules v2

이번 버전은 `아파요` 같은 짧은 사람 말소리가 BEATs에서 `unknown`으로 나와도 STT에 들어가도록 완화한 버전입니다.

## 핵심 변경

- `ALLOW_UNKNOWN_STT=true`
- `MIN_RMS_FOR_STT=0.004`
- `MIN_PEAK_FOR_STT=0.030`
- `unknown`이라도 음량이 충분하면 `speech candidate`로 보고 Whisper STT 실행
- `아파요`, `도와`, `살려`, `119`, `다쳤`, `쓰러` 같은 짧은 위급 표현은 STT 필터에서 제거하지 않음

## 흐름

```text
BEATs 분류
├─ nature -> pass
├─ speech -> Whisper STT -> GPT/Rule 판단
├─ footstep -> 무단침입
├─ emergency_sound -> 위급
└─ unknown
   ├─ 음량 충분 -> Whisper STT
   └─ 음량 부족 -> pass
```

## .env 권장값

```env
MIN_RMS_FOR_STT=0.004
MIN_PEAK_FOR_STT=0.030
ALLOW_UNKNOWN_STT=true
```

말했는데도 STT가 안 되면 더 낮추세요.

```env
MIN_RMS_FOR_STT=0.002
MIN_PEAK_FOR_STT=0.020
```

무음인데 STT가 너무 자주 돌면 올리세요.

```env
MIN_RMS_FOR_STT=0.008
MIN_PEAK_FOR_STT=0.060
```
