# SoundGuard

SoundGuard는 소리를 기반으로 위험 구역의 무단침입과 응급상황을 감지하는 프로젝트입니다.

마이크로 5초 단위 음성을 수집한 뒤, 프로젝트 데이터로 전이학습한 BEATs 모델이 환경음을 5개 클래스로 분류합니다. 이후 Whisper STT와 판단 로직을 함께 사용해 배경음, 침입 가능성, 긴급 구조 요청을 구분합니다.

## 처리 흐름

```text
소리 수집
-> BEATs 전이학습 모델 분류
-> background이면 이벤트 없음
-> intrusion이면 경고 후 지속 감지 시 상황실 전송
-> emergency이면 즉시 긴급 알림
-> 필요 시 Whisper STT로 구조 요청 문장 확인
```

## 구조

```text
main.py   실행 파일
backend/  최종 백엔드 코드
```

## 실행

```powershell
cd C:\Users\Chan\Desktop\a
C:\Users\Chan\anaconda3\envs\firstaid-gpu\python.exe main.py
```

처음 받는 팀원은 `backend\.env.example`을 참고해 `backend\.env`를 만들고 API 키를 입력해야 합니다.

## 모델 파일

전이학습 모델과 원본 BEATs 체크포인트는 용량이 커서 git에 올리지 않습니다.

필요한 로컬 위치:

```text
backend/checkpoints/best_beats_project.pt
backend/checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
```

팀원에게는 위 두 파일을 별도로 전달하거나 Git LFS/클라우드 링크를 사용하세요.

## 전이학습 클래스

```text
background
intrusion
emergency
impact_noise
loud_noise
```

전처리/전이학습 실험 코드와 성능 그래프 자료는 git에서 제외하고 별도 백업 폴더에 보관합니다.
