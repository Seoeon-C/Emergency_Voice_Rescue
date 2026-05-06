# SoundGuard 서버 운영 가이드

> 아래 명령어에서 `<값>` 부분만 본인 환경에 맞게 교체해서 사용하세요.

---

## 공통 변수

| 변수 | 값 |
|------|-----|
| `<SSH_KEY>` | `C:\Users\사용자이름\.ssh\사용자키파일.pem` |
| `<SERVER_IP>` | `SERVER_IP` |

---

## 1. 서버 SSH 접속

```powershell
ssh -i <SSH_KEY> ubuntu@<SERVER_IP>
```

---

## 2. 파일 서버에 올리기

### 단일 파일 업로드

```powershell
scp -i <SSH_KEY> <로컬_파일_경로> ubuntu@<SERVER_IP>:~/backend_on_server/
```

예시:
```powershell
scp -i C:\Users\seoeo\.ssh\Oracle_SSHKey.pem backend_on_server\server.py ubuntu@SERVER_IP:~/backend_on_server/
```

### 폴더 전체 업로드

```powershell
scp -i <SSH_KEY> -r <로컬_폴더_경로> ubuntu@<SERVER_IP>:~/backend_on_server/
```

### 프론트엔드 빌드 후 업로드

```powershell
cd frontend
npm run build
cd ..
scp -i <SSH_KEY> -r frontend\dist\* ubuntu@<SERVER_IP>:/var/www/html/
```

업로드 후 서버에서:
```bash
chmod 755 /var/www/html/assets
```

---

## 3. 백엔드 서비스 관리

### 서비스 재시작 (파일 업로드 후 반드시 실행)

```bash
sudo systemctl restart soundguard
```

### 서비스 상태 확인

```bash
sudo systemctl status soundguard
```

### 서비스 시작 / 중지

```bash
sudo systemctl start soundguard   # 시작
sudo systemctl stop soundguard    # 중지
```

---

## 4. 로그 확인

### 실시간 로그 (Ctrl+C로 종료)

```bash
sudo journalctl -u soundguard -f
```

### 최근 N줄 확인 (즉시 출력)

```bash
sudo journalctl -u soundguard -n 50 --no-pager
```

### 키워드 필터

```bash
sudo journalctl -u soundguard --no-pager | grep "<검색어>"
```

예시:
```bash
sudo journalctl -u soundguard --no-pager | grep "upload"
sudo journalctl -u soundguard --no-pager | grep "사용 모델"
sudo journalctl -u soundguard --no-pager | grep "TTS 파일"
```

### 오늘 로그만

```bash
sudo journalctl -u soundguard --since today --no-pager
```

### 에러 로그만

```bash
sudo journalctl -u soundguard -p err --no-pager
```

---

## 5. 파일 확인

### TTS mp3 파일 확인

```bash
ls -lh ~/backend_on_server/assets/tts/
```

### 수신된 오디오 파일 확인

```bash
ls -lh ~/backend_on_server/received/
```

### 수신된 오디오 파일 삭제

```bash
rm ~/backend_on_server/received/*
```

### 현재 사용 중인 AI 모델 확인

```bash
sudo journalctl -u soundguard --no-pager | grep "사용 모델"
```

### 구역 DB 확인

```bash
ls -lh ~/backend_on_server/zones.db
```

---

## 6. 디스크 / 메모리 확인

```bash
free -h          # 메모리 사용량
df -h            # 디스크 사용량
```

---

## 7. 접속 정보 요약

| 항목 | 값 |
|------|-----|
| 대시보드 URL | `http://SERVER_IP` |
| 로그인 ID | `admin` |
| 로그인 PW | (팀 내부 공유) |
| 백엔드 포트 | `8000` |
| SSH 유저 | `ubuntu` |
