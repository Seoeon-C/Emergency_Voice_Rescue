📖 팀 프로젝트 협업 가이드 (Git/GitHub)
이 문서는 우리 Emergency Voice Rescue 프로젝트의 협업 규칙을 담고 있습니다. 모두의 코드가 섞이지 않고 안전하게 관리될 수 있도록 아래 절차를 꼭 숙지해 주세요!

1. 🛠️ 환경 구축 (초기 세팅)
프로젝트를 본인의 노트북으로 처음 가져올 때 실행합니다.

Bash
# 1. 저장소 복제 (Clone)
git clone https://github.com/Seoeon-C/Emergency_Voice_Rescue.git
cd Emergency_Voice_Rescue

# 2. dev 브랜치로 이동 (우리 프로젝트의 기준점입니다)
git checkout dev
2. 🌿 브랜치 관리 전략
우리 프로젝트는 Main - Dev - Feature 구조로 운영됩니다.

main: 최종 배포용 브랜치 (팀장만 건드림)

dev: 팀원들의 코드를 합치는 통합 개발 브랜치

feature/기능명: 여러분이 실제로 작업할 개인 브랜치

💡 새 작업 시작할 때 (반드시 수행)
Bash
# 항상 dev에서 최신 코드를 받아온 뒤 시작하세요
git checkout dev
git pull origin dev

# 본인만의 기능 브랜치 생성 (예: feature/audio-ml, feature/nlp-gemma 등)
git checkout -b feature/본인기능명
3. 📤 작업 저장 및 업로드 (Commit & Push)
코드를 수정했다면 본인의 온라인 브랜치에 저장해야 합니다.

Bash
# 1. 변경된 파일 확인
git status

# 2. 저장할 파일 선택 (데이터 파일이 포함되지 않게 주의!)
git add .

# 3. 커밋 메시지 작성 (무엇을 했는지 명확하게)
git commit -m "Feat: 실시간 음성 감지 필터링 로직 구현"

# 4. 내 브랜치에 올리기
git push origin feature/본인기능명
🤝 4. 내 코드 합치기 (Pull Request - PR)
내 브랜치에 올린 코드를 공용 브랜치인 dev에 합쳐달라고 팀장에게 요청하는 과정입니다.

GitHub 웹사이트 접속

상단의 [Compare & pull request] 노란색 버튼 클릭

중요: base: dev ← compare: feature/본인기능명 인지 꼭 확인! (main이 아님)

작업 내용 요약 작성 후 Create pull request 클릭

팀장의 코드 리뷰 및 승인(Merge) 기다리기

🚫 5. 주의사항: 절대 올리면 안 되는 것들
우리 프로젝트는 용량 제한과 보안을 위해 아래 항목들을 .gitignore에 등록해 두었습니다. 혹시라도 올라가지 않도록 주의해 주세요.

data/: .wav, .npy 등 모든 원본 및 전처리 데이터

models/: 학습 완료된 가중치 파일 (.pkl, .eim, .tflite)

venv/, node_modules/: 개별 환경 설치 폴더

.env: API 키 등 개인 설정 파일