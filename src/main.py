from faster_whisper import WhisperModel
import torch

print("모델 로드 중...")
# GPU(cuda)를 사용하고, 메모리 절약을 위해 float16 연산 사용
model = WhisperModel("base", device="cuda", compute_type="float16")

print("학습 환경 준비 완료!")
print(f"사용 장치: {torch.cuda.get_device_name(0)}")
print("이제 data/ 폴더에 음성 파일을 넣고 테스트할 준비가 되었습니다.")