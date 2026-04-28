\
"""
main.py

SoundGuard 실행 파일

사용 모델:
- BEATs: 환경음/위험음 분류
- Whisper API: 사람 말소리 STT
- Gemini 1.5 Pro: 상황 0/1/2 판단
- Fixed TTS: 저장된 고정 안내 음성 출력

실행:
    python main.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# 현재 main.py가 있는 폴더의 절대 경로를 구합니다.
current_dir = Path(__file__).resolve().parent
# beats 폴더의 절대 경로를 시스템 경로(sys.path)에 추가합니다.
beats_path = str(current_dir / "beats")
if beats_path not in sys.path:
    sys.path.insert(0, beats_path)

import getpass
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import settings
from decision import GeminiDecisionEngine
from environmental_sound import BeatsEnvironmentClassifier, SoundEvent
from output import EventLoggerAndMessenger, FixedMessageSpeaker
from stt import WhisperAPI


TEMP_DIR = Path("outputs/temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class AuthorizationManager:
    """
    허가된 사용자 구분:
    - 버튼 대신 콘솔 입력으로 구현
    - 비밀번호 성공 시 10초 동안 감지 로직 OFF
    """

    def __init__(self) -> None:
        self.disabled_until = 0.0

    def is_disabled(self) -> bool:
        return time.time() < self.disabled_until

    def remaining_seconds(self) -> int:
        return max(0, int(self.disabled_until - time.time()))

    def request_auth_if_needed(self) -> bool:
        print("[AUTH] 허가 사용자라면 비밀번호를 입력하세요. 아니면 Enter를 누르세요.")
        password = getpass.getpass("password: ")

        if not password:
            return False

        if password == settings.auth_password:
            self.disabled_until = time.time() + settings.auth_disable_seconds
            print(f"[AUTH] 인증 성공. {settings.auth_disable_seconds}초 동안 감지 로직을 끕니다.")
            return True

        print("[AUTH] 인증 실패.")
        return False


class DwellTimeTracker:
    """
    위험구역 체류시간 계산.
    실제 제품에서는 PIR 센서, 거리 센서, 라이다, GPS, 카메라 등의 접근 정보와 결합할 수 있음.
    이 MVP에서는 '사람 존재 추정 소리'가 연속 감지되는 시간을 체류시간으로 봄.
    """

    def __init__(self) -> None:
        self.person_detected_since: Optional[float] = None

    def update(self, sound_event: SoundEvent) -> float:
        now = time.time()

        if sound_event.person_detected or sound_event.label in {"footstep", "speech", "unknown"}:
            if self.person_detected_since is None:
                self.person_detected_since = now
            return now - self.person_detected_since

        if not sound_event.danger_sound_detected:
            self.person_detected_since = None

        return 0.0


class SoundGuardApp:
    def __init__(self) -> None:
        self.env_classifier = BeatsEnvironmentClassifier()
        self.stt = WhisperAPI()
        self.decision_engine = GeminiDecisionEngine()
        self.speaker = FixedMessageSpeaker()
        self.logger = EventLoggerAndMessenger()
        self.auth = AuthorizationManager()
        self.dwell_tracker = DwellTimeTracker()

    def run(self) -> None:
        print("=" * 70)
        print("SoundGuard 실행")
        print(f"위험구역: {settings.zone_name}")
        print(f"위치: {settings.location_text}")
        print(f"좌표: {settings.latitude}, {settings.longitude}")
        print("종료: Ctrl+C")
        print("=" * 70)

        try:
            while True:
                if self.auth.is_disabled():
                    print(f"[AUTH] 허가 사용자 통과 중. 남은 시간: {self.auth.remaining_seconds()}초")
                    time.sleep(1)
                    continue

                audio = self._record_audio()
                audio_path = self._save_audio(audio)

                sound_event = self.env_classifier.classify(audio, settings.sample_rate)
                print(
                    f"[ENV] label={sound_event.label}, "
                    f"conf={sound_event.confidence:.3f}, "
                    f"raw={sound_event.raw_label}"
                )

                dwell_seconds = self.dwell_tracker.update(sound_event)
                print(f"[ZONE] 체류 추정 시간: {dwell_seconds:.1f}초")

                # 사람 말소리 또는 분류 불확실한 경우 STT 시도
                stt_text = ""
                if sound_event.label in {"speech", "unknown"} or sound_event.person_detected:
                    try:
                        stt_text = self.stt.transcribe(audio_path)
                        if stt_text:
                            print(f"[STT] {stt_text}")
                    except Exception as exc:
                        print(f"[WARN] STT 실패: {exc}")

                # 초기 침입 감지 단계에서 허가 사용자 인증 기회 제공
                authorized = False
                if (sound_event.person_detected or sound_event.label == "unknown") and dwell_seconds < 5:
                    authorized = self.auth.request_auth_if_needed()

                decision = self.decision_engine.decide(
                    sound_event=sound_event,
                    stt_text=stt_text,
                    dwell_seconds=dwell_seconds,
                    authorized=authorized,
                )

                print(
                    f"[DECISION] 상황 {decision.situation}: {decision.situation_name} | "
                    f"위험도={decision.risk_level} | {decision.reason}"
                )
                print(f"[ACTION] {decision.action}")

                if decision.tts_key != "NONE":
                    self.speaker.speak(decision.tts_key)

                # 상황 0도 로컬 기록하고 싶다면 아래 조건을 제거하면 됨.
                if decision.situation in {1, 2} or decision.send_to_control_room:
                    event = self.logger.record_and_send(
                        decision=decision,
                        sound_event=sound_event,
                        stt_text=stt_text,
                        dwell_seconds=dwell_seconds,
                    )
                    print(f"[EVENT] 기록/전송 대상 이벤트: {event['event_time']}")

                print("-" * 70)

        except KeyboardInterrupt:
            print("\nSoundGuard 종료")

    def _record_audio(self) -> np.ndarray:
        print(f"[REC] {settings.chunk_seconds}초 녹음 중...")
        audio = sd.rec(
            int(settings.sample_rate * settings.chunk_seconds),
            samplerate=settings.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio.squeeze()

    def _save_audio(self, audio: np.ndarray) -> Path:
        path = TEMP_DIR / "latest.wav"
        sf.write(path, audio, settings.sample_rate)
        return path


if __name__ == "__main__":
    app = SoundGuardApp()
    app.run()
