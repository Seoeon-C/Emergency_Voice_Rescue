import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# 1. 경로 설정
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
beats_path = str(current_dir / "beats")
if beats_path not in sys.path:
    sys.path.insert(0, beats_path)

app = FastAPI()

# 2. CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. DecisionResult 클래스 직접 정의 (ImportError 방지)
class DecisionResult:
    def __init__(self, situation, situation_name, risk_level, reason, action, tts_key, send_to_control_room, emergency_candidate, source):
        self.situation = situation
        self.situation_name = situation_name
        self.risk_level = risk_level
        self.reason = reason
        self.action = action
        self.tts_key = tts_key
        self.send_to_control_room = send_to_control_room
        self.emergency_candidate = emergency_candidate
        self.source = source

guard_app = None

def get_guard_app():
    global guard_app
    if guard_app is None:
        import main
        guard_app = main.SoundGuardApp()
    return guard_app

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ [Server] 리액트 대시보드 연결 성공")

    try:
        app_instance = get_guard_app() 
        import main # 로직 참조용
        
        while True:
            # 변수 초기화 (UnboundLocalError 방지)
            stt_text = ""
            
            # 리액트에게 상태 알림
            await websocket.send_json({"type": "status", "message": "recording"})
            
            # A. 녹음 및 저장
            audio = app_instance._record_audio()
            audio_path = app_instance._save_audio(audio)

            # B. 환경 소리 분석 (main.settings 대신 getattr 사용으로 안전하게 가져옴)
            sample_rate = getattr(app_instance, 'sample_rate', 16000)
            sound_event = app_instance.env_classifier.classify(audio, sample_rate)

            # C. STT 트리거 로직 (수치 직접 입력으로 에러 방지)
            min_rms = 0.004  
            min_peak = 0.03
            stt_trigger = (
                sound_event.situation in {1, 2}
                or sound_event.rms >= min_rms
                or sound_event.peak >= min_peak
            )

            if stt_trigger:
                stt_text = app_instance._try_stt(audio, audio_path)

            # D. 체류 시간 및 상황 판단
            dwell_seconds = app_instance.dwell_tracker.update(sound_event, stt_text=stt_text)
            
            decision = app_instance.decision_engine.decide(
                sound_event=sound_event,
                stt_text=stt_text,
                dwell_seconds=dwell_seconds,
                authorized=False,
            )

            # [튜닝 로직] 1차 경고 미발령 시 2차 경고 강제 변환
            if decision.tts_key == "INTRUSION_WARN_2" and not app_instance.dwell_tracker.warn1_issued:
                decision = DecisionResult(
                    situation=decision.situation,
                    situation_name=decision.situation_name,
                    risk_level="low",
                    reason=decision.reason,
                    action="1차 경고 방송",
                    tts_key="INTRUSION_WARN_1",
                    send_to_control_room=decision.send_to_control_room,
                    emergency_candidate=decision.emergency_candidate,
                    source=decision.source + " (forced warn1)",
                )

            # 1차 발령 기록 업데이트
            if decision.tts_key == "INTRUSION_WARN_1" and sound_event.situation in {1, 2}:
                app_instance.dwell_tracker.warn1_issued = True

            # E. 데이터 전송
            payload = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "env_label": sound_event.label,
                "stt_text": stt_text,
                "situation": decision.situation,
                "situation_name": decision.situation_name,
                "reason": decision.reason,
                "action": decision.action,
                "dwell_seconds": round(dwell_seconds, 1)
            }
            await websocket.send_json(payload)
            print(f"📡 [보냄] {sound_event.label} ({decision.situation_name})")

            # F. TTS 출력
            if decision.tts_key != "NONE":
                app_instance.speaker.speak(decision.tts_key)

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        print("🔌 [Server] 리액트 연결 종료")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ [Server] 에러 발생: {e}")
    finally:
        print("🔌 [Server] WebSocket 세션 종료")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)