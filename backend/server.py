import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# 경로 설정
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
if str(current_dir / "beats") not in sys.path:
    sys.path.insert(0, str(current_dir / "beats"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# DecisionResult 클래스 (main.py와 일치)
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
    app_instance = get_guard_app()
    print("✅ [Control Room] 연결 성공")

    try:
        while True:
            # 1. 시스템 제어 메시지 확인 (비동기 대기)

            # 2. 일시정지 상태면 분석 생략
            if app_instance.auth.is_disabled():
                await websocket.send_json({"type": "status", "message": "paused"})
                await asyncio.sleep(1)
                continue
            try:
                # 0.1초 동안 프론트엔드에서 보낸 명령이 있는지 확인
                raw_command = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                command = json.loads(raw_command)
                
                if command.get("type") == "CONTROL":
                    action = command.get("action")
                    if action == "FORCE_TTS":
                        key = command.get("key", "INTRUSION_WARN_1")
                        app_instance.speaker.speak(key)
                        print(f"📣 [CONTROL] 강제 방송 송출: {key}")
                    elif action == "PAUSE":
                        # 60초간 비활성화 (main.py의 AuthorizationManager 활용)
                        app_instance.auth._disable_until = datetime.now().timestamp() + 60
                        print("⏸ [CONTROL] 시스템 60초 일시정지")
            except asyncio.TimeoutError:
                pass # 명령 없으면 감지 로직 계속 실행

            # 2. 감지 및 분석 로직
            if app_instance.auth.is_disabled():
                await websocket.send_json({"type": "status", "message": "paused", "remain": round(app_instance.auth.remaining_seconds())})
                await asyncio.sleep(1)
                continue

            await websocket.send_json({"type": "status", "message": "recording"})
            
            # 음성 획득 및 분석
            audio = app_instance._record_audio()
            audio_path = app_instance._save_audio(audio)
            
            sample_rate = getattr(app_instance, 'sample_rate', 16000)
            sound_event = app_instance.env_classifier.classify(audio, sample_rate)
            
            # STT 로직
            stt_text = ""
            if sound_event.situation in {1, 2} or sound_event.rms >= 0.004:
                stt_text = app_instance._try_stt(audio, audio_path)

            dwell_seconds = app_instance.dwell_tracker.update(sound_event, stt_text=stt_text)
            decision = app_instance.decision_engine.decide(sound_event, stt_text, dwell_seconds, False)


            
            # 데이터 전송 (이미지 레이아웃에 맞춤)
            payload = {
                "type": "data",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "status": {
                    "level": decision.situation,
                    "name": decision.situation_name,
                    "duration": round(dwell_seconds)
                },
                "analysis": {
                    "label": sound_event.label,
                    "rms": float(sound_event.rms),
                    "peak": float(sound_event.peak),
                    # 대시보드 그래프용 확률 (가상값 포함)
                    "scores": {
                        "footstep": round(float(sound_event.rms * 10000), 1), 
                        "voice": 10 if stt_text else 0,
                        "noise": round(float(sound_event.peak * 100), 1)
                    }
                },
                "stt_text": stt_text,
                "action_msg": decision.action
            }
            await websocket.send_json(payload)
            

            if decision.tts_key != "NONE":
                app_instance.speaker.speak(decision.tts_key)

    except WebSocketDisconnect:
        print("🔌 연결 종료")
    except Exception as e:
        print(f"❌ 에러: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)