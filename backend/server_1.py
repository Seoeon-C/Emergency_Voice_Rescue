import sys
from pathlib import Path
from datetime import datetime
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .app import SoundGuardApp

current_dir = Path(__file__).resolve().parent
beats_path = str(current_dir / "beats")
if beats_path not in sys.path:
    sys.path.insert(0, beats_path)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ [Server] 대시보드 연결 수락됨")

    guard_app = None

    try:
        print("🔄 [Server] AI 모델(BEATs, Whisper) 로딩 중...")
        guard_app = SoundGuardApp()
        paused = False

        custom_tts = {
            "INTRUSION_WARN_1": "",
            "INTRUSION_WARN_2": "",
            "EMERGENCY_GUIDE": "",
        }

        async def read_dashboard_commands():
            nonlocal paused, custom_tts

            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
            except asyncio.TimeoutError:
                return

            msg_type = msg.get("type")

            if msg_type == "pause":
                paused = bool(msg.get("paused", False))
                print(f"[DASHBOARD] 감지 {'일시정지' if paused else '재개'}")
                await websocket.send_json({
                    "type": "pause_state",
                    "paused": paused,
                })

            elif msg_type == "tts_config":
                custom_tts.update({
                    "INTRUSION_WARN_1": msg.get("w1", ""),
                    "INTRUSION_WARN_2": msg.get("w2", ""),
                    "EMERGENCY_GUIDE": msg.get("emg", ""),
                })
                print("[DASHBOARD] 안내 멘트 설정 반영 완료")
        print("🚀 [Server] 모델 로딩 완료. 분석 루프 시작")
        paused = False

        
        while True:
            await read_dashboard_commands()

            if paused:
                await websocket.send_json({
                    "type": "status",
                    "message": "paused",
                })
                await asyncio.sleep(0.2)
                continue
            await websocket.send_json({"type": "status", "message": "recording"})

            print(f"\n🎤 [{datetime.now().strftime('%H:%M:%S')}] {settings.chunk_seconds}초 녹음 및 분석 시작...")

            audio = guard_app._record_audio()
            audio_path = guard_app._save_audio(audio)

            sound_event = guard_app.env_classifier.classify(audio, settings.sample_rate)

            stt_text = ""
            stt_trigger = (
                sound_event.situation in {1, 2}
                or sound_event.rms >= settings.min_rms_for_stt
                or sound_event.peak >= settings.min_peak_for_stt
            )

            if guard_app.stt is not None and stt_trigger:
                if guard_app.stt.should_transcribe(audio):
                    stt_text = guard_app.stt.transcribe(audio_path)

            dwell_seconds = guard_app.dwell_tracker.update(sound_event, stt_text=stt_text)

            decision = guard_app.decision_engine.decide(
                sound_event=sound_event,
                stt_text=stt_text,
                dwell_seconds=dwell_seconds,
                authorized=False,
                
            )
            default_tts = guard_app.speaker.get_message(decision.tts_key)

            display_tts_message = custom_tts.get(decision.tts_key) or default_tts
            payload = {
                
                "tts_message": display_tts_message,
                "type": "analysis",
                "timestamp": datetime.now().strftime("%H:%M:%S"),

                "situation": decision.situation,
                "situation_name": decision.situation_name,
                "risk_level": decision.risk_level,
                "reason": decision.reason,
                "action": decision.action,
                "tts_key": decision.tts_key,

                "env_label": decision.situation_name,
                "beats_label": sound_event.label,
                "beats_raw_label": sound_event.raw_label,
                "beats_confidence": sound_event.confidence,
                "rms": sound_event.rms,
                "peak": sound_event.peak,

                "stt_text": stt_text,
                "dwell_seconds": dwell_seconds,

                "beats": {
                    "foot": 80 if sound_event.situation == 1 else 0,
                    "voice": 80 if stt_text else 0,
                    "scream": 80 if decision.situation == 2 else 0,
                    "env": 90 if decision.situation == 0 else 5,
                },
            }

            await websocket.send_json(payload)

            print(
                f"📡 [Server] 전송 완료: "
                f"BEATs={sound_event.raw_label}/{sound_event.label} | "
                f"Final={decision.situation_name} | STT={stt_text or '없음'}"
            )

            if decision.tts_key != "NONE":
                print(f"[TTS] {display_tts_message}")
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        print("🔌 [Server] 대시보드 연결 종료")
    except Exception as e:
        print(f"❌ [Server] 런타임 에러 발생: {e}")
    finally:
        print("🔌 [Server] WebSocket 세션 종료")