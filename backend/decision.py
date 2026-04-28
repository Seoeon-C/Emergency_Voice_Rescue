from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, Any, Optional

from openai import OpenAI

from config import settings
from environmental_sound import SoundEvent


@dataclass
class DecisionResult:
    situation: int
    situation_name: str
    risk_level: str
    reason: str
    action: str
    tts_key: str
    send_to_control_room: bool
    emergency_candidate: bool
    raw_gpt: str = ""


GPT_SYSTEM_PROMPT = """
너는 음향 기반 위험구역 안전관리 시스템의 판단 모듈이다.

상황 정의:
상황 0: 이상없음
상황 1: 무단침입
상황 2: 위험 감지

판단 기준:
- 허가 사용자 인증 상태면 상황 0으로 판단한다.
- 발소리 또는 사람 말소리가 감지되고 위험구역 체류시간이 5초 이상이면 상황 1의 1차 대응이다.
- 발소리 또는 사람 말소리가 감지되고 위험구역 체류시간이 15초 이상이면 상황 1의 2차 대응이다.
- STT 텍스트가 있고 침입 의도 또는 사람 존재를 암시하면 상황 1로 볼 수 있다.
- 비명, 충격음, 유리 깨짐, 구조 요청 문장이 감지되면 상황 2다.
- STT 텍스트가 "시청해주셔서 감사합니다", "구독해주세요" 같은 자막형 문구이고 환경음 신뢰도가 낮으면 위험 판단에 사용하지 않는다.

응답 형식:
반드시 JSON만 출력한다. 마크다운 코드블록을 쓰지 마라.

JSON 필드:
{
  "situation": 0 또는 1 또는 2,
  "situation_name": "이상없음" 또는 "무단침입" 또는 "위험 감지",
  "risk_level": "low" 또는 "medium" 또는 "high",
  "reason": "판단 이유",
  "action": "수행할 대응",
  "tts_key": "NONE" 또는 "INTRUSION_WARN_1" 또는 "INTRUSION_WARN_2" 또는 "EMERGENCY_GUIDE" 또는 "EVACUATION_GUIDE",
  "send_to_control_room": true 또는 false,
  "emergency_candidate": true 또는 false
}
"""


class GPTDecisionEngine:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY가 .env에 필요합니다.")

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_llm_model

    def decide(
        self,
        sound_event: SoundEvent,
        stt_text: str,
        dwell_seconds: float,
        authorized: bool,
    ) -> DecisionResult:
        rule_result = self._rule_based_decision(
            sound_event=sound_event,
            stt_text=stt_text,
            dwell_seconds=dwell_seconds,
            authorized=authorized,
        )

        payload = {
            "zone_name": settings.zone_name,
            "location_text": settings.location_text,
            "latitude": settings.latitude,
            "longitude": settings.longitude,
            "sound_event": asdict(sound_event),
            "stt_text": stt_text,
            "dwell_seconds": round(dwell_seconds, 2),
            "authorized": authorized,
            "rule_based_result": rule_result,
        }

        gpt_raw = self._ask_gpt(payload)
        gpt_result = self._parse_json(gpt_raw)

        final = gpt_result or rule_result

        if rule_result["situation"] == 2 and final.get("situation") == 0:
            final = rule_result

        return DecisionResult(
            situation=int(final.get("situation", rule_result["situation"])),
            situation_name=str(final.get("situation_name", rule_result["situation_name"])),
            risk_level=str(final.get("risk_level", rule_result["risk_level"])),
            reason=str(final.get("reason", rule_result["reason"])),
            action=str(final.get("action", rule_result["action"])),
            tts_key=str(final.get("tts_key", rule_result["tts_key"])),
            send_to_control_room=bool(final.get("send_to_control_room", rule_result["send_to_control_room"])),
            emergency_candidate=bool(final.get("emergency_candidate", rule_result["emergency_candidate"])),
            raw_gpt=gpt_raw,
        )

    def _rule_based_decision(
        self,
        sound_event: SoundEvent,
        stt_text: str,
        dwell_seconds: float,
        authorized: bool,
    ) -> Dict[str, Any]:
        if authorized:
            return {
                "situation": 0,
                "situation_name": "이상없음",
                "risk_level": "low",
                "reason": "허가된 사용자 인증으로 감지 로직 일시 정지",
                "action": "감지 로직 비활성 상태 유지",
                "tts_key": "NONE",
                "send_to_control_room": False,
                "emergency_candidate": False,
            }

        normalized_text = (stt_text or "").replace(" ", "")

        emergency_keywords = [
            "도와주세요", "살려주세요", "119", "구조", "불났", "불이야", "쓰러졌",
            "다쳤", "피나요", "갇혔", "위험해", "큰일났", "사람이쓰러", "구해주세요",
        ]

        if sound_event.danger_sound_detected or any(k in normalized_text for k in emergency_keywords):
            return {
                "situation": 2,
                "situation_name": "위험 감지",
                "risk_level": "high",
                "reason": "위험음 또는 응급구조신호 감지",
                "action": "상황실 긴급 알림 전송 및 현장 대피/응급 안내 방송",
                "tts_key": "EMERGENCY_GUIDE",
                "send_to_control_room": True,
                "emergency_candidate": True,
            }

        if sound_event.person_detected or sound_event.label in {"footstep", "speech"} or bool(stt_text):
            if dwell_seconds >= 15:
                return {
                    "situation": 1,
                    "situation_name": "무단침입",
                    "risk_level": "medium",
                    "reason": "위험구역 내 사람 존재 추정 및 15초 이상 체류",
                    "action": "2차 경고 방송, 이벤트 기록, 상황실 위치정보 전송",
                    "tts_key": "INTRUSION_WARN_2",
                    "send_to_control_room": True,
                    "emergency_candidate": False,
                }

            if dwell_seconds >= 5 or bool(stt_text):
                return {
                    "situation": 1,
                    "situation_name": "무단침입",
                    "risk_level": "low",
                    "reason": "사람 음성 또는 위험구역 체류 조건 감지",
                    "action": "1차 경고 방송 및 이벤트 기록",
                    "tts_key": "INTRUSION_WARN_1",
                    "send_to_control_room": True,
                    "emergency_candidate": False,
                }

        return {
            "situation": 0,
            "situation_name": "이상없음",
            "risk_level": "low",
            "reason": "사람 존재, 체류시간, 위험음, 응급구조신호 조건 미충족",
            "action": "감시 지속",
            "tts_key": "NONE",
            "send_to_control_room": False,
            "emergency_candidate": False,
        }

    def _ask_gpt(self, payload: Dict[str, Any]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": GPT_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"[WARN] GPT 판단 실패. rule 기반 판단을 사용합니다: {exc}")
            return ""

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            return json.loads(cleaned)
        except Exception:
            return None
