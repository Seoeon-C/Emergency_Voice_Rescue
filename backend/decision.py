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
너는 위험구역 음향 감지 시스템의 판단 모듈이다.

1차 분류:
- nature: 자연/배경 소리. pass.
- speech: 사람 말소리. STT 텍스트로 무단침입/위급 분류.
- footstep: 발소리. 무단침입.
- emergency_sound: 비명/충격음/파손음. 위급.
- unknown: STT 텍스트가 있으면 speech 후보로 판단한다. STT 텍스트가 없으면 pass.

상황:
0 이상없음
1 무단침입
2 위험 감지

규칙:
1. nature는 상황 0.
2. footstep은 상황 1.
3. emergency_sound는 상황 2.
4. speech 또는 unknown+STT에서 구조 요청/사고/부상/화재/갇힘/쓰러짐/아픔 표현이면 상황 2.
5. speech 또는 unknown+STT에서 일반 대화, 침입 의도, 탐색, 인기척은 상황 1.
6. unknown + STT 없음은 상황 0.
7. 실제 구조대 자동 신고가 아니라 상황실 확인 대상으로 표시한다.

무단침입 단계:
- 체류시간 15초 이상: INTRUSION_WARN_2
- 그 외: INTRUSION_WARN_1

반드시 JSON만 출력한다.
필드:
situation, situation_name, risk_level, reason, action, tts_key, send_to_control_room, emergency_candidate
"""


class GPTDecisionEngine:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY가 .env에 필요합니다.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_llm_model

    def decide(self, sound_event: SoundEvent, stt_text: str, dwell_seconds: float, authorized: bool) -> DecisionResult:
        rule_result = self._rule_based_decision(sound_event, stt_text, dwell_seconds, authorized)

        if self._is_strong_rule(sound_event, stt_text):
            return self._to_result(rule_result, "")

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
        final = self._validate_decision(gpt_result or rule_result, sound_event, stt_text, rule_result)
        return self._to_result(final, gpt_raw)

    def _rule_based_decision(self, sound_event: SoundEvent, stt_text: str, dwell_seconds: float, authorized: bool) -> Dict[str, Any]:
        if authorized:
            return self._normal("허가된 사용자 인증으로 감지 로직 일시 정지")

        if sound_event.label == "nature":
            return self._normal("자연/배경 소리로 판단하여 pass")

        if sound_event.label == "footstep":
            return self._intrusion(dwell_seconds, "발소리 감지로 무단침입 판단")

        if sound_event.label == "emergency_sound":
            return self._emergency("비명/충격음/파손음 등 위험음 감지")

        text = (stt_text or "").replace(" ", "")

        emergency_keywords = [
            "아파", "아파요", "도와", "살려", "119", "구조", "불났", "불이야",
            "쓰러", "다쳤", "피", "갇혔", "위험", "큰일", "죽겠", "사고", "넘어졌",
        ]

        intrusion_keywords = [
            "들어가", "가보자", "몰래", "넘어가", "열어", "문열",
            "누구없", "안에", "들어왔", "사람있", "뭐야", "여기",
        ]

        if sound_event.label in {"speech", "unknown"}:
            if not text:
                if sound_event.label == "speech":
                    return self._normal("말소리로 분류되었지만 유효한 STT 텍스트가 없음")
                return self._normal("unknown이며 유효한 STT 텍스트가 없어 pass")

            if any(k in text for k in emergency_keywords):
                return self._emergency("STT에서 응급/위험 표현 감지")

            if any(k in text for k in intrusion_keywords):
                return self._intrusion(dwell_seconds, "STT에서 위험구역 진입/탐색 의도 감지")

            return self._intrusion(dwell_seconds, "사람 음성 텍스트 감지")

        return self._normal("판단 조건 미충족")

    def _normal(self, reason: str) -> Dict[str, Any]:
        return {
            "situation": 0,
            "situation_name": "이상없음",
            "risk_level": "low",
            "reason": reason,
            "action": "감시 지속",
            "tts_key": "NONE",
            "send_to_control_room": False,
            "emergency_candidate": False,
        }

    def _intrusion(self, dwell_seconds: float, reason: str) -> Dict[str, Any]:
        if dwell_seconds >= settings.intrusion_warn_2_seconds:
            return {
                "situation": 1,
                "situation_name": "무단침입",
                "risk_level": "medium",
                "reason": f"{reason}; 체류시간 {dwell_seconds:.1f}초로 2차 경고 기준 충족",
                "action": "2차 경고 방송, 이벤트 기록, 상황실 위치정보 전송",
                "tts_key": "INTRUSION_WARN_2",
                "send_to_control_room": True,
                "emergency_candidate": False,
            }

        return {
            "situation": 1,
            "situation_name": "무단침입",
            "risk_level": "low",
            "reason": f"{reason}; 1차 경고 대상",
            "action": "1차 경고 방송 및 이벤트 기록",
            "tts_key": "INTRUSION_WARN_1",
            "send_to_control_room": True,
            "emergency_candidate": False,
        }

    def _emergency(self, reason: str) -> Dict[str, Any]:
        return {
            "situation": 2,
            "situation_name": "위험 감지",
            "risk_level": "high",
            "reason": reason,
            "action": "상황실 긴급 문자/알림 전송 및 현장 응급 안내",
            "tts_key": "EMERGENCY_GUIDE",
            "send_to_control_room": True,
            "emergency_candidate": True,
        }

    def _is_strong_rule(self, sound_event: SoundEvent, stt_text: str) -> bool:
        if sound_event.label in {"nature", "footstep", "emergency_sound"}:
            return True
        text = (stt_text or "").replace(" ", "")
        if any(k in text for k in ["아파", "도와", "살려", "119", "불났", "쓰러", "다쳤", "갇혔"]):
            return True
        if sound_event.label == "speech" and not text:
            return True
        if sound_event.label == "unknown" and not text:
            return True
        return False

    def _validate_decision(self, final: Dict[str, Any], sound_event: SoundEvent, stt_text: str, rule_result: Dict[str, Any]) -> Dict[str, Any]:
        if sound_event.label in {"nature", "footstep", "emergency_sound"}:
            return rule_result
        if sound_event.label == "unknown" and not stt_text:
            return rule_result
        return final

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

    def _to_result(self, data: Dict[str, Any], raw_gpt: str) -> DecisionResult:
        return DecisionResult(
            situation=int(data.get("situation", 0)),
            situation_name=str(data.get("situation_name", "이상없음")),
            risk_level=str(data.get("risk_level", "low")),
            reason=str(data.get("reason", "")),
            action=str(data.get("action", "")),
            tts_key=str(data.get("tts_key", "NONE")),
            send_to_control_room=bool(data.get("send_to_control_room", False)),
            emergency_candidate=bool(data.get("emergency_candidate", False)),
            raw_gpt=raw_gpt,
        )
