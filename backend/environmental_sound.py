from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch

from config import settings


@dataclass
class SoundEvent:
    label: str
    confidence: float
    raw_label: str
    person_detected: bool
    danger_sound_detected: bool
    rms: float = 0.0
    peak: float = 0.0


def get_audio_stats(audio: np.ndarray) -> tuple[float, float]:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    return rms, peak


class BeatsEnvironmentClassifier:
    def __init__(self) -> None:
        self.sample_rate = settings.sample_rate
        self.device = torch.device(settings.device)
        self.model = None
        self.labels: List[str] = []
        self.ready = False

        try:
            self._load_beats()
            self.ready = True
            print("[BEATs] 모델 로드 성공")
        except Exception as exc:
            print(f"[WARN] BEATs 모델 로드 실패. fallback 모드로 실행합니다: {exc}")

    def _load_beats(self) -> None:
        beats_py = Path(settings.beats_py_dir) / "BEATs.py"
        checkpoint_path = Path(settings.beats_checkpoint_path)

        if not beats_py.exists():
            raise FileNotFoundError(f"BEATs.py 파일을 찾을 수 없습니다: {beats_py}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"BEATs 체크포인트를 찾을 수 없습니다: {checkpoint_path}")

        spec = importlib.util.spec_from_file_location("beats_module", str(beats_py))
        if spec is None or spec.loader is None:
            raise RuntimeError("BEATs.py import 준비 실패")

        module = importlib.util.module_from_spec(spec)
        sys.modules["beats_module"] = module
        spec.loader.exec_module(module)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = module.BEATsConfig(checkpoint["cfg"])

        model = module.BEATs(cfg)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model.to(self.device)

        self.model = model
        self.labels = checkpoint.get("label_names", [])

    def classify(self, audio: np.ndarray, sr: int) -> SoundEvent:
        rms, peak = get_audio_stats(audio)
        audio = self._prepare_audio(audio, sr)

        if audio.size == 0:
            return SoundEvent("background", 0.0, "empty", False, False, rms, peak)

        if not self.ready:
            return self._fallback_classify(audio, rms, peak)

        with torch.no_grad():
            wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
            padding_mask = torch.zeros(wav.shape, dtype=torch.bool).to(self.device)

            result = self.model.extract_features(wav, padding_mask=padding_mask)[0]

            if result.ndim == 3:
                logits = result.mean(dim=1)
            else:
                logits = result

            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])

        raw_label = self.labels[idx] if self.labels and idx < len(self.labels) else f"class_{idx}"
        label = self._map_raw_label(raw_label)

        if confidence < 0.01 and rms < settings.min_rms_for_stt and peak < settings.min_peak_for_stt:
            label = "background"

        return SoundEvent(
            label=label,
            confidence=confidence,
            raw_label=raw_label,
            person_detected=label in {"footstep", "speech"},
            danger_sound_detected=label in {"scream", "impact", "glass_break"},
            rms=rms,
            peak=peak,
        )

    def _prepare_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak

        return audio.astype(np.float32)

    def _map_raw_label(self, raw_label: str) -> str:
        text = raw_label.lower()

        if any(k in text for k in ["footstep", "walk", "walking", "steps"]):
            return "footstep"
        if any(k in text for k in ["speech", "talk", "conversation", "human voice", "male speech", "female speech"]):
            return "speech"
        if any(k in text for k in ["scream", "screaming", "yell", "shout", "cry"]):
            return "scream"
        if any(k in text for k in ["crash", "bang", "thump", "slam", "impact", "explosion"]):
            return "impact"
        if any(k in text for k in ["glass", "shatter", "breaking"]):
            return "glass_break"
        if any(k in text for k in ["silence", "ambient", "background"]):
            return "background"

        return "unknown"

    def _fallback_classify(self, audio: np.ndarray, rms: float, peak: float) -> SoundEvent:
        if rms < settings.min_rms_for_stt and peak < settings.min_peak_for_stt:
            return SoundEvent("background", 0.8, "fallback_silence", False, False, rms, peak)

        # fallback에서는 정확히 speech를 모른다. 그래서 unknown으로 두고 main에서 음량 기준 STT를 시도한다.
        return SoundEvent("unknown", min(0.95, rms * 10), "fallback_audio_detected", False, False, rms, peak)
