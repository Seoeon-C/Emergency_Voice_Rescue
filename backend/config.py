\
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # API
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_stt_model: str = os.getenv("OPENAI_STT_MODEL", "whisper-1")

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    # BEATs
    beats_py_dir: str = os.getenv("BEATS_PY_DIR", "beats")
    beats_checkpoint_path: str = os.getenv(
        "BEATS_CHECKPOINT_PATH",
        "checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    )

    # Runtime
    device: str = os.getenv("DEVICE", "cpu")
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    chunk_seconds: int = int(os.getenv("CHUNK_SECONDS", "5"))

    # Location
    zone_name: str = os.getenv("ZONE_NAME", "위험구역 A")
    location_text: str = os.getenv("LOCATION_TEXT", "폐공사장 A구역 입구")
    latitude: str = os.getenv("LATITUDE", "37.000000")
    longitude: str = os.getenv("LONGITUDE", "127.000000")

    # Control room
    control_room_webhook: str = os.getenv("CONTROL_ROOM_WEBHOOK", "")

    # Auth
    auth_password: str = os.getenv("AUTH_PASSWORD", "1234")
    auth_disable_seconds: int = int(os.getenv("AUTH_DISABLE_SECONDS", "10"))


settings = Settings()
