# tts.py
import base64, tempfile, os
try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except Exception:
    COQUI_AVAILABLE = False
from gtts import gTTS

COQUI_MODEL = os.getenv("COQUI_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")

def synthesize(text: str, lang: str = "en", engine: str = "auto") -> str:
    """
    returns base64-encoded mp3 bytes
    engine: auto | coqui | gtts
    """
    if engine == "coqui" or (engine=="auto" and COQUI_AVAILABLE):
        try:
            tts = CoquiTTS(COQUI_MODEL)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.tts_to_file(text=text, file_path=tmp.name)
            with open(tmp.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            os.unlink(tmp.name)
            return b64
        except Exception:
            pass
    # fallback gTTS
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    gTTS(text, lang=lang).save(tmp.name)
    with open(tmp.name, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp.name)
    return b64
