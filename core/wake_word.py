"""Local wake-word detection ("Hey JARVIS") using openWakeWord.

Runs entirely on-device — no audio leaves the machine until the wake
word is heard. Optional: if openwakeword isn't installed or its model
files can't be loaded, the caller should treat the assistant as
always-awake (see main.py's fallback).
"""

import sys
import time
from pathlib import Path

WAKE_MODEL_NAME = "hey_jarvis"   # openWakeWord's stock "Hey JARVIS" model
WAKE_THRESHOLD  = 0.5

try:
    import openwakeword
    from openwakeword.model import Model
    from openwakeword.utils import download_models, download_file
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def available() -> bool:
    return _AVAILABLE


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


# openWakeWord defaults to downloading into its own site-packages folder,
# which is read-only for a non-admin user (e.g. under Program Files on
# Windows). Use a writable, project-local folder instead.
_MODELS_DIR = _base_dir() / "config" / "wake_models"


def _download_with_retries(model_name: str, attempts: int = 5) -> None:
    """GitHub's release CDN occasionally resets mid-download. Already-fetched
    files are skipped on retry, so each attempt only needs to finish the rest."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            download_models(model_names=[model_name], target_directory=str(_MODELS_DIR))
            return
        except Exception as e:
            last_error = e
            print(f"[WakeWord] Download attempt {attempt}/{attempts} failed: {e}")
            if attempt < attempts:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download wake word models after {attempts} attempts: {last_error}")


def _ensure_onnx_files(model_name: str, attempts: int = 5) -> None:
    """openWakeWord's download_models() only checks the .tflite file before
    skipping a model — if a prior run got the .tflite but died mid-.onnx,
    the .onnx never gets retried. Fetch any missing .onnx directly."""
    tflite_urls = [m["download_url"] for m in openwakeword.FEATURE_MODELS.values()]
    tflite_urls += [m["download_url"] for k, m in openwakeword.MODELS.items() if k == model_name]

    for tflite_url in tflite_urls:
        onnx_url  = tflite_url.replace(".tflite", ".onnx")
        onnx_path = _MODELS_DIR / onnx_url.split("/")[-1]
        if onnx_path.exists():
            continue
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                download_file(onnx_url, str(_MODELS_DIR))
                last_error = None
                break
            except Exception as e:
                last_error = e
                print(f"[WakeWord] {onnx_path.name} attempt {attempt}/{attempts} failed: {e}")
                if attempt < attempts:
                    time.sleep(2 * attempt)
        if last_error:
            raise RuntimeError(f"Could not download {onnx_path.name}: {last_error}")


class WakeWordDetector:
    """Thin wrapper around openWakeWord: load once, predict per audio chunk."""

    def __init__(self, model_name: str = WAKE_MODEL_NAME, threshold: float = WAKE_THRESHOLD):
        if not _AVAILABLE:
            raise RuntimeError(
                "openwakeword not installed. Run: pip install openwakeword"
            )
        self.threshold = threshold

        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _download_with_retries(model_name)
        _ensure_onnx_files(model_name)

        wakeword_path = next(_MODELS_DIR.glob(f"{model_name}*.onnx"), None)
        if not wakeword_path:
            raise RuntimeError(f"Wake word model '{model_name}' not found after download.")

        self.model_key = wakeword_path.stem  # matches the key Model.predict() returns

        self.model = Model(
            wakeword_models=[str(wakeword_path)],
            inference_framework="onnx",
            melspec_model_path=str(_MODELS_DIR / "melspectrogram.onnx"),
            embedding_model_path=str(_MODELS_DIR / "embedding_model.onnx"),
        )

    def predict(self, pcm_chunk) -> tuple[str, float] | None:
        """Feed one int16 mono PCM chunk. Returns (model_name, score) if the
        wake word crossed the threshold this call, else None."""
        predictions = self.model.predict(pcm_chunk)
        score = predictions.get(self.model_key, 0.0)
        if score >= self.threshold:
            self.model.reset()
            return self.model_key, score
        return None
