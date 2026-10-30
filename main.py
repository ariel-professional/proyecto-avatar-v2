"""Desktop avatar with speech recognition, JSON responses, and sprite animation."""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pygame
    from numpy.typing import NDArray

Dialogue = list[tuple[str, str]]


@dataclass(frozen=True)
class UIEvent:
    """An update sent from the assistant worker to the UI thread."""

    kind: str
    text: str = ""
    state: str = "idle"


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = Path(os.environ.get("AVATAR_RESOURCES_DIR") or BASE_DIR / "resources")
SAMPLE_RATE = 16000
BLOCK_SIZE = 512
FRAME_DURATION_MS = 40
SPRITE_SCALE = 8


def ensure_local_environment() -> None:
    """Prepare and select the local Windows environment, including IDE launches."""
    if sys.platform != "win32":
        return
    environment_python = BASE_DIR / ".venv/Scripts/python.exe"
    if not environment_python.is_file():
        print("[Startup] Preparing the avatar's Python environment...", flush=True)
        subprocess.check_call([sys.executable, "-m", "venv", str(BASE_DIR / ".venv")])
    dependency_status = subprocess.call(
        [
            str(environment_python),
            "-c",
            "import pygame, sounddevice, numpy, faster_whisper",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if dependency_status:
        print("[Startup] Installing required dependencies...", flush=True)
        subprocess.check_call(
            [
                str(environment_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(BASE_DIR / "requirements.txt"),
            ]
        )
    if Path(sys.executable).resolve() != environment_python.resolve():
        sys.exit(
            subprocess.call(
                [str(environment_python), str(BASE_DIR / "main.py"), *sys.argv[1:]]
            )
        )


def normalize_text(text: str) -> str:
    """Normalize case and accents for keyword matching."""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", text.casefold())
        if not unicodedata.combining(c)
    )


def load_dialogue() -> Dialogue:
    """Load keyword-response pairs using the existing JSON schema."""
    data = json.loads(
        (RESOURCES_DIR / "dialog/preguntas.json").read_text(encoding="utf-8")
    )
    # Preserve the legacy priority: the first matching keyword in sorted order.
    return sorted(
        {
            normalize_text(entry["clave"]): entry["respuesta"]
            for entry in data["preguntas"]
        }.items()
    )


def get_response(text: str, dialogue: Dialogue) -> str:
    """Return the first matching response or the Spanish fallback."""
    query_text = normalize_text(text)
    return next(
        (response for keyword, response in dialogue if keyword in query_text),
        "Disculpa, no entendí tu consulta.",
    )


def record_utterance(
    microphone: int | None, threshold: float, stop_event: threading.Event
) -> NDArray[np.float32] | None:
    """Capture a bounded utterance with pre-roll and trailing silence."""
    import numpy as np
    import sounddevice as sd

    chunks = []
    pre_roll = deque(
        maxlen=10
    )  # Keep about 320 ms before activation to preserve word beginnings.
    silent_samples = 0
    speech_samples = 0
    # Blocking reads stay in the worker and never block the UI thread.
    with sd.InputStream(
        device=microphone,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
    ) as stream:
        while not stop_event.is_set():
            block, overflow = stream.read(BLOCK_SIZE)
            if overflow:
                print(
                    "[Audio] Input overflow; restarting capture.",
                    flush=True,
                )
                chunks.clear()
                pre_roll.clear()
                speech_samples = silent_samples = 0
                continue
            is_active = float(np.max(np.abs(block))) >= threshold
            samples = block[:, 0].copy()
            if not chunks:
                pre_roll.append(samples)
                if not is_active:
                    continue
                chunks = list(pre_roll)
                pre_roll.clear()
            else:
                chunks.append(samples)
            if is_active:
                speech_samples += len(block)
                silent_samples = 0
            elif chunks:
                silent_samples += len(block)
            if chunks and (
                silent_samples >= SAMPLE_RATE * 1.1
                or len(chunks) * BLOCK_SIZE >= SAMPLE_RATE * 12
            ):
                if speech_samples >= SAMPLE_RATE * 0.2:
                    return np.concatenate(chunks)
                chunks.clear()
                pre_roll.clear()
                speech_samples = silent_samples = 0
    return None


def speak(text: str, stop_event: threading.Event) -> None:
    """Speak through the system synthesizer and allow shutdown to interrupt it."""
    if sys.platform == "win32":
        # Encode speech as data instead of interpolating it into shell syntax.
        encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
        script = """
        $ErrorActionPreference = 'Stop'
        Add-Type -AssemblyName System.Speech
        $voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
        try {
            $spanishVoice = $voice.GetInstalledVoices() | Where-Object {
                $_.Enabled -and $_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq 'es'
            } | Select-Object -First 1
            if ($spanishVoice) { $voice.SelectVoice($spanishVoice.VoiceInfo.Name) }
            else { Write-Warning 'No hay voz española instalada; se usa la predeterminada.' }
            $text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('ENCODED_TEXT'))
            $voice.Speak($text)
        } finally { $voice.Dispose() }
        """.replace("ENCODED_TEXT", encoded_text)
        encoded_script = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        command = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded_script,
        ]
    elif sys.platform == "darwin":
        command = ["say", text]
    elif shutil.which("spd-say"):
        command = ["spd-say", "-w", "-l", "es", "--", text]
    elif shutil.which("espeak-ng"):
        command = ["espeak-ng", "-v", "es", "--", text]
    else:
        raise RuntimeError(
            "Falta el sintetizador: instalá speech-dispatcher o espeak-ng."
        )

    options = (
        {"creationflags": subprocess.CREATE_NO_WINDOW}
        if sys.platform == "win32"
        else {}
    )
    with subprocess.Popen(command, **options) as process:
        started_at = time.monotonic()
        while process.poll() is None:
            if stop_event.wait(0.05) or time.monotonic() - started_at > 60:
                process.terminate()
                process.wait()
                return
        if process.returncode:
            raise RuntimeError(
                f"El sintetizador terminó con código {process.returncode}."
            )


def run_assistant(
    options: argparse.Namespace,
    dialogue: Dialogue,
    events: queue.Queue[UIEvent],
    queries: queue.Queue[str],
    stop_event: threading.Event,
) -> None:
    """Run capture, transcription, dialogue, and speech outside the UI thread."""

    def publish_status(animation: str, text: str) -> None:
        events.put(UIEvent("state", text, animation))
        print(f"[Status] {text}", flush=True)

    try:
        model = None
        if not options.text:
            publish_status(
                "idle",
                f"Cargando Whisper {options.model}. La primera vez se descarga el modelo...",
            )
            from faster_whisper import WhisperModel

            model = WhisperModel(options.model, device="cpu", compute_type="int8")
        while not stop_event.is_set():
            if options.text:
                publish_status("idle", "Escribí una consulta y presioná Enter.")
                events.put(UIEvent("ready"))
                while not stop_event.is_set():
                    try:
                        text = queries.get(timeout=0.1)
                        break
                    except queue.Empty:
                        pass
                else:
                    break
            else:
                publish_status("listening", "Te escucho. Hablá y luego dejá una pausa.")
                audio = record_utterance(
                    options.microphone, options.threshold, stop_event
                )
                if audio is None or stop_event.is_set():
                    break
                publish_status("idle", "Transcribiendo...")
                assert model is not None
                segments, _ = model.transcribe(
                    audio,
                    language="es",
                    beam_size=5,
                    temperature=0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters={
                        "min_speech_duration_ms": 200,
                        "min_silence_duration_ms": 500,
                        "speech_pad_ms": 200,
                    },
                )
                text = "".join(segment.text for segment in segments).strip()
                if not text or "[" in text or "(" in text:
                    continue
            if stop_event.is_set():
                break
            response = get_response(text, dialogue)
            print(f"[User] {text}\n[Assistant] {response}", flush=True)
            events.put(UIEvent("user", text))
            events.put(UIEvent("response", response))
            if not options.mute:
                publish_status("speaking", "Respondiendo...")
                try:
                    speak(response, stop_event)
                except (OSError, RuntimeError) as error:
                    events.put(UIEvent("warning", str(error)))
                    print(f"[Speech] {error}", flush=True)
            # Keep the microphone closed during transcription and speech playback.
    except ModuleNotFoundError as error:
        message = (
            f"Falta {error.name} en el Python usado para abrir la app. "
            "En Windows, cerrá y ejecutá iniciar.cmd. "
            "Si sigue fallando, instalá requirements.txt en .venv."
        )
        events.put(UIEvent("error", message))
        print(
            f"[Error] {message}\nIntérprete: {sys.executable}",
            file=sys.stderr,
            flush=True,
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        events.put(UIEvent("error", message))
        print(f"[Error] {message}", file=sys.stderr, flush=True)


def load_animations() -> dict[str, list[pygame.Surface]]:
    """Read sprite rectangles and pre-scale each animation frame."""
    import pygame

    animations = {}
    for state, name in [
        ("idle", "AnimacionReposo"),
        ("listening", "escuchando"),
        ("speaking", "hablando"),
    ]:
        base = RESOURCES_DIR / "animations" / name
        sheet = pygame.image.load(str(base.with_suffix(".png"))).convert_alpha()
        data = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        frames = []
        for item in data["frames"].values():
            frame_data = item["frame"]
            crop = sheet.subsurface(
                (frame_data["x"], frame_data["y"], frame_data["w"], frame_data["h"])
            )
            frames.append(
                pygame.transform.scale(
                    crop,
                    (frame_data["w"] * SPRITE_SCALE, frame_data["h"] * SPRITE_SCALE),
                )
            )
        animations[state] = frames
    return animations


def draw_text(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    y: int,
    color: tuple[int, int, int] = (45, 45, 55),
    max_lines: int = 3,
) -> int:
    """Draw wrapped text and return the next available vertical position."""
    lines = [""]
    for word in text.split():
        candidate = (lines[-1] + " " + word).strip()
        if font.size(candidate)[0] > 740 and lines[-1]:
            lines.append(word)
        else:
            lines[-1] = candidate
    for line in lines[:max_lines]:
        screen.blit(font.render(line, True, color), (30, y))
        y += font.get_linesize()
    return y


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options while preserving the original Spanish aliases."""
    parser = argparse.ArgumentParser(
        description="Avatar virtual: voz, respuestas y animaciones del legado."
    )
    parser.add_argument(
        "--text",
        "--texto",
        action="store_true",
        help="Consultar con teclado, sin micrófono ni modelo.",
    )
    parser.add_argument(
        "--mute",
        "--sin-voz",
        action="store_true",
        help="Mostrar las respuestas sin sintetizar audio.",
    )
    parser.add_argument(
        "--model",
        "--modelo",
        default="small",
        help="Modelo faster-whisper o carpeta local (default: small).",
    )
    parser.add_argument(
        "--microphone",
        "--microfono",
        type=int,
        help="Número de dispositivo; por defecto el del sistema.",
    )
    parser.add_argument(
        "--list-microphones",
        "--listar-microfonos",
        action="store_true",
        help="Listar dispositivos y salir.",
    )
    parser.add_argument(
        "--threshold",
        "--umbral",
        type=float,
        default=0.03,
        help="Sensibilidad de entrada (default: 0.03).",
    )
    options = parser.parse_args()
    if not 0 < options.threshold <= 1:
        parser.error("--umbral debe estar entre 0 (excluido) y 1.")
    return options


def main() -> None:
    """Run the pygame event loop and keep all drawing on the main thread."""
    options = parse_arguments()
    if options.list_microphones:
        import sounddevice as sd

        print(sd.query_devices())
        return

    import pygame

    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Asistente Virtual - Python")
    stop_event = threading.Event()
    worker = None
    try:
        animations = load_animations()
        dialogue = load_dialogue()
        events: queue.Queue[UIEvent] = queue.Queue()
        queries: queue.Queue[str] = queue.Queue(maxsize=1)
        worker = threading.Thread(
            target=run_assistant,
            args=(options, dialogue, events, queries, stop_event),
            daemon=True,
        )
        worker.start()
        font = pygame.font.SysFont("segoeui", 20)
        small_font = pygame.font.SysFont("segoeui", 17)
        clock = pygame.time.Clock()
        state, message = "idle", "Iniciando..."
        user_text = response = input_text = warning = ""
        animation_started_at = pygame.time.get_ticks()
        running = True
        busy = True
        pygame.key.start_text_input()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                elif options.text and event.type == pygame.TEXTINPUT:
                    input_text = (input_text + event.text)[:250]
                elif options.text and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif (
                        event.key == pygame.K_RETURN and input_text.strip() and not busy
                    ):
                        queries.put_nowait(input_text.strip())
                        input_text = warning = ""
                        busy = True
            while True:
                try:
                    update = events.get_nowait()
                except queue.Empty:
                    break
                if update.kind == "state":
                    new_state, message = update.state, update.text
                    if new_state != state:
                        animation_started_at = pygame.time.get_ticks()
                    state = new_state
                    busy = True
                elif update.kind == "ready":
                    busy = False
                elif update.kind == "user":
                    user_text = update.text
                elif update.kind == "response":
                    response = update.text
                elif update.kind == "warning":
                    warning = update.text
                elif update.kind == "error":
                    state, message, warning = (
                        "idle",
                        "No se pudo continuar. Revisá la consola.",
                        update.text,
                    )
                    busy = True

            screen.fill((255, 255, 255))
            draw_text(screen, small_font, message, 20, max_lines=2)
            frames = animations[state]
            frame = frames[
                ((pygame.time.get_ticks() - animation_started_at) // FRAME_DURATION_MS)
                % len(frames)
            ]
            screen.blit(frame, frame.get_rect(center=(400, 235)))
            draw_text(screen, font, f"Vos: {user_text}", 380, max_lines=2)
            draw_text(screen, font, f"Avatar: {response}", 435, max_lines=2)
            if options.text:
                draw_text(screen, small_font, "> " + input_text[-85:], 505, max_lines=2)
            draw_text(
                screen,
                small_font,
                warning or "Esc: cerrar",
                560,
                color=(170, 45, 45) if warning else (110, 110, 110),
                max_lines=1,
            )
            pygame.display.flip()
            clock.tick(60)
    finally:
        stop_event.set()
        pygame.quit()
        if worker is not None:
            worker.join(timeout=1)


if __name__ == "__main__":
    try:
        ensure_local_environment()
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"[Error] {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
