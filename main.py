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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pygame
    from faster_whisper import WhisperModel
    from numpy.typing import NDArray

Dialogue = list[tuple[str, str]]


@dataclass(frozen=True)
class UIEvent:
    """An update sent from the assistant worker to the UI thread."""

    kind: str
    text: str = ""
    state: str = "idle"


@dataclass
class VoiceControls:
    """Share capture controls between the UI and audio worker."""

    push_to_talk: threading.Event = field(default_factory=threading.Event)
    held: threading.Event = field(default_factory=threading.Event)
    mode_changed: threading.Event = field(default_factory=threading.Event)
    ready: threading.Event = field(default_factory=threading.Event)


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = Path(os.environ.get("AVATAR_RESOURCES_DIR") or BASE_DIR / "resources")
SAMPLE_RATE = 16000
BLOCK_SIZE = 512
TRAILING_SILENCE_SECONDS = 0.7
PARTIAL_TRANSCRIPTION_SECONDS = 1.0
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


def find_response(text: str, dialogue: Dialogue) -> str | None:
    """Return a matching response without treating the fallback as a match."""
    query_text = normalize_text(text)
    return next(
        (response for keyword, response in dialogue if keyword in query_text),
        None,
    )


def get_response(text: str, dialogue: Dialogue) -> str:
    """Return the first matching response or the Spanish fallback."""
    response = find_response(text, dialogue)
    return response if response is not None else "Disculpa, no entendí tu consulta."


def transcribe_audio(model: WhisperModel, audio: NDArray[np.float32]) -> str:
    """Use the same recognition settings for partial and final audio."""
    segments, _ = model.transcribe(
        audio,
        language="es",
        beam_size=1,
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
    return text if "[" not in text and "(" not in text else ""


class LiveTranscriber:
    """Recognize snapshots serially and stop as soon as a response is found."""

    def __init__(
        self,
        model: WhisperModel,
        dialogue: Dialogue,
        events: queue.Queue[UIEvent],
        stop_event: threading.Event,
    ) -> None:
        self.model = model
        self.dialogue = dialogue
        self.events = events
        self.stop_event = stop_event
        self.cancelled = threading.Event()
        self.matched = threading.Event()
        self.finished = threading.Event()
        self.result: tuple[str, str] | None = None
        self.error: Exception | None = None
        self._pending: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue(
            maxsize=1
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, audio: NDArray[np.float32], final: bool = False) -> None:
        if self.finished.is_set() or self.cancelled.is_set() or self.matched.is_set():
            return
        # Keep only the newest snapshot when inference is slower than capture.
        try:
            self._pending.get_nowait()
        except queue.Empty:
            pass
        self._pending.put_nowait((audio, final))

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set() and not self.cancelled.is_set():
                try:
                    audio, final = self._pending.get(timeout=0.05)
                except queue.Empty:
                    continue
                text = transcribe_audio(self.model, audio)
                if self.stop_event.is_set() or self.cancelled.is_set():
                    return
                if text:
                    self.events.put(UIEvent("partial", text))
                    response = find_response(text, self.dialogue)
                    if response is not None:
                        self.result = (text, response)
                        self.matched.set()
                        self.events.put(UIEvent("matched"))
                        return
                if final:
                    if text:
                        self.result = (text, get_response(text, self.dialogue))
                    return
        except Exception as error:
            self.error = error
        finally:
            self.finished.set()

    def finish(
        self, audio: NDArray[np.float32] | None, mode_changed: threading.Event
    ) -> tuple[str, str] | None:
        if audio is not None:
            self.submit(audio, final=True)
        while not self.finished.wait(0.05):
            if self.stop_event.is_set() or mode_changed.is_set():
                return None
        if self.error is not None:
            raise self.error
        return self.result

    def close(self) -> None:
        self.cancelled.set()
        # shortcut: wait for native inference on mode changes; use a cancellable backend if needed.
        while self._thread.is_alive() and not self.stop_event.is_set():
            self._thread.join(timeout=0.05)
        try:
            self._pending.get_nowait()
        except queue.Empty:
            pass


def record_utterance(
    microphone: int | None,
    threshold: float,
    stop_event: threading.Event,
    controls: VoiceControls,
    recognizer: LiveTranscriber,
) -> NDArray[np.float32] | None:
    """Capture until silence or key release while publishing audio snapshots."""
    import numpy as np
    import sounddevice as sd

    chunks = []
    pre_roll = deque(
        maxlen=10
    )  # Keep about 320 ms before activation to preserve word beginnings.
    silent_samples = 0
    speech_samples = 0
    total_samples = 0
    next_partial = int(SAMPLE_RATE * PARTIAL_TRANSCRIPTION_SECONDS)
    manual = controls.push_to_talk.is_set()
    # Blocking reads stay in the worker and never block the UI thread.
    with sd.InputStream(
        device=microphone,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
    ) as stream:
        while not stop_event.is_set():
            if controls.mode_changed.is_set():
                return None
            if recognizer.error is not None:
                raise recognizer.error
            block, overflow = stream.read(BLOCK_SIZE)
            if overflow:
                if chunks or total_samples:
                    recognizer.cancelled.set()
                    recognizer.events.put(
                        UIEvent("warning", "Se perdió audio; repetí la consulta.")
                    )
                    return None
                print(
                    "[Audio] Input overflow; restarting capture.",
                    flush=True,
                )
                chunks.clear()
                pre_roll.clear()
                speech_samples = silent_samples = 0
                continue
            is_active = float(np.max(np.abs(block))) >= threshold
            if manual and not controls.held.is_set():
                if total_samples:
                    break
                continue
            samples = block[:, 0].copy()
            if not total_samples:
                if manual:
                    chunks.append(samples)
                    recognizer.events.put(
                        UIEvent(
                            "state",
                            "Escuchando; soltá Espacio para responder.",
                            "listening",
                        )
                    )
                else:
                    pre_roll.append(samples)
                    if not is_active:
                        continue
                    chunks = list(pre_roll)
                    pre_roll.clear()
                total_samples = len(chunks) * BLOCK_SIZE
            else:
                total_samples += len(block)
                if not recognizer.matched.is_set():
                    chunks.append(samples)
            if is_active:
                speech_samples += len(block)
                silent_samples = 0
            else:
                silent_samples += len(block)
            if (
                not recognizer.matched.is_set()
                and total_samples >= next_partial
                and (manual or speech_samples >= SAMPLE_RATE * 0.2)
            ):
                recognizer.submit(np.concatenate(chunks))
                next_partial = total_samples + int(
                    SAMPLE_RATE * PARTIAL_TRANSCRIPTION_SECONDS
                )
            if total_samples >= SAMPLE_RATE * 12 or (
                not manual and silent_samples >= SAMPLE_RATE * TRAILING_SILENCE_SECONDS
            ):
                if manual or speech_samples >= SAMPLE_RATE * 0.2:
                    break
                # A false activation must not leak partial results into the next turn.
                return None
    if recognizer.matched.is_set():
        return None
    if chunks and total_samples >= SAMPLE_RATE * 0.2:
        return np.concatenate(chunks)
    return None


class WindowsSpeechEngine:
    """Keep one system synthesizer alive for the assistant's lifetime."""

    SCRIPT = """
        $ErrorActionPreference = 'Stop'
        $ProgressPreference = 'SilentlyContinue'
        $voice = $null
        try {
            Add-Type -AssemblyName System.Speech
            $voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
            $spanishVoice = $voice.GetInstalledVoices() | Where-Object {
                $_.Enabled -and $_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq 'es'
            } | Select-Object -First 1
            if ($spanishVoice) { $voice.SelectVoice($spanishVoice.VoiceInfo.Name) }
            else { [Console]::Error.WriteLine('No Spanish voice installed; using the default.') }
            [Console]::Out.WriteLine('READY')
            [Console]::Out.Flush()
            while ($null -ne ($line = [Console]::In.ReadLine())) {
                $text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($line))
                $voice.Speak($text)
                [Console]::Out.WriteLine('DONE')
                [Console]::Out.Flush()
            }
        } catch {
            [Console]::Error.WriteLine($_.Exception.Message)
            exit 1
        } finally {
            if ($null -ne $voice) { $voice.Dispose() }
        }
    """

    def __init__(self, stop_event: threading.Event) -> None:
        encoded_script = base64.b64encode(self.SCRIPT.encode("utf-16-le")).decode(
            "ascii"
        )
        self.process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded_script,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="ascii",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self._signals: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read_signals, daemon=True)
        self._reader.start()
        try:
            self._wait_for("READY", 15, stop_event)
        except BaseException:
            self.close()
            raise

    def _read_signals(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self._signals.put(line.strip())
        finally:
            self._signals.put(None)

    def _wait_for(
        self, expected: str, timeout: float, stop_event: threading.Event
    ) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if stop_event.is_set():
                raise InterruptedError("Speech interrupted by shutdown.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Speech engine timed out waiting for {expected}.")
            try:
                signal = self._signals.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if signal != expected:
                raise RuntimeError(
                    f"Speech engine stopped or sent an invalid signal: {signal!r}."
                )
            return

    def speak(self, text: str, stop_event: threading.Event) -> None:
        """Send speech as data and wait until actual playback completes."""
        try:
            if stop_event.is_set():
                raise InterruptedError("Speech interrupted by shutdown.")
            if self.process.poll() is not None:
                raise RuntimeError("Speech engine is no longer running.")
            assert self.process.stdin is not None
            encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
            self.process.stdin.write(encoded_text + "\n")
            self.process.stdin.flush()
            self._wait_for("DONE", 60, stop_event)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Stop playback and release the process, pipes, and reader thread."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=0.5)
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        self._reader.join(timeout=0.5)
        if self.process.stdout is not None:
            self.process.stdout.close()


def speak(
    text: str,
    stop_event: threading.Event,
    windows_engine: WindowsSpeechEngine | None = None,
) -> None:
    """Speak through the system synthesizer and allow shutdown to interrupt it."""
    if sys.platform == "win32":
        if windows_engine is None:
            raise RuntimeError("Windows speech engine is not initialized.")
        windows_engine.speak(text, stop_event)
        return
    if sys.platform == "darwin":
        command = ["say", text]
    elif shutil.which("spd-say"):
        command = ["spd-say", "-w", "-l", "es", "--", text]
    elif shutil.which("espeak-ng"):
        command = ["espeak-ng", "-v", "es", "--", text]
    else:
        raise RuntimeError(
            "Falta el sintetizador: instalá speech-dispatcher o espeak-ng."
        )

    with subprocess.Popen(command) as process:
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
    controls: VoiceControls | None = None,
) -> None:
    """Run capture, transcription, dialogue, and speech outside the UI thread."""

    def publish_status(animation: str, text: str) -> None:
        events.put(UIEvent("state", text, animation))
        print(f"[Status] {text}", flush=True)

    windows_engine = None
    speech_enabled = not options.mute
    if controls is None:
        controls = VoiceControls()
        if getattr(options, "push_to_talk", False):
            controls.push_to_talk.set()
    try:
        model = None
        if not options.text:
            publish_status(
                "idle",
                f"Cargando Whisper {options.model}. La primera vez se descarga el modelo...",
            )
            from faster_whisper import WhisperModel

            model = WhisperModel(options.model, device="cpu", compute_type="int8")
        if speech_enabled and sys.platform == "win32" and not stop_event.is_set():
            publish_status("idle", "Preparando voz...")
            try:
                windows_engine = WindowsSpeechEngine(stop_event)
            except (OSError, RuntimeError) as error:
                speech_enabled = False
                events.put(
                    UIEvent(
                        "warning", "Voz desactivada; continúo con respuestas escritas."
                    )
                )
                print(f"[Speech] {error}", flush=True)
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
                assert model is not None
                controls.mode_changed.clear()
                controls.held.clear()
                if controls.push_to_talk.is_set():
                    publish_status(
                        "idle", "Mantené Espacio para hablar; soltalo para responder."
                    )
                else:
                    publish_status(
                        "listening", "Te escucho. Hablá y luego dejá una pausa."
                    )
                recognizer = LiveTranscriber(model, dialogue, events, stop_event)
                try:
                    controls.ready.set()
                    audio = record_utterance(
                        options.microphone,
                        options.threshold,
                        stop_event,
                        controls,
                        recognizer,
                    )
                    controls.ready.clear()
                    controls.held.clear()
                    if (
                        stop_event.is_set()
                        or controls.mode_changed.is_set()
                        or recognizer.cancelled.is_set()
                    ):
                        continue
                    if audio is None and not recognizer.matched.is_set():
                        continue
                    if not recognizer.matched.is_set():
                        publish_status("idle", "Finalizando transcripción...")
                    result = recognizer.finish(audio, controls.mode_changed)
                    if (
                        result is None
                        or stop_event.is_set()
                        or controls.mode_changed.is_set()
                    ):
                        continue
                    text, response = result
                finally:
                    controls.ready.clear()
                    controls.held.clear()
                    recognizer.close()
                if not text:
                    continue
            if stop_event.is_set():
                break
            if options.text:
                response = get_response(text, dialogue)
            print(f"[User] {text}\n[Assistant] {response}", flush=True)
            events.put(UIEvent("user", text))
            events.put(UIEvent("response", response))
            if speech_enabled:
                publish_status("speaking", "Respondiendo...")
                try:
                    speak(response, stop_event, windows_engine)
                except (OSError, RuntimeError) as error:
                    speech_enabled = False
                    if windows_engine is not None:
                        windows_engine.close()
                    events.put(
                        UIEvent(
                            "warning",
                            "Voz desactivada; continúo con respuestas escritas.",
                        )
                    )
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
    finally:
        if windows_engine is not None:
            windows_engine.close()


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
        "--push-to-talk",
        "--pulsar-para-hablar",
        action="store_true",
        help="Mantener Espacio para hablar; F2 alterna con la escucha automática.",
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
    controls = VoiceControls()
    if options.push_to_talk:
        controls.push_to_talk.set()
    worker = None
    try:
        animations = load_animations()
        dialogue = load_dialogue()
        events: queue.Queue[UIEvent] = queue.Queue()
        queries: queue.Queue[str] = queue.Queue(maxsize=1)
        worker = threading.Thread(
            target=run_assistant,
            args=(options, dialogue, events, queries, stop_event, controls),
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
                elif event.type == pygame.WINDOWFOCUSLOST:
                    controls.held.clear()
                elif not options.text and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F2:
                        if controls.push_to_talk.is_set():
                            controls.push_to_talk.clear()
                        else:
                            controls.push_to_talk.set()
                        controls.held.clear()
                        controls.mode_changed.set()
                    elif (
                        event.key == pygame.K_SPACE
                        and controls.push_to_talk.is_set()
                        and controls.ready.is_set()
                    ):
                        controls.held.set()
                elif not options.text and event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        controls.held.clear()
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
                elif update.kind == "partial":
                    user_text = update.text
                elif update.kind == "matched":
                    if state == "listening":
                        message = (
                            "Respuesta preparada. Terminá de hablar o soltá Espacio."
                        )
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
            else:
                mode = (
                    "Pulsar para hablar (Espacio)"
                    if controls.push_to_talk.is_set()
                    else "Escucha automática"
                )
                draw_text(
                    screen, small_font, f"{mode} | F2: cambiar modo", 515, max_lines=1
                )
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
        controls.held.clear()
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
