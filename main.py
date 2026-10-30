"""Prototipo del avatar: pygame + micrófono + Whisper + respuestas del JSON."""

import argparse
import base64
from collections import deque
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import unicodedata

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parent
RESOURCES = Path(os.environ.get("AVATAR_RESOURCES_DIR") or ROOT / "resources")
SAMPLE_RATE = 16000
BLOCK_SIZE = 512


def preparar_python():
    """En Windows, usar y preparar el entorno local aunque se abra desde el editor."""
    if sys.platform != "win32":
        return
    python_local = ROOT / ".venv/Scripts/python.exe"
    if not python_local.is_file():
        print("[Inicio] Preparando Python para el avatar...", flush=True)
        subprocess.check_call([sys.executable, "-m", "venv", str(ROOT / ".venv")])
    faltan = subprocess.call(
        [str(python_local), "-c", "import pygame, sounddevice, numpy, faster_whisper"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if faltan:
        print("[Inicio] Instalando las dependencias necesarias...", flush=True)
        subprocess.check_call([str(python_local), "-m", "pip", "install",
                               "--disable-pip-version-check", "-r", str(ROOT / "requirements.txt")])
    if Path(sys.executable).resolve() != python_local.resolve():
        sys.exit(subprocess.call([str(python_local), str(ROOT / "main.py"), *sys.argv[1:]]))


def normalizar(texto):
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.casefold())
        if not unicodedata.combining(c)
    )


def cargar_dialogo():
    datos = json.loads((RESOURCES / "dialog/preguntas.json").read_text(encoding="utf-8"))
    # Mismo criterio del legado: primera coincidencia en orden de clave.
    return sorted({normalizar(p["clave"]): p["respuesta"] for p in datos["preguntas"]}.items())


def responder(texto, dialogo):
    consulta = normalizar(texto)
    return next((respuesta for clave, respuesta in dialogo if clave in consulta),
                "Disculpa, no entendí tu consulta.")


def escuchar(microfono, umbral, detener):
    import numpy as np
    import sounddevice as sd

    bloques = []
    previos = deque(maxlen=10)  # ~320 ms: conservar el comienzo de las palabras.
    silencio = 0
    voz = 0
    # Lectura bloqueante en el trabajador: sin callback ni vector compartido.
    with sd.InputStream(device=microfono, samplerate=SAMPLE_RATE,
                        channels=1, dtype="float32", blocksize=BLOCK_SIZE) as stream:
        while not detener.is_set():
            bloque, overflow = stream.read(BLOCK_SIZE)
            if overflow:
                print("[Audio] Se perdieron muestras; se reinicia esta captura.", flush=True)
                bloques.clear()
                previos.clear()
                voz = silencio = 0
                continue
            activo = float(np.max(np.abs(bloque))) >= umbral
            muestras = bloque[:, 0].copy()
            if not bloques:
                previos.append(muestras)
                if not activo:
                    continue
                bloques = list(previos)
                previos.clear()
            else:
                bloques.append(muestras)
            if activo:
                voz += len(bloque)
                silencio = 0
            elif bloques:
                silencio += len(bloque)
            if bloques and (silencio >= SAMPLE_RATE * 1.1
                            or len(bloques) * BLOCK_SIZE >= SAMPLE_RATE * 12):
                if voz >= SAMPLE_RATE * 0.2:
                    return np.concatenate(bloques)
                bloques.clear()
                previos.clear()
                voz = silencio = 0
    return None


def hablar(texto, detener):
    if sys.platform == "win32":
        # El texto se codifica como datos; nunca se interpola en un shell.
        contenido = base64.b64encode(texto.encode("utf-8")).decode("ascii")
        script = """
        $ErrorActionPreference = 'Stop'
        Add-Type -AssemblyName System.Speech
        $voz = New-Object System.Speech.Synthesis.SpeechSynthesizer
        try {
            $es = $voz.GetInstalledVoices() | Where-Object {
                $_.Enabled -and $_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq 'es'
            } | Select-Object -First 1
            if ($es) { $voz.SelectVoice($es.VoiceInfo.Name) }
            else { Write-Warning 'No hay voz española instalada; se usa la predeterminada.' }
            $texto = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('CONTENIDO'))
            $voz.Speak($texto)
        } finally { $voz.Dispose() }
        """.replace("CONTENIDO", contenido)
        codificado = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        comando = ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", codificado]
    elif sys.platform == "darwin":
        comando = ["say", texto]
    elif shutil.which("spd-say"):
        comando = ["spd-say", "-w", "-l", "es", "--", texto]
    elif shutil.which("espeak-ng"):
        comando = ["espeak-ng", "-v", "es", "--", texto]
    else:
        raise RuntimeError("Falta el sintetizador: instalá speech-dispatcher o espeak-ng.")

    opciones = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    with subprocess.Popen(comando, **opciones) as proceso:
        inicio = time.monotonic()
        while proceso.poll() is None:
            if detener.wait(0.05) or time.monotonic() - inicio > 60:
                proceso.terminate()
                proceso.wait()
                return
        if proceso.returncode:
            raise RuntimeError(f"El sintetizador terminó con código {proceso.returncode}.")


def trabajar(opciones, dialogo, eventos, consultas, detener):
    def estado(animacion, texto):
        eventos.put(("estado", (animacion, texto)))
        print(f"[Estado] {texto}", flush=True)

    try:
        modelo = None
        if not opciones.texto:
            estado("reposo", f"Cargando Whisper {opciones.modelo}. La primera vez se descarga el modelo...")
            from faster_whisper import WhisperModel

            modelo = WhisperModel(opciones.modelo, device="cpu", compute_type="int8")
        while not detener.is_set():
            if opciones.texto:
                estado("reposo", "Escribí una consulta y presioná Enter.")
                while not detener.is_set():
                    try:
                        texto = consultas.get(timeout=0.1)
                        break
                    except queue.Empty:
                        pass
                else:
                    break
            else:
                estado("escuchando", "Te escucho. Hablá y luego dejá una pausa.")
                audio = escuchar(opciones.microfono, opciones.umbral, detener)
                if audio is None or detener.is_set():
                    break
                estado("reposo", "Transcribiendo...")
                segmentos, _ = modelo.transcribe(
                    audio, language="es", beam_size=5, temperature=0,
                    condition_on_previous_text=False, vad_filter=True,
                    vad_parameters={"min_speech_duration_ms": 200,
                                    "min_silence_duration_ms": 500,
                                    "speech_pad_ms": 200},
                )
                texto = "".join(segmento.text for segmento in segmentos).strip()
                if not texto or "[" in texto or "(" in texto:
                    continue
            if detener.is_set():
                break
            respuesta = responder(texto, dialogo)
            print(f"[Usuario] {texto}\n[Asistente] {respuesta}", flush=True)
            eventos.put(("usuario", texto))
            eventos.put(("respuesta", respuesta))
            if not opciones.sin_voz:
                estado("hablando", "Respondiendo...")
                try:
                    hablar(respuesta, detener)
                except (OSError, RuntimeError) as error:
                    eventos.put(("aviso", str(error)))
                    print(f"[Voz] {error}", flush=True)
            # El micrófono está cerrado mientras se transcribe y habla.
    except ModuleNotFoundError as error:
        mensaje = (f"Falta {error.name} en el Python usado para abrir la app. "
                   "En Windows, cerrá y ejecutá iniciar.cmd. "
                   "Si sigue fallando, instalá requirements.txt en .venv.")
        eventos.put(("error", mensaje))
        print(f"[Error] {mensaje}\nIntérprete: {sys.executable}", file=sys.stderr, flush=True)
    except Exception as error:
        mensaje = f"{type(error).__name__}: {error}"
        eventos.put(("error", mensaje))
        print(f"[Error] {mensaje}", file=sys.stderr, flush=True)


def cargar_animaciones(pygame):
    animaciones = {}
    for estado, nombre in [("reposo", "AnimacionReposo"),
                           ("escuchando", "escuchando"), ("hablando", "hablando")]:
        base = RESOURCES / "animations" / nombre
        hoja = pygame.image.load(str(base.with_suffix(".png"))).convert_alpha()
        datos = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        frames = []
        for item in datos["frames"].values():
            f = item["frame"]
            recorte = hoja.subsurface((f["x"], f["y"], f["w"], f["h"]))
            frames.append(pygame.transform.scale(recorte, (f["w"] * 8, f["h"] * 8)))
        animaciones[estado] = frames
    return animaciones


def dibujar_texto(pantalla, fuente, texto, y, color=(45, 45, 55), max_lineas=3):
    lineas = [""]
    for palabra in texto.split():
        candidata = (lineas[-1] + " " + palabra).strip()
        if fuente.size(candidata)[0] > 740 and lineas[-1]:
            lineas.append(palabra)
        else:
            lineas[-1] = candidata
    for linea in lineas[:max_lineas]:
        pantalla.blit(fuente.render(linea, True, color), (30, y))
        y += fuente.get_linesize()
    return y


def main():
    parser = argparse.ArgumentParser(description="Avatar virtual: voz, respuestas y animaciones del legado.")
    parser.add_argument("--texto", action="store_true", help="Consultar con teclado, sin micrófono ni modelo.")
    parser.add_argument("--sin-voz", action="store_true", help="Mostrar las respuestas sin sintetizar audio.")
    parser.add_argument("--modelo", default="small", help="Modelo faster-whisper o carpeta local (default: small).")
    parser.add_argument("--microfono", type=int, help="Número de dispositivo; por defecto el del sistema.")
    parser.add_argument("--listar-microfonos", action="store_true", help="Listar dispositivos y salir.")
    parser.add_argument("--umbral", type=float, default=0.03, help="Sensibilidad de entrada (default: 0.03).")
    opciones = parser.parse_args()
    if not 0 < opciones.umbral <= 1:
        parser.error("--umbral debe estar entre 0 (excluido) y 1.")
    if opciones.listar_microfonos:
        import sounddevice as sd

        print(sd.query_devices())
        return

    import pygame

    pygame.display.init()
    pygame.font.init()
    pantalla = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Asistente Virtual - Python")
    detener = threading.Event()
    trabajador = None
    try:
        animaciones = cargar_animaciones(pygame)
        dialogo = cargar_dialogo()
        eventos = queue.Queue()
        consultas = queue.Queue(maxsize=1)
        trabajador = threading.Thread(target=trabajar,
                                     args=(opciones, dialogo, eventos, consultas, detener), daemon=True)
        trabajador.start()
        fuente = pygame.font.SysFont("segoeui", 20)
        pequena = pygame.font.SysFont("segoeui", 17)
        reloj = pygame.time.Clock()
        estado, mensaje = "reposo", "Iniciando..."
        usuario = respuesta = entrada = aviso = ""
        inicio_frame = pygame.time.get_ticks()
        ejecutando = True
        ocupado = True
        pygame.key.start_text_input()
        while ejecutando:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT or (evento.type == pygame.KEYDOWN and evento.key == pygame.K_ESCAPE):
                    ejecutando = False
                elif opciones.texto and evento.type == pygame.TEXTINPUT:
                    entrada = (entrada + evento.text)[:250]
                elif opciones.texto and evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_BACKSPACE:
                        entrada = entrada[:-1]
                    elif evento.key == pygame.K_RETURN and entrada.strip() and not ocupado:
                        consultas.put_nowait(entrada.strip())
                        entrada = aviso = ""
                        ocupado = True
            while True:
                try:
                    tipo, valor = eventos.get_nowait()
                except queue.Empty:
                    break
                if tipo == "estado":
                    nuevo_estado, mensaje = valor
                    if nuevo_estado != estado:
                        inicio_frame = pygame.time.get_ticks()
                    estado = nuevo_estado
                    ocupado = not mensaje.startswith("Escribí")
                elif tipo == "usuario":
                    usuario = valor
                elif tipo == "respuesta":
                    respuesta = valor
                elif tipo == "aviso":
                    aviso = valor
                elif tipo == "error":
                    estado, mensaje, aviso = "reposo", "No se pudo continuar. Revisá la consola.", valor
                    ocupado = True

            pantalla.fill((255, 255, 255))
            dibujar_texto(pantalla, pequena, mensaje, 20, max_lineas=2)
            frames = animaciones[estado]
            frame = frames[((pygame.time.get_ticks() - inicio_frame) // 40) % len(frames)]
            pantalla.blit(frame, frame.get_rect(center=(400, 235)))
            dibujar_texto(pantalla, fuente, f"Vos: {usuario}", 380, max_lineas=2)
            dibujar_texto(pantalla, fuente, f"Avatar: {respuesta}", 435, max_lineas=2)
            if opciones.texto:
                dibujar_texto(pantalla, pequena, "> " + entrada[-85:], 505, max_lineas=2)
            dibujar_texto(pantalla, pequena, aviso or "Esc: cerrar", 560,
                          color=(170, 45, 45) if aviso else (110, 110, 110), max_lineas=1)
            pygame.display.flip()
            reloj.tick(60)
    finally:
        detener.set()
        pygame.quit()
        if trabajador is not None:
            trabajador.join(timeout=1)


if __name__ == "__main__":
    try:
        preparar_python()
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"[Error] {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
