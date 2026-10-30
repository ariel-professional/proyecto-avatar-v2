# Avatar en Python — prototipo

Migración simple del proyecto de `legacy/`, para probar e iterar. Un archivo
principal, recursos propios copiados del legado y cuatro dependencias principales. Sin tests,
Docker, servidor ni servicios pagos. No modifica el proyecto C++.

Escucha el micrófono, transcribe en español con Whisper, busca una respuesta en
el mismo JSON y la reproduce mientras anima el mismo avatar. Mantiene las doce
respuestas, búsqueda por substring y prioridad alfabética de claves. No hay un
chat generativo ni historial. Los textos también se muestran en la ventana y consola.

## Instalar y ejecutar en Windows

**Inicio rápido: doble clic en `iniciar.cmd`.** Solo necesita Python instalado.
El arranque prepara el entorno e instala lo que falte automáticamente. Si abrís
`main.py` desde el editor o con `python main.py`, también selecciona el entorno
correcto. La primera preparación necesita Internet.

Usar Python **3.11 o 3.12 de 64 bits**. Desde la raíz del repositorio, en PowerShell:

```powershell
cd proyecto_avatar_python
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

No hace falta activar el entorno. Si ya existe `.venv`, reutilizarlo.
Después de instalar, se puede abrir **`iniciar.cmd` con doble clic** o ejecutar
`.\iniciar.cmd` desde esta carpeta. Usa siempre el Python de `.venv` y activa
el reconocimiento de voz por defecto. Para el modo teclado: `.\iniciar.cmd --texto`.

Si aparece `No module named faster_whisper`, probablemente se abrió con otro
Python (por ejemplo desde el botón de ejecutar del editor). Usar `iniciar.cmd`
o seleccionar `.venv/Scripts/python.exe` como intérprete del editor.

La primera ejecución por voz descarga el modelo `small` de faster-whisper
(del orden de 500 MB), por lo que necesita Internet y puede demorar.
Después se usa el modelo guardado en la caché de Hugging Face del usuario.
No utiliza el `ggml-base.bin` del legado: son formatos distintos.
Una vez descargado y con las dependencias instaladas, el procesamiento es local.

Windows usa el sintetizador instalado mediante `System.Speech`, sin paquetes
adicionales. Selecciona una voz española si está disponible; si no, usa la
predeterminada y avisa en consola. La disponibilidad de voces depende de Windows.

## Prueba rápida sin micrófono ni descarga del modelo

```powershell
.\.venv\Scripts\python.exe main.py --texto
```

Escribir `hola`, `cómo estás`, `quién sos` o `gracias` y presionar Enter.
Con `--texto --sin-voz` se puede probar solo ventana, animaciones y diálogo,
sin utilizar audio. Escape o cerrar la ventana termina la aplicación.
La entrada por teclado está disponible únicamente en modo `--texto`.

## Micrófono y sensibilidad

```powershell
.\.venv\Scripts\python.exe main.py --listar-microfonos
.\.venv\Scripts\python.exe main.py --microfono 1
.\.venv\Scripts\python.exe main.py --umbral 0.03
```

El número de micrófono depende del equipo. Por defecto usa el dispositivo de
entrada del sistema a 16 kHz, mono. El umbral predeterminado es `0.03`.
Si no detecta voz, probar `--umbral 0.01`, revisar el dispositivo y el permiso de micrófono
para aplicaciones de escritorio. Si se activa con ruido, subir el umbral.
Si el dispositivo rechaza 16 kHz, seleccionar otra entrada compatible.

Se conserva aproximadamente 0,3 segundos de audio previo a la activación para
no perder el inicio de las palabras. Se graban intervenciones de hasta doce
segundos y se transcriben después de unos 1,1 segundos de silencio. Se utiliza
el detector de voz Silero integrado en faster-whisper para descartar tramos sin
habla, y decodificación con cinco candidatos y temperatura cero.
Mientras transcribe o responde no escucha; esperar
a que aparezca de nuevo “Te escucho” antes de hablar. No hay cancelación de eco,
interrupción del asistente.

## Linux

En Ubuntu/Debian, además de Python y su soporte para venv:

```bash
sudo apt install python3-venv libportaudio2 speech-dispatcher speech-dispatcher-espeak-ng espeak-ng
cd proyecto_avatar_python
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

Necesita una sesión gráfica y audio configurado. Usa `spd-say` en español;
si no está disponible, intenta `espeak-ng`. En macOS se contempla el comando
`say`, con la voz predeterminada del sistema, pero no se verificó ese entorno.

## Cambiar respuestas y modelo

- Editar `resources/dialog/preguntas.json` y reiniciar. Conserva el formato
  `{"preguntas": [{"clave": "hola", "respuesta": "Hola!"}]}`.
- `--modelo small` es el predeterminado, para priorizar precisión. Es más lento
  y ocupa más memoria que `base`. Para una máquina con pocos recursos se puede
  usar `--modelo base` o `--modelo tiny`, con menor precisión esperable.
  Cada modelo se descarga al primer uso. Se ejecuta en CPU con int8;
  no requiere CUDA. También admite una carpeta local de modelo CTranslate2.
- `--sin-voz` desactiva solamente la reproducción de las respuestas.
- `AVATAR_RESOURCES_DIR` permite usar otra carpeta de recursos. Normalmente
  no se necesita: las rutas se resuelven desde `main.py`, sin depender del cwd.

## Diferencias pequeñas respecto al legado

- Normaliza mayúsculas y tildes para buscar respuestas.
- Muestra la animación de escucha y los textos de la interacción.
- Conserva los silencios dentro de la grabación y acepta intervenciones breves.
- Transcribe fuera del hilo gráfico; el micrófono se cierra durante la respuesta.
- La animación de habla dura lo que tarda el comando de síntesis, en lugar de
  calcular una duración a partir de la cantidad de letras.
- Analiza los metadatos de animación con el parser JSON de Python.

No se busca equivalencia exacta de transcripciones con whisper.cpp: cambió el
motor de inferencia, aunque se mantiene Whisper `base` multilingüe en español.
La precisión y latencia deben evaluarse hablando en el equipo objetivo.
Un error de modelo o micrófono se muestra en pantalla y consola; corregirlo y
reiniciar. Un fallo de síntesis conserva la respuesta escrita y permite continuar.
El cierre de la ventana es inmediato; una inferencia nativa en curso no tiene
cancelación fina en este prototipo.

## Comprobación realizada

En Windows con Python 3.12: instalación e imports correctos, dependencias
compatibles (`pip check`), carga del modelo `base`, lectura del micrófono
predeterminado a 16 kHz, síntesis a un WAV con la voz española Helena y
transcripción de ese audio como “Hola, como estas?”. La consulta produjo
“Todo bien por aca, y vos?”. También se ejecutó la ventana con driver SDL
sin pantalla, ingreso por teclado, animaciones y cierre, y se inspeccionó
una captura del resultado. Los siete archivos de recursos son idénticos
a los originales. No se agregaron tests.

Falta comprobar una conversación real de punta a punta, la reproducción por
los altavoces y el funcionamiento en Linux/macOS. Los modelos `base` y `small`
ya quedaron descargados en la caché del usuario de este equipo.

Después del ajuste de reconocimiento se comprobó también el filtro Silero:
una frase sintetizada de 5,31 segundos se transcribió completa; muestras de
silencio y ruido sintético no produjeron texto. Esto no mide la precisión
con la voz del usuario ni con ruido real del ambiente.

Referencias de las bibliotecas: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[sounddevice](https://python-sounddevice.readthedocs.io/en/latest/),
[pygame](https://www.pygame.org/docs/).
