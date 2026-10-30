# Avatar virtual en Python

Esta versión permite seguir trabajando sobre el proyecto original en C++,
conservando sus animaciones y respuestas. El reconocimiento de voz se ejecuta
localmente.

## Empezar en Windows

1. Instalá **Python 3.11 o 3.12 de 64 bits** y seleccioná **Add Python to PATH**
   durante la instalación.
2. Abrí la carpeta `proyecto_avatar_python` y hacé doble clic en **`iniciar.cmd`**.
3. Esperá a que aparezca **“Te escucho”**, hablá y dejá una pausa al terminar.

El inicio crea el entorno `.venv` e instala los paquetes que falten. La primera
vez también descarga el modelo de reconocimiento `small`, de unos **500 MB**.
Necesitás Internet para esa preparación; después el modelo queda guardado
en la caché de Hugging Face del usuario y puede usarse sin conexión.

Mientras transcribe o responde, el micrófono queda cerrado. Esperá a que vuelva
a mostrar “Te escucho” antes de hacer otra consulta. Para salir, presioná
**Escape** o cerrá la ventana.

También podés iniciarlo desde PowerShell, estando en esta carpeta:

```powershell
.\iniciar.cmd
```

## Dependencias

### Python y paquetes

Usá Python **3.11 o 3.12 de 64 bits**, con `pip` y `venv`. La versión comprobada
en este proyecto es Python 3.12 en Windows.

Los paquetes y sus rangos de versión están declarados en
[`requirements.txt`](requirements.txt):

| Paquete | Versión admitida | Para qué se usa |
|---|---|---|
| `pygame` | `>=2.6,<3` | Ventana, animaciones, teclado y eventos. |
| `sounddevice` | `>=0.5,<0.6` | Captura del micrófono mediante PortAudio. |
| `faster-whisper` | `>=1.2,<2` | Transcripción en español y detección de voz. |
| `numpy` | `>=1.26,<3` | Manejo de las muestras de audio. |
| `av` (PyAV) | `>=11,<17` | Procesamiento de audio requerido por faster-whisper. El límite evita una incompatibilidad detectada con versiones posteriores. |

`pip` instala también las dependencias de esos paquetes, entre ellas
`ctranslate2` para la inferencia, `onnxruntime` para el detector de voz Silero,
`huggingface-hub` para descargar modelos, `tokenizers`, `tqdm` y `cffi`.
Sus propias dependencias se resuelven automáticamente; no hay que instalarlas
una por una. Los módulos como `json`, `threading` y `pathlib` vienen con Python.

Si preferís hacer la instalación manual en Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

### Sistema, audio y modelo

| Requisito | Detalle |
|---|---|
| Interfaz gráfica | Una sesión de escritorio para abrir la ventana. |
| Entrada de audio | Micrófono con permiso de acceso y soporte para 16 kHz, mono. |
| Salida de audio | Altavoces o auriculares para escuchar las respuestas. |
| Modelo de voz | Whisper `small` multilingüe, en formato CTranslate2. Se descarga automáticamente. El archivo `ggml-base.bin` del legado no sirve para este motor. |
| Windows | Windows PowerShell y `System.Speech` para sintetizar voz. Usa una voz española instalada si la encuentra; de lo contrario, la predeterminada. CTranslate2 también requiere el runtime de Visual C++, que suele estar instalado. |
| Linux | PortAudio (`libportaudio2`) y Speech Dispatcher con su módulo espeak-ng, o el comando `espeak-ng` como alternativa. |
| macOS | El comando del sistema `say`, con la voz predeterminada. Este entorno todavía no fue verificado. |

En Windows y macOS, la instalación de `sounddevice` con `pip` incluye PortAudio.
Las distribuciones binarias de pygame incluyen las bibliotecas SDL necesarias
para la ventana. PyAV incluye FFmpeg en sus paquetes binarios. La aplicación
usa CPU con int8: no requiere una GPU ni CUDA.

Referencias de instalación: [sounddevice](https://python-sounddevice.readthedocs.io/en/latest/installation.html),
[CTranslate2 y runtime de Visual C++](https://opennmt.net/CTranslate2/installation.html),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Probar con el teclado

Para consultar sin micrófono ni cargar el modelo de reconocimiento:

```powershell
.\iniciar.cmd --text
```

Escribí una consulta y presioná **Enter**. Podés probar con `hola`, `cómo estás`,
`quién sos` o `gracias`. Para desactivar también la lectura de las respuestas:

```powershell
.\iniciar.cmd --text --mute
```

El modo teclado usa las mismas respuestas del JSON. El campo para escribir
solo aparece cuando se inicia con `--text`.

## Ajustar micrófono y reconocimiento

Estas opciones se agregan al comando de inicio:

Las opciones usan nombres en inglés. Por compatibilidad, también se aceptan
los anteriores: `--texto`, `--sin-voz`, `--modelo`, `--microfono`,
`--listar-microfonos` y `--umbral`. La interfaz y las respuestas siguen en español.

| Opción | Ejemplo | Efecto |
|---|---|---|
| `--list-microphones` | `.\iniciar.cmd --list-microphones` | Muestra los dispositivos disponibles y sale. |
| `--microphone N` | `.\iniciar.cmd --microphone 1` | Elige una entrada por su número. Por defecto usa la del sistema. |
| `--threshold VALOR` | `.\iniciar.cmd --threshold 0.01` | Cambia la sensibilidad. El valor predeterminado es `0.03`; un valor menor detecta sonidos más suaves. |
| `--model NOMBRE` | `.\iniciar.cmd --model base` | Cambia el modelo. `small` es el predeterminado; `base` y `tiny` consumen menos recursos, con menor precisión esperable. |
| `--mute` | `.\iniciar.cmd --mute` | Muestra las respuestas sin leerlas en voz alta. |

La grabación conserva unos 0,3 segundos previos a la activación para no perder
el comienzo de las palabras. Espera unos 1,1 segundos de silencio para cerrar
la frase y admite hasta 12 segundos por intervención. Antes de transcribir,
Silero filtra los tramos sin voz.

Cada modelo se descarga al usarlo por primera vez. `--model` también admite
la ruta a una carpeta local con un modelo CTranslate2.

## Editar las respuestas

La base está en [`resources/dialog/preguntas.json`](resources/dialog/preguntas.json).
Contiene un array llamado `preguntas`, con una `clave` y una `respuesta` por entrada:

```json
{
  "preguntas": [
    {
      "clave": "hola",
      "respuesta": "Hola, ¿en qué te puedo ayudar?"
    }
  ]
}
```

Podés agregar o modificar entradas respetando ese formato. Guardá el archivo
y reiniciá la aplicación para cargar los cambios.

La búsqueda ignora mayúsculas y tildes. Si encuentra una clave dentro de la
consulta, devuelve su respuesta. Cuando coinciden varias, elige la primera
en orden alfabético de clave. Si no encuentra ninguna, responde
“Disculpa, no entendí tu consulta”.

Los recursos se buscan junto a `main.py`, por lo que podés abrir el programa
desde otra carpeta. Si necesitás usar recursos distintos, definí la variable
de entorno `AVATAR_RESOURCES_DIR` con la ruta a esa carpeta.

## Instalar en Linux

En Ubuntu o Debian, instalá Python, el soporte para entornos virtuales y los
componentes de audio:

```bash
sudo apt install python3 python3-venv python3-pip libportaudio2 speech-dispatcher speech-dispatcher-espeak-ng espeak-ng
```

Después, desde `proyecto_avatar_python`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

Las opciones son las mismas; por ejemplo, `.venv/bin/python main.py --text`.
La sesión debe tener una pantalla y el audio configurados. La instalación
automática de `iniciar.cmd` corresponde únicamente a Windows.
