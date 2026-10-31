# Ensayo

Separa la voz, la batería, el bajo y más de cualquier canción con IA, directo en el navegador. Cambia la velocidad y el tono, ve los acordes mientras suena, repite secciones y toca con un metrónomo que sigue la canción. By Anggelo Berdejo.

**Abrir la app:** https://anggeloberdejo1-byte.github.io/Ensayo/

## Qué hace

- **Separación de pistas** con IA, en dos modos:
  - **Rápido · 4 pistas:** voz, batería, bajo y otros ([HTDemucs](https://github.com/facebookresearch/demucs)).
  - **Estudio · 9 pistas:** voz, bombo, redoblante, toms, platillos, bajo, guitarra, piano y otros (HTDemucs 6 pistas + [drumsep](https://github.com/inagoy/drumsep) para las piezas de la batería).
  - Silenciar, solo, volumen por pista y descarga en .wav.
- **Letra sincronizada** con IA ([Whisper](https://github.com/openai/whisper) vía [Transformers.js](https://github.com/huggingface/transformers.js)): escucha solo las partes donde se canta y va llenando cada palabra de color mientras suena. Toca una línea para saltar ahí.
  - **Tu propia letra:** pégala (o carga un .txt) al subir la canción o desde el reproductor; la IA solo pone los tiempos y se ve tu texto tal cual. Un archivo .lrc ya trae los tiempos.
  - **Karaoke a pantalla completa** (botón Karaoke o tecla K).
- **Velocidad** del 50 % al 150 % sin cambiar la afinación, y **tono** de −12 a +12 semitonos.
- **Acordes** en una tira que corre con la música. Se calculan sin la voz ni la batería y con el bajo como guía. Se transponen solos si cambias el tono.
- **Tonalidad, BPM e inicio de compás.**
- **Secciones** (Intro, Estrofa, Coro, Puente). Toca una para repetirla, o arrastra sobre las ondas para marcar tu propio loop.
- **Metrónomo** sobre los pulsos detectados de la canción.
- **Ajustes:** español o inglés, 12 paletas de colores y notación Do Re Mi o C D E.

Todo pasa en tu equipo: la canción no se sube a ningún servidor. Los modelos de IA se descargan la primera vez que se usan y quedan guardados (rápido 90 MB, estudio 230 MB, letra 205 o 410 MB); después funciona sin internet. Usa la GPU con WebGPU (Chrome, Edge, Safari reciente); sin GPU también funciona, pero más lento.

## Archivos

| Archivo | Qué es |
|---|---|
| `index.html` | Toda la app (HTML, CSS y JS): interfaz, motor de audio, separador y análisis |
| `models/htdemucs.onnx` | Modelo de 4 pistas (HTDemucs exportado a ONNX, pesos en float16) |
| `models/htdemucs_6s.onnx` | Modelo de 6 pistas para el Modo estudio |
| `models/bateria.onnx.1` y `.2` | drumsep (bombo, redoblante, platillos, toms), partido en dos porque GitHub no acepta archivos de más de 100 MB |
| `sello.webp` | Sello "By Anggelo Berdejo" |
| `manifest.json`, `sw.js`, íconos | Para instalarla como app y usarla sin internet |
| `herramientas/` | Scripts para volver a exportar el modelo desde PyTorch |
| `escritorio/` | Versión de escritorio con servidor Python (también separa en 6 pistas) |

Cada vez que se actualice `index.html`, hay que subir la versión del caché en `sw.js` (`ensayo-v1` → `ensayo-v2`…).

## Limitaciones

- Acordes: mayor, menor, 7 y m7.
- Asume compás de 4/4.
- Los nombres de las secciones son una estimación por repetición y energía.
- La letra puede tener errores, sobre todo con mucha reverberación o coros encima de la voz.

---

Hecha por Anggelo Berdejo — El Ñañe · [@anggelo.berdejo](https://instagram.com/anggelo.berdejo) · [Otras herramientas](https://anggeloberdejo1-byte.github.io/apps/)
