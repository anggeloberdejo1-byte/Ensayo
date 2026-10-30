# Ensayo

Separa la voz, la batería, el bajo y más de cualquier canción con IA, directo en el navegador. Cambia la velocidad y el tono, ve los acordes mientras suena, repite secciones y toca con un metrónomo que sigue la canción. By Anggelo Berdejo.

**Abrir la app:** https://anggeloberdejo1-byte.github.io/Ensayo/

## Qué hace

- **Separación de pistas** con IA ([HTDemucs](https://github.com/facebookresearch/demucs)): voz, batería, bajo y otros. Silenciar, solo, volumen por pista y descarga en .wav.
- **Velocidad** del 50 % al 150 % sin cambiar la afinación, y **tono** de −12 a +12 semitonos.
- **Acordes** en una tira que corre con la música. Se calculan sin la voz ni la batería y con el bajo como guía. Se transponen solos si cambias el tono.
- **Tonalidad, BPM e inicio de compás.**
- **Secciones** (Intro, Estrofa, Coro, Puente). Toca una para repetirla, o arrastra sobre las ondas para marcar tu propio loop.
- **Metrónomo** sobre los pulsos detectados de la canción.
- **Ajustes:** español o inglés, 12 paletas de colores y notación Do Re Mi o C D E.

Todo pasa en tu equipo: la canción no se sube a ningún servidor. La primera vez descarga el modelo de IA (90 MB) y lo guarda; después funciona sin internet. Usa la GPU con WebGPU (Chrome, Edge, Safari reciente); sin GPU también funciona, pero más lento.

## Archivos

| Archivo | Qué es |
|---|---|
| `index.html` | Toda la app (HTML, CSS y JS): interfaz, motor de audio, separador y análisis |
| `models/htdemucs.onnx` | El modelo de IA (HTDemucs exportado a ONNX, pesos en float16) |
| `manifest.json`, `sw.js`, íconos | Para instalarla como app y usarla sin internet |
| `herramientas/` | Scripts para volver a exportar el modelo desde PyTorch |
| `escritorio/` | Versión de escritorio con servidor Python (también separa en 6 pistas) |

Cada vez que se actualice `index.html`, hay que subir la versión del caché en `sw.js` (`ensayo-v1` → `ensayo-v2`…).

## Limitaciones

- Acordes: mayor, menor, 7 y m7.
- Asume compás de 4/4.
- Los nombres de las secciones son una estimación por repetición y energía.

---

Hecha por Anggelo Berdejo — El Ñañe · [@anggelo.berdejo](https://instagram.com/anggelo.berdejo) · [Otras herramientas](https://anggeloberdejo1-byte.github.io/apps/)
