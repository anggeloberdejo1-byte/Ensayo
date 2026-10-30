# Ensayo

Separa la voz, la batería, el bajo y más de cualquier canción con IA. Cambia el tempo y el tono, ve los acordes mientras suena, repite secciones y toca con un metrónomo que sigue la canción. By Anggelo Berdejo.

## Qué hace

- **Separación de pistas** con [Demucs](https://github.com/facebookresearch/demucs): 4 pistas (voz, batería, bajo, otros) o 6 pistas (más guitarra y piano). Silenciar, solo, volumen por pista y descarga en .wav.
- **Velocidad** del 50 % al 150 % sin cambiar la afinación, y **tono** de −12 a +12 semitonos.
- **Acordes** en una tira que corre con la música. Se calculan sin la voz ni la batería y con el bajo como guía. Se transponen solos si cambias el tono.
- **Tonalidad, BPM e inicio de compás.**
- **Secciones** (Intro, Estrofa, Coro, Puente). Toca una para repetirla, o arrastra sobre las ondas para marcar tu propio loop.
- **Metrónomo** sobre los pulsos detectados de la canción.
- **Ajustes:** español o inglés, 12 paletas de colores y notación Do Re Mi o C D E.

## Cómo usarla

Ensayo necesita correr en una computadora, porque la separación con IA usa el procesador gráfico. Funciona mejor en Mac con chip M.

1. Descarga el repositorio (botón **Code → Download ZIP**) y descomprímelo.
2. Doble clic en `iniciar.command`. La primera vez instala lo necesario (unos minutos y ~2 GB).
3. Se abre en el navegador: http://localhost:8765

Requisitos: Python 3.9 o más reciente.

## Archivos

| Archivo | Función |
|---|---|
| `server.py` | Servidor (FastAPI): subir canciones, cola de procesamiento, Demucs |
| `analysis.py` | BPM, compás, tonalidad, acordes y secciones |
| `static/engine.js` | Motor de audio: mezcla, velocidad y tono (WSOLA), loops y metrónomo |
| `static/index.html` | Interfaz |

## Limitaciones

- Acordes: mayor, menor, 7 y m7.
- Asume compás de 4/4.
- Los nombres de las secciones son una estimación por repetición y energía.

---

Hecha por Anggelo Berdejo — El Ñañe · [@anggelo.berdejo](https://instagram.com/anggelo.berdejo) · [Otras herramientas](https://anggeloberdejo1-byte.github.io/apps/)
