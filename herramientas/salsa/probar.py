"""Prueba el modelo del Modo salsa con una canción de verdad.

Primero saca la percusión de la canción con HTDemucs (como hace Ensayo) y después
la separa con el modelo entrenado. Guarda un .wav por instrumento para escucharlos.

Uso:
  python probar.py salida/modelo_salsa.pt cancion.mp3 resultado/
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.hdemucs import HDemucs
from demucs.pretrained import get_model

SR = 44100


def decodificar(ruta):
    import imageio_ffmpeg
    salida = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(ruta), "-vn", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        capture_output=True, check=True,
    ).stdout
    return torch.from_numpy(np.frombuffer(salida, dtype=np.float32).reshape(-1, 2).T.copy())


def separar(modelo, audio, dispositivo):
    ref = audio.mean(0)
    media, desv = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        salida = apply_model(modelo, ((audio - media) / desv)[None], device=dispositivo, shifts=0, split=True, overlap=0.25)[0]
    return salida * desv + media


def main(pesos, cancion, destino):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    dispositivo = "mps" if torch.backends.mps.is_available() else "cpu"
    audio = decodificar(cancion)

    base = get_model("htdemucs")
    pistas = separar(base, audio, dispositivo)
    bateria = pistas[base.sources.index("drums")]
    sf.write(destino / "00_percusion_completa.wav", bateria.T.numpy(), SR)

    paquete = torch.load(pesos, map_location="cpu", weights_only=True)
    modelo = HDemucs(*paquete["args"], **paquete["kwargs"])
    modelo.load_state_dict(paquete["state"])
    modelo.eval()
    piezas = separar(modelo, bateria, dispositivo)
    for i, nombre in enumerate(modelo.sources):
        sf.write(destino / f"{i + 1:02d}_{nombre}.wav", piezas[i].T.numpy(), SR)
        print(f"{nombre}: {20 * np.log10(float(piezas[i].pow(2).mean().sqrt()) + 1e-9):.0f} dB")
    print("listo:", destino)


if __name__ == "__main__":
    main(*sys.argv[1:4])
