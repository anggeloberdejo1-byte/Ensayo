"""Prepara las grabaciones de percusión para entrenar el Modo salsa.

Lee una carpeta por instrumento (cualquier formato de audio), recorta los silencios,
iguala el volumen y corta todo en pedazos de 10 s guardados como .npy.

Uso:
  python preparar.py ~/Documents/Ensayo-grabaciones datos/
"""
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 44100
LARGO = 10 * SR          # pedazos de 10 s
EXTENSIONES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".aif", ".aiff", ".caf", ".mp4", ".mov", ".opus", ".wma"}


def decodificar(ruta):
    """Cualquier audio -> float32 estéreo (2, N) a 44.1 kHz."""
    import imageio_ffmpeg
    salida = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(ruta), "-vn", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(salida, dtype=np.float32).reshape(-1, 2).T.copy()


def partes_con_sonido(audio):
    """Devuelve los tramos con sonido (une silencios de menos de 1.5 s)."""
    ventana = SR // 20
    n = audio.shape[1] // ventana
    energia = np.sqrt((audio[:, : n * ventana].reshape(2, n, ventana) ** 2).mean(axis=(0, 2)))
    umbral = max(np.percentile(energia, 95) * 0.03, 1e-4)
    activo = energia > umbral
    tramos, inicio, ultimo = [], None, -100
    for k, a in enumerate(activo):
        if a:
            if inicio is None:
                inicio = k
            ultimo = k
        elif inicio is not None and k - ultimo > 30:
            tramos.append((inicio * ventana, (ultimo + 1) * ventana))
            inicio = None
    if inicio is not None:
        tramos.append((inicio * ventana, (ultimo + 1) * ventana))
    return tramos


def main(origen, destino):
    origen, destino = Path(origen).expanduser(), Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    resumen = {"clases": [], "entrenar": {}, "validar": {}}
    for carpeta in sorted(p for p in origen.iterdir() if p.is_dir()):
        archivos = sorted(f for f in carpeta.rglob("*") if f.suffix.lower() in EXTENSIONES)
        if not archivos:
            continue
        clase = carpeta.name
        resumen["clases"].append(clase)
        (destino / clase).mkdir(exist_ok=True)
        # Uno de cada diez archivos se guarda para validar (nunca se usa para entrenar)
        random.shuffle(archivos)
        validar = set(archivos[: max(1, len(archivos) // 10)]) if len(archivos) >= 4 else set()
        resumen["entrenar"][clase], resumen["validar"][clase] = [], []
        segundos = 0
        for archivo in archivos:
            try:
                audio = decodificar(archivo)
            except subprocess.CalledProcessError:
                print(f"  no se pudo leer {archivo.name}, se salta")
                continue
            for a, z in partes_con_sonido(audio):
                tramo = audio[:, a:z]
                rms = np.sqrt((tramo ** 2).mean()) + 1e-9
                tramo = tramo * (0.1 / rms)  # todos al mismo volumen (-20 dBFS)
                tramo = np.clip(tramo, -1, 1)
                for i in range(0, tramo.shape[1] - SR * 2, LARGO):
                    pedazo = tramo[:, i: i + LARGO]
                    nombre = f"{clase}/{archivo.stem}_{a // SR}_{i // SR}.npy"
                    np.save(destino / nombre, pedazo.astype(np.float16))
                    (resumen["validar"] if archivo in validar else resumen["entrenar"])[clase].append(nombre)
                    segundos += pedazo.shape[1] / SR
        print(f"{clase}: {len(archivos)} archivos, {segundos / 60:.1f} min útiles, "
              f"{len(resumen['entrenar'][clase])} pedazos para entrenar, {len(resumen['validar'][clase])} para validar")
    (destino / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False))
    print("listo:", destino / "resumen.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
