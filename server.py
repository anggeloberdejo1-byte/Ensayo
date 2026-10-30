"""Ensayo: servidor local que separa pistas (Demucs) y analiza BPM, tonalidad, acordes y secciones."""
import json
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import analysis

ROOT = Path(__file__).parent
SONGS = ROOT / "songs"
SONGS.mkdir(exist_ok=True)
SR = 44100
MODELS = {"4": "htdemucs", "6": "htdemucs_6s"}
STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]

app = FastAPI()
jobs = queue.Queue()
status = {}  # id -> {"stage", "progress", "error"}


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode(path):
    """Cualquier audio/video -> float32 estéreo (2, T) a 44.1 kHz."""
    out = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).reshape(-1, 2).T.copy()


_models = {}


def get_model(name):
    if name not in _models:
        from demucs.pretrained import get_model as load
        m = load(name)
        m.eval()
        _models[name] = m
    return _models[name]


def separate(audio, model_name, on_progress):
    import torch
    import demucs.apply as dapply

    model = get_model(model_name)
    wav = torch.from_numpy(audio)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    # demucs muestra el progreso con tqdm; lo interceptamos para la barra de la app
    real_tqdm = dapply.tqdm.tqdm

    def tracking_tqdm(iterable, *args, **kwargs):
        items = list(iterable)
        for i, item in enumerate(items):
            on_progress(i / max(len(items), 1))
            yield item
        on_progress(1.0)

    dapply.tqdm.tqdm = tracking_tqdm
    try:
        devices = (["mps"] if torch.backends.mps.is_available() else []) + ["cpu"]
        for dev in devices:
            try:
                with torch.no_grad():
                    out = dapply.apply_model(model, wav[None], device=dev, shifts=1, split=True,
                                             overlap=0.25, progress=True)[0]
                break
            except Exception:
                if dev == devices[-1]:
                    raise
                traceback.print_exc()
                print(f"Demucs falló en {dev}, reintentando en CPU")
    finally:
        dapply.tqdm.tqdm = real_tqdm
    out = (out * std + mean).cpu().numpy()
    return {name: out[i] for i, name in enumerate(model.sources)}


def set_status(song_id, stage, progress, error=None):
    status[song_id] = {"stage": stage, "progress": round(progress, 3), "error": error}


def process(song_id):
    d = SONGS / song_id
    meta = json.loads((d / "meta.json").read_text())
    try:
        set_status(song_id, "Decodificando audio", 0.02)
        audio = decode(d / meta["file"])
        if audio.shape[1] < SR:
            raise ValueError("El audio es demasiado corto")

        set_status(song_id, "Separando pistas con IA", 0.05)
        stems = separate(audio, MODELS[meta["model"]],
                         lambda p: set_status(song_id, "Separando pistas con IA", 0.05 + 0.8 * p))

        set_status(song_id, "Guardando pistas", 0.86)
        (d / "stems").mkdir(exist_ok=True)
        names = [n for n in STEM_ORDER if n in stems]
        for n in names:
            sf.write(d / "stems" / f"{n}.wav", np.clip(stems[n].T, -1, 1), SR, subtype="PCM_16")

        set_status(song_id, "Analizando tempo, tonalidad y acordes", 0.9)
        result = analysis.analyze(audio, stems, SR)
        result["stems"] = names
        result["peaks"] = {n: analysis.peaks(stems[n]) for n in names}
        (d / "analysis.json").write_text(json.dumps(result))
        set_status(song_id, "Listo", 1.0)
    except Exception as e:
        traceback.print_exc()
        set_status(song_id, "Error", 0, f"{type(e).__name__}: {e}")


def worker():
    while True:
        song_id = jobs.get()
        process(song_id)


threading.Thread(target=worker, daemon=True).start()


def song_info(song_id):
    d = SONGS / song_id
    meta = json.loads((d / "meta.json").read_text())
    if (d / "analysis.json").exists():
        meta["status"] = {"stage": "Listo", "progress": 1.0, "error": None}
    else:
        meta["status"] = status.get(song_id) or {"stage": "Interrumpido", "progress": 0, "error": "El servidor se reinició antes de terminar"}
    return meta


def check_id(song_id):
    if not re.fullmatch(r"[0-9a-f]{12}", song_id) or not (SONGS / song_id / "meta.json").exists():
        raise HTTPException(404, "Canción no encontrada")


@app.get("/api/songs")
def list_songs():
    ids = [p.name for p in SONGS.iterdir() if (p / "meta.json").exists()]
    songs = [song_info(i) for i in ids]
    return sorted(songs, key=lambda s: s["created"], reverse=True)


@app.post("/api/songs")
async def upload(file: UploadFile = File(...), model: str = Form("4")):
    if model not in MODELS:
        raise HTTPException(400, "Modelo inválido")
    song_id = uuid.uuid4().hex[:12]
    d = SONGS / song_id
    d.mkdir()
    ext = Path(file.filename or "audio").suffix.lower()[:8] or ".audio"
    with open(d / f"original{ext}", "wb") as f:
        shutil.copyfileobj(file.file, f)
    title = Path(file.filename or "Sin título").stem
    meta = {"id": song_id, "title": title, "file": f"original{ext}", "model": model, "created": time.time()}
    (d / "meta.json").write_text(json.dumps(meta))
    set_status(song_id, "En cola", 0)
    jobs.put(song_id)
    return song_info(song_id)


@app.get("/api/songs/{song_id}")
def get_song(song_id: str):
    check_id(song_id)
    info = song_info(song_id)
    a = SONGS / song_id / "analysis.json"
    if a.exists():
        info["analysis"] = json.loads(a.read_text())
    return info


@app.delete("/api/songs/{song_id}")
def delete_song(song_id: str):
    check_id(song_id)
    if status.get(song_id, {}).get("stage") not in (None, "Listo", "Error"):
        raise HTTPException(409, "La canción todavía se está procesando")
    shutil.rmtree(SONGS / song_id)
    status.pop(song_id, None)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/files", StaticFiles(directory=SONGS), name="files")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
