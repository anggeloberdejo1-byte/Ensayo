"""Entrena el modelo de percusión del Modo salsa.

Parte del modelo de batería drumsep (HDemucs) y lo reentrena con las clases de
`preparar.py` (tumbadoras, bongo, timbal, campana…). Cada ejemplo es una mezcla
al azar de pedazos de distintos instrumentos, con cambios de volumen, tono,
paneo y ecualización, para que aprenda a reconocerlos dentro de cualquier canción.

Uso:
  python entrenar.py datos/ drumsep.pt salida/ [--pasos 20000] [--seguir]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from demucs.hdemucs import HDemucs
from scipy.signal import lfilter

SR = 44100
SEGMENTO = 6 * SR


class Mezclador:
    """Arma mezclas al azar a partir de los pedazos de cada instrumento."""

    def __init__(self, carpeta, lista, clases, semilla=None):
        self.carpeta = Path(carpeta)
        self.clases = clases
        self.lista = {c: lista.get(c, []) for c in clases}
        self.rng = random.Random(semilla)
        self.cache = {}

    def pedazo(self, nombre):
        if nombre not in self.cache:
            if len(self.cache) > 3000:
                self.cache.pop(next(iter(self.cache)))
            self.cache[nombre] = np.load(self.carpeta / nombre).astype(np.float32)
        return self.cache[nombre]

    def fuente(self, clase):
        rng = self.rng
        audio = self.pedazo(rng.choice(self.lista[clase]))
        # Velocidad y tono juntos (±8 %), como tocar un poco más rápido o con otra afinación
        factor = rng.uniform(0.92, 1.08)
        necesita = int(SEGMENTO * factor) + 2
        if audio.shape[1] > necesita:
            ini = rng.randrange(audio.shape[1] - necesita)
            audio = audio[:, ini: ini + necesita]
        x_viejo = np.arange(audio.shape[1])
        x_nuevo = np.arange(SEGMENTO) * factor
        audio = np.stack([np.interp(x_nuevo, x_viejo, canal, right=0) for canal in audio])
        # Ecualización simple: a veces más opaco, a veces más brillante
        if rng.random() < 0.6:
            a = rng.uniform(0.05, 0.6)
            filtrado = lfilter([a], [1, a - 1], audio, axis=1)  # pasa-bajos de un polo
            audio = filtrado if rng.random() < 0.5 else audio + rng.uniform(0.3, 1.0) * (audio - filtrado)
        # Volumen, paneo y polaridad
        audio = audio * 10 ** (rng.uniform(-12, 2) / 20)
        pan = rng.uniform(-0.6, 0.6)
        audio = audio * np.array([[1 - pan], [1 + pan]]) / (1 + abs(pan))
        if rng.random() < 0.5:
            audio = -audio
        return audio.astype(np.float32)

    def ejemplo(self):
        objetivos = np.zeros((len(self.clases), 2, SEGMENTO), dtype=np.float32)
        presentes = [c for c in self.clases if self.lista[c] and self.rng.random() < 0.75]
        if not presentes:
            presentes = [self.rng.choice([c for c in self.clases if self.lista[c]])]
        for c in presentes:
            objetivos[self.clases.index(c)] = self.fuente(c)
        mezcla = objetivos.sum(0)
        pico = np.abs(mezcla).max()
        if pico > 0.99:
            objetivos /= pico / 0.99
            mezcla /= pico / 0.99
        return mezcla, objetivos

    def lote(self, n):
        ejemplos = [self.ejemplo() for _ in range(n)]
        return (torch.from_numpy(np.stack([e[0] for e in ejemplos])),
                torch.from_numpy(np.stack([e[1] for e in ejemplos])))


def crear_modelo(base, clases):
    """HDemucs con las clases nuevas, con los pesos de drumsep donde la forma coincide."""
    paquete = torch.load(base, map_location="cpu", weights_only=True)
    kwargs = dict(paquete["kwargs"], sources=clases, segment=SEGMENTO / SR)
    modelo = HDemucs(**kwargs)
    propio = modelo.state_dict()
    copiados = {k: v for k, v in paquete["state"].items() if k in propio and propio[k].shape == v.shape}
    modelo.load_state_dict(copiados, strict=False)
    print(f"pesos de drumsep reutilizados: {len(copiados)} de {len(propio)} (las capas finales se aprenden de cero)")
    return modelo, kwargs


def sdr(estimado, real):
    num = (real ** 2).sum(dim=(-1, -2))
    den = ((real - estimado) ** 2).sum(dim=(-1, -2))
    return 10 * torch.log10((num + 1e-8) / (den + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datos")
    ap.add_argument("base", help="pesos de drumsep (formato {args, kwargs, state})")
    ap.add_argument("salida")
    ap.add_argument("--pasos", type=int, default=20000)
    ap.add_argument("--lote", type=int, default=2)
    ap.add_argument("--aprendizaje", type=float, default=3e-4)
    ap.add_argument("--evaluar", type=int, default=500, help="cada cuántos pasos medir y guardar")
    ap.add_argument("--seguir", action="store_true", help="continuar desde el último guardado")
    args = ap.parse_args()

    datos, salida = Path(args.datos), Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    resumen = json.loads((datos / "resumen.json").read_text())
    clases = resumen["clases"]
    print("clases:", clases)

    dispositivo = "mps" if torch.backends.mps.is_available() else "cpu"
    modelo, kwargs = crear_modelo(args.base, clases)
    paso_inicial = 0
    guardado = salida / "modelo_salsa.pt"
    if args.seguir and guardado.exists():
        paquete = torch.load(guardado, map_location="cpu", weights_only=True)
        modelo.load_state_dict(paquete["state"])
        paso_inicial = paquete.get("paso", 0)
        print("continuando desde el paso", paso_inicial)
    modelo.to(dispositivo).train()
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=args.aprendizaje, weight_decay=0.01)

    entrenar = Mezclador(datos, resumen["entrenar"], clases)
    # Mezclas de validación fijas, para comparar avances de forma justa
    validar = Mezclador(datos, resumen["validar"] if any(resumen["validar"].values()) else resumen["entrenar"], clases, semilla=1)
    lotes_validar = [validar.lote(args.lote) for _ in range(8)]

    historial = []
    t0 = time.time()
    for paso in range(paso_inicial + 1, args.pasos + 1):
        mezcla, objetivos = entrenar.lote(args.lote)
        estimado = modelo(mezcla.to(dispositivo))
        perdida = (estimado - objetivos.to(dispositivo)).abs().mean()
        optimizador.zero_grad()
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
        optimizador.step()
        if paso % 20 == 0:
            print(f"paso {paso}  pérdida {perdida.item():.4f}  {(time.time() - t0) / (paso - paso_inicial):.2f} s/paso", flush=True)
        if paso % args.evaluar == 0 or paso == args.pasos:
            modelo.eval()
            with torch.no_grad():
                valores = torch.cat([sdr(modelo(m.to(dispositivo)).cpu(), o).reshape(-1, len(clases)) for m, o in lotes_validar])
            # Solo cuenta las fuentes que sonaban (las mudas dan SDR sin sentido)
            activos = torch.cat([(o.abs().amax(dim=(-1, -2)) > 1e-3).reshape(-1, len(clases)) for _, o in lotes_validar])
            por_clase = {c: round(float(valores[activos[:, i], i].mean()), 2) if activos[:, i].any() else None for i, c in enumerate(clases)}
            historial.append({"paso": paso, "sdr": por_clase})
            print("validación (SDR en dB, más es mejor):", por_clase, flush=True)
            torch.save({"args": (), "kwargs": kwargs, "state": {k: v.cpu() for k, v in modelo.state_dict().items()}, "paso": paso}, guardado)
            (salida / "historial.json").write_text(json.dumps(historial, indent=1, ensure_ascii=False))
            modelo.train()
    print("listo:", guardado)


if __name__ == "__main__":
    main()
