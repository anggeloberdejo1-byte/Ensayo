"""Análisis musical: pulsos, compases, BPM, tonalidad, acordes y secciones.

Usa las pistas separadas: la armonía se calcula sin voz ni batería, y el bajo ayuda a elegir la raíz.
"""
import numpy as np
import librosa
import scipy.cluster.hierarchy as sch

SR = 22050
HOP = 512

# Perfiles de Krumhansl-Kessler
MAJ_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

QUALITIES = {"maj": ([0, 4, 7], 1.0), "min": ([0, 3, 7], 1.0), "7": ([0, 4, 7, 10], 0.94), "m7": ([0, 3, 7, 10], 0.94)}


def mono(x, sr_in):
    return librosa.resample(np.asarray(x).mean(axis=0), orig_sr=sr_in, target_sr=SR)


def peaks(x, n=1500):
    m = np.abs(np.asarray(x)).max(axis=0)
    size = int(np.ceil(len(m) / n))
    m = np.pad(m, (0, size * n - len(m)))
    p = m.reshape(n, size).max(axis=1)
    return [int(v) for v in np.round(np.clip(p, 0, 1) * 100)]


def zscore(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)


def detect_key(chroma):
    c = chroma.mean(axis=1)
    best = (-2, 0, "maj")
    for k in range(12):
        for mode, prof in (("maj", MAJ_PROFILE), ("min", MIN_PROFILE)):
            r = np.corrcoef(c, np.roll(prof, k))[0, 1]
            if r > best[0]:
                best = (r, k, mode)
    return {"root": best[1], "mode": best[2], "confidence": round(float(best[0]), 3)}


def chord_templates():
    labels, temps, weights = [], [], []
    for q, (ivs, w) in QUALITIES.items():
        for root in range(12):
            t = np.zeros(12)
            for i in ivs:
                t[(root + i) % 12] = 1.0
            labels.append((root, q))
            temps.append(t / np.linalg.norm(t))
            weights.append(w)
    return labels, np.array(temps), np.array(weights)


def detect_chords(chroma_b, bass_b, energy_b, bounds):
    labels, temps, weights = chord_templates()
    c = chroma_b / (np.linalg.norm(chroma_b, axis=0, keepdims=True) + 1e-8)
    with np.errstate(all="ignore"):  # numpy en Apple Silicon avisa falsamente en matmul
        scores = (temps @ c) * weights[:, None]
    # La nota más fuerte del bajo favorece acordes con esa raíz
    b = bass_b / (bass_b.max(axis=0, keepdims=True) + 1e-8)
    roots = np.array([r for r, _ in labels])
    scores += 0.12 * b[roots, :]
    prob = np.exp(18 * scores)
    prob /= prob.sum(axis=0, keepdims=True)
    trans = librosa.sequence.transition_loop(len(labels), 0.7)
    path = librosa.sequence.viterbi_discriminative(prob, trans)

    quiet = energy_b < 0.08 * np.percentile(energy_b, 90)
    out = []
    for j, s in enumerate(path):
        root, q = (-1, "N") if quiet[j] else labels[s]
        start, end = float(bounds[j]), float(bounds[j + 1])
        if out and out[-1]["root"] == root and out[-1]["quality"] == q:
            out[-1]["end"] = end
        else:
            out.append({"start": start, "end": end, "root": root, "quality": q})
    return out


def detect_downbeats(beat_frames, onset):
    """Asume compás de 4/4 y elige la fase cuyo pulso tiene más golpe grave."""
    if len(beat_frames) < 8:
        return 0
    strength = onset[np.clip(beat_frames, 0, len(onset) - 1)]
    return int(np.argmax([strength[p::4].mean() for p in range(4)]))


def name_sections(segs):
    """Pone nombres aproximados a las secciones según repetición y energía."""
    from collections import Counter
    counts = Counter(s["label"] for s in segs)
    energy = {}
    for s in segs:
        energy.setdefault(s["label"], []).append(s["energy"])
    repeated = [l for l, c in counts.items() if c > 1]
    chorus = max(repeated, key=lambda l: np.mean(energy[l]), default=None)
    verse_opts = [l for l in repeated if l != chorus]
    verse = max(verse_opts, key=lambda l: counts[l], default=None)
    for i, s in enumerate(segs):
        l = s["label"]
        if l == chorus:
            s["name"] = "Coro"
        elif l == verse:
            s["name"] = "Estrofa"
        elif i == 0 and counts[l] == 1:
            s["name"] = "Intro"
        elif i == len(segs) - 1 and counts[l] == 1:
            s["name"] = "Final"
        else:
            s["name"] = "Puente" if counts[l] == 1 else f"Parte {l}"
    return segs


def novelty(F, L=4):
    """Curva de novedad de Foote: kernel de tablero de ajedrez sobre la matriz de autosimilitud."""
    X = F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-8)
    S = X.T @ X
    g = np.exp(-0.5 * (np.linspace(-1.5, 1.5, 2 * L)) ** 2)
    K = np.outer(g, g) * np.outer(np.r_[-np.ones(L), np.ones(L)], np.r_[-np.ones(L), np.ones(L)])
    Sp = np.pad(S, L, mode="edge")
    return np.array([(K * Sp[i:i + 2 * L, i:i + 2 * L]).sum() for i in range(S.shape[0])])


def detect_sections(feat_b, energy_b, beat_times, downbeat_phase, duration):
    """feat_b: rasgos por pulso. Se agrupan por compás, se buscan cambios y se etiquetan las repeticiones."""
    n = feat_b.shape[1]
    downs = list(range(downbeat_phase, n, 4))
    if len(downs) < 8:
        return [{"start": 0.0, "end": duration, "label": "A", "name": "Canción", "energy": 1.0}]
    bar_edges = [0] + downs[1:] + [n]  # la anacrusa se une al primer compás
    bars = np.array([feat_b[:, a:b].mean(axis=1) for a, b in zip(bar_edges[:-1], bar_edges[1:])]).T
    bars = np.clip(zscore(bars), -3, 3)  # que el silencio del final no domine
    nov = novelty(bars)
    nov = (nov - nov.mean()) / (nov.std() + 1e-8)
    # Picos de novedad separados al menos 4 compases
    cand = [i for i in range(2, len(nov) - 1) if nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1] and nov[i] > 0.3]
    cuts = []
    for i in sorted(cand, key=lambda i: -nov[i]):
        if all(abs(i - c) >= 4 for c in cuts) and i >= 4 and len(bars.T) - i >= 4:
            cuts.append(i)
    bar_cuts = [0] + sorted(cuts) + [bars.shape[1]]
    means = np.array([bars[:, a:b].mean(axis=1) for a, b in zip(bar_cuts[:-1], bar_cuts[1:])])
    if len(means) > 1:
        z = sch.linkage(means, method="average", metric="euclidean")
        groups = sch.fcluster(z, t=0.45 * z[:, 2].max(), criterion="distance")
    else:
        groups = [1]
    letters, segs = {}, []
    e_ref = energy_b.max() + 1e-8
    for (a, b), g in zip(zip(bar_cuts[:-1], bar_cuts[1:]), groups):
        letters.setdefault(g, "ABCDEFGHIJKLMNOP"[len(letters)])
        ba, bb = bar_edges[a], bar_edges[b]
        start = 0.0 if ba == 0 else float(beat_times[ba])
        end = duration if bb >= n else float(beat_times[bb])
        segs.append({"start": start, "end": end, "label": letters[g],
                     "energy": round(float(energy_b[ba:bb].mean() / e_ref), 3)})
    return name_sections(segs)


def analyze(audio, stems, sr_in):
    y = mono(audio, sr_in)
    duration = len(y) / SR
    harmonic_src = sum(stems[n] for n in stems if n not in ("drums", "vocals"))
    yh = mono(harmonic_src, sr_in)
    ybass = mono(stems["bass"], sr_in)
    yperc = mono(stems["drums"] + stems["bass"], sr_in)

    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP)
    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_times = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)
    # Pendiente de la recta pulso->tiempo: evita el redondeo por cuadro de la mediana
    bpm = float(60 / np.polyfit(np.arange(len(beat_times)), beat_times, 1)[0]) if len(beat_times) > 2 else 0.0

    low_onset = librosa.onset.onset_strength(y=yperc, sr=SR, hop_length=HOP, fmax=200)
    phase = detect_downbeats(beat_frames, low_onset)

    chroma = librosa.feature.chroma_cqt(y=yh, sr=SR, hop_length=HOP)
    bass_chroma = librosa.feature.chroma_cqt(y=ybass, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("E1"), n_octaves=4)
    rms = librosa.feature.rms(y=yh, hop_length=HOP)[0]
    rms_full = librosa.feature.rms(y=y, hop_length=HOP)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=14, hop_length=HOP)

    T = min(chroma.shape[1], bass_chroma.shape[1], len(rms), mfcc.shape[1])
    keep = (beat_frames > 0) & (beat_frames < T)
    beat_frames, beat_times = beat_frames[keep], beat_times[keep]
    idx = librosa.util.fix_frames(beat_frames, x_min=0, x_max=T)
    bounds = librosa.frames_to_time(idx, sr=SR, hop_length=HOP)
    bounds[-1] = duration
    sync = lambda X: librosa.util.sync(X[..., :T], idx, aggregate=np.median, pad=False)
    chroma_b = sync(chroma)
    bass_b = sync(bass_chroma)
    energy_b = sync(rms[None])[0]
    energy_full_b = sync(rms_full[None])[0]
    mfcc_b = librosa.util.sync(mfcc[:, :T], idx, aggregate=np.mean, pad=False)

    key = detect_key(chroma)
    chords = detect_chords(chroma_b, bass_b, energy_b, bounds)
    # Energía de cada pista separada: entra la voz, la batería sube en el coro, etc.
    stem_energy = []
    for name in stems:
        r = librosa.feature.rms(y=mono(stems[name], sr_in), hop_length=HOP)[0]
        stem_energy.append(np.log(r[:T] + 1e-4))
    stem_b = librosa.util.sync(np.array(stem_energy), idx, aggregate=np.mean, pad=False)
    feat = np.nan_to_num(np.vstack([2.0 * zscore(stem_b), 0.7 * zscore(mfcc_b[1:]), 0.3 * zscore(chroma_b)]))
    # El primer bloque de sync es lo que hay antes del primer pulso; los índices de sección van por pulso
    sections = detect_sections(feat[:, 1:], energy_full_b[1:], beat_times, phase, duration)

    return {
        "duration": duration,
        "bpm": round(bpm, 1),
        "key": key,
        "beats": [round(float(t), 3) for t in beat_times],
        "downbeat_phase": phase,
        "chords": chords,
        "sections": sections,
    }
