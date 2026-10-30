// Motor de audio (AudioWorklet): mezcla las pistas, cambia tempo/tono y toca el metrónomo.
// Tempo: WSOLA (superposición de ventanas alineadas por correlación). Tono: WSOLA + remuestreo.

const N = 2048;        // tamaño de ventana
const HS = N / 2;      // salto de síntesis
const SEARCH = 512;    // margen de búsqueda de alineación
const RING = 8192;
const MASK = RING - 1;

class Engine extends AudioWorkletProcessor {
  constructor() {
    super();
    this.stems = [];
    this.gains = [];
    this.len = 0;
    this.playing = false;
    this.pos = 0;           // posición en la canción original, en muestras
    this.tempo = 1;
    this.pitch = 1;
    this.loopA = -1;
    this.loopB = -1;
    this.beats = [];
    this.accents = [];
    this.nextBeat = 0;
    this.clickOn = false;
    this.clickVol = 0.5;
    this.clickT = -1;
    this.clickFreq = 1000;
    this.blocks = 0;

    this.win = new Float32Array(N);
    for (let i = 0; i < N; i++) this.win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
    this.acc = [new Float32Array(N), new Float32Array(N)];
    this.frame = [new Float32Array(N), new Float32Array(N)];
    this.ring = [new Float32Array(RING), new Float32Array(RING)];
    this.tpl = new Float32Array(HS);
    this.cand = new Float32Array(HS + 2 * SEARCH);
    this.one = [new Float32Array(1), new Float32Array(1)];

    this.port.onmessage = (e) => this.onMessage(e.data);
    this.reset();
  }

  onMessage(m) {
    switch (m.type) {
      case "load":
        this.stems = m.stems;
        this.len = m.len;
        this.gains = m.stems.map(() => 1 / 32768);
        this.playing = false;
        this.seekTo(0);
        break;
      case "gains": this.gains = m.gains; break;
      case "play": this.playing = true; break;
      case "pause": this.playing = false; break;
      case "seek": this.seekTo(Math.max(0, Math.min(this.len - 1, Math.round(m.time * sampleRate)))); break;
      case "rate":
        this.tempo = m.tempo;
        this.pitch = m.pitch;
        this.reset();
        break;
      case "loop":
        this.loopA = m.a == null ? -1 : Math.round(m.a * sampleRate);
        this.loopB = m.b == null ? -1 : Math.round(m.b * sampleRate);
        break;
      case "beats":
        this.beats = m.beats.map((t) => t * sampleRate);
        this.accents = m.accents;
        this.nextBeat = this.lowerBound(this.pos);
        break;
      case "click": this.clickOn = m.on; this.clickVol = m.vol; break;
    }
  }

  lowerBound(p) {
    let lo = 0, hi = this.beats.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.beats[mid] < p) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  seekTo(p) {
    this.pos = p;
    this.nextBeat = this.lowerBound(p);
    this.reset();
  }

  reset() {
    this.acc[0].fill(0);
    this.acc[1].fill(0);
    this.ringW = 0;
    this.ringR = 0;
    this.anaPos = this.pos;
    this.prevX = Math.round(this.pos) - HS;
  }

  // Suma las pistas en [start, start+count) dentro de L/R (que deben venir en cero)
  mixRange(start, count, L, R) {
    const lo = Math.max(0, -start), hi = Math.min(count, this.len - start);
    for (let k = 0; k < this.stems.length; k++) {
      const g = this.gains[k];
      if (!g) continue;
      const d = this.stems[k];
      for (let j = lo; j < hi; j++) {
        const i = (start + j) * 2;
        L[j] += d[i] * g;
        R[j] += d[i + 1] * g;
      }
    }
  }

  mixMono(start, count, out) {
    out.fill(0);
    const lo = Math.max(0, -start), hi = Math.min(count, this.len - start);
    for (let k = 0; k < this.stems.length; k++) {
      const g = this.gains[k];
      if (!g) continue;
      const d = this.stems[k];
      for (let j = lo; j < hi; j++) {
        const i = (start + j) * 2;
        out[j] += (d[i] + d[i + 1]) * g;
      }
    }
  }

  // Genera HS muestras más de la etapa de estiramiento de tiempo (WSOLA)
  wsolaFrame() {
    const nominal = Math.round(this.anaPos);
    // Buscar el inicio cuyo contenido continúa mejor lo que ya sonó
    this.mixMono(this.prevX + HS, HS, this.tpl);
    this.mixMono(nominal - SEARCH, HS + 2 * SEARCH, this.cand);
    let best = SEARCH, bestScore = -Infinity;
    for (let off = 0; off <= 2 * SEARCH; off += 4) {
      let dot = 0, en = 1e-9;
      for (let j = 0; j < HS; j += 4) {
        const c = this.cand[off + j];
        dot += this.tpl[j] * c;
        en += c * c;
      }
      const score = dot / Math.sqrt(en);
      if (score > bestScore) { bestScore = score; best = off; }
    }
    const x = nominal - SEARCH + best;

    const [fL, fR] = this.frame;
    fL.fill(0); fR.fill(0);
    this.mixRange(x, N, fL, fR);
    const [aL, aR] = this.acc;
    for (let j = 0; j < N; j++) {
      aL[j] += fL[j] * this.win[j];
      aR[j] += fR[j] * this.win[j];
    }
    const [rL, rR] = this.ring;
    for (let j = 0; j < HS; j++) {
      rL[(this.ringW + j) & MASK] = aL[j];
      rR[(this.ringW + j) & MASK] = aR[j];
    }
    this.ringW += HS;
    aL.copyWithin(0, HS); aL.fill(0, N - HS);
    aR.copyWithin(0, HS); aR.fill(0, N - HS);

    this.prevX = x;
    this.anaPos += (HS * this.tempo) / this.pitch;
  }

  process(inputs, outputs) {
    const out = outputs[0];
    const L = out[0], R = out[1] || out[0];
    const n = L.length;
    L.fill(0);
    if (R !== L) R.fill(0);
    if (!this.playing || !this.len) return true;

    const direct = this.tempo === 1 && this.pitch === 1;
    const one = this.one;
    for (let j = 0; j < n; j++) {
      if (this.loopB > this.loopA && this.loopA >= 0 && this.pos >= this.loopB) this.seekTo(this.loopA);
      if (this.pos >= this.len) {
        this.playing = false;
        this.seekTo(0);
        this.port.postMessage({ type: "ended" });
        break;
      }
      let l, r;
      if (direct) {
        one[0][0] = 0; one[1][0] = 0;
        this.mixRange(Math.floor(this.pos), 1, one[0], one[1]);
        l = one[0][0]; r = one[1][0];
      } else {
        while (this.ringW - Math.floor(this.ringR) < 3) this.wsolaFrame();
        const i = Math.floor(this.ringR), f = this.ringR - i;
        const [rL, rR] = this.ring;
        l = rL[i & MASK] * (1 - f) + rL[(i + 1) & MASK] * f;
        r = rR[i & MASK] * (1 - f) + rR[(i + 1) & MASK] * f;
        this.ringR += this.pitch;
      }

      while (this.nextBeat < this.beats.length && this.beats[this.nextBeat] <= this.pos) {
        if (this.clickOn) {
          this.clickT = 0;
          this.clickFreq = this.accents[this.nextBeat] ? 1760 : 1175;
        }
        this.nextBeat++;
      }
      if (this.clickT >= 0) {
        const t = this.clickT / sampleRate;
        const c = Math.sin(2 * Math.PI * this.clickFreq * t) * Math.exp(-t * 120) * this.clickVol;
        l += c; r += c;
        if (++this.clickT > sampleRate * 0.05) this.clickT = -1;
      }

      L[j] = l;
      R[j] = r;
      this.pos += direct ? 1 : this.tempo;
    }

    if (++this.blocks % 4 === 0) this.port.postMessage({ type: "pos", time: this.pos / sampleRate });
    return true;
  }
}

registerProcessor("engine", Engine);
