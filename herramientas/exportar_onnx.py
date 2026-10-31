"""Exporta el núcleo de un modelo Demucs (HTDemucs v4 o HDemucs v3) a ONNX para el navegador.

La STFT/iSTFT se hacen en JavaScript; el modelo recibe el espectrograma
(real/imag como canales) y el audio, y devuelve ambos ramales ya desnormalizados.

Uso:
  python exportar_onnx.py htdemucs            salida.onnx
  python exportar_onnx.py htdemucs_6s         salida.onnx
  python exportar_onnx.py drumsep:modelo.pt   salida.onnx   (pesos de github.com/inagoy/drumsep)
  python exportar_onnx.py drumsep:salida/modelo_salsa.pt salsa.onnx   (el modelo del Modo salsa)
"""
import sys
import torch

nombre = sys.argv[1]
salida = sys.argv[2]

if nombre.startswith("drumsep:"):
    from demucs.hdemucs import HDemucs
    paquete = torch.load(nombre.split(":", 1)[1], map_location="cpu", weights_only=True)
    modelo = HDemucs(*paquete["args"], **paquete["kwargs"])
    modelo.load_state_dict(paquete["state"])
    largo = 441000  # 10 s: HDemucs acepta cualquier largo
else:
    from demucs.pretrained import get_model
    modelo = get_model(nombre).models[0]
    largo = int(modelo.segment * modelo.samplerate)  # HTDemucs usa su largo de entrenamiento
modelo.eval()
es_v4 = hasattr(modelo, "crosstransformer")

# ONNX Runtime Web no tiene EyeLike: con largo fijo, la matriz identidad de la
# atención local (LocalState) puede ir como constante.
_eye = torch.eye
torch.eye = lambda n, *a, **k: _eye(int(n), *a, **k)


class Nucleo(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, mix, mag):
        m = self.m
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)
        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        if es_v4:
            if m.crosstransformer:
                if m.bottom_channels:
                    b, c, f, t = x.shape
                    x = m.channel_upsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t)
                    xt = m.channel_upsampler_t(xt)
                x, xt = m.crosstransformer(x, xt)
                if m.bottom_channels:
                    b, c, f, t = x.shape
                    x = m.channel_downsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t)
                    xt = m.channel_downsampler_t(xt)
        else:
            # HDemucs v3 arranca el decodificador en cero (la señal pasa por los saltos)
            x = torch.zeros_like(x)
            xt = torch.zeros_like(x)
        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)
        S = len(m.sources)
        x = x.view(B, S, -1, Fq, T) * std[:, None] + mean[:, None]
        xt = xt.view(B, S, -1, largo) * stdt[:, None] + meant[:, None]
        return x, xt


nucleo = Nucleo(modelo).eval()
torch.manual_seed(0)
mix = torch.randn(1, 2, largo) * 0.1
z = modelo._spec(mix)
mag = modelo._magnitude(z)
print("largo", largo, "mag", tuple(mag.shape), "fuentes", modelo.sources)

# Comprobar que el núcleo da lo mismo que el modelo completo
with torch.no_grad():
    ref = modelo(mix)
    x, xt = nucleo(mix, mag)
    completo = xt + modelo._ispec(modelo._mask(z, x), largo)
print("diferencia máxima con demucs:", float((completo - ref).abs().max()))

torch.onnx.export(nucleo, (mix, mag), salida, input_names=["mix", "mag"], output_names=["spec", "wave"],
                  opset_version=17, dynamo=False)
print("exportado", salida)
