"""Exporta el núcleo de HTDemucs a ONNX para correrlo en el navegador.

La STFT/iSTFT se hacen en JavaScript; el modelo recibe el espectrograma
(real/imag como canales) y el audio, y devuelve ambos ramales ya desnormalizados.
"""
import sys
import types
import torch
from demucs.pretrained import get_model

name = sys.argv[1] if len(sys.argv) > 1 else "htdemucs"
out = sys.argv[2] if len(sys.argv) > 2 else f"{name}-fp32.onnx"

bag = get_model(name)
model = bag.models[0]
model.eval()
L = int(model.segment * model.samplerate)


class Core(torch.nn.Module):
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
        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = m.channel_upsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = m.channel_downsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = m.channel_downsampler_t(xt)
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
        xt = xt.view(B, S, -1, L) * stdt[:, None] + meant[:, None]
        return x, xt


core = Core(model).eval()
mix = torch.randn(1, 2, L)
z = model._spec(mix)
mag = model._magnitude(z)
print("L", L, "mag", tuple(mag.shape), "sources", model.sources, "weights", list(bag.weights) if hasattr(bag, "weights") else None)

# Comprobar que el núcleo da lo mismo que el modelo completo
with torch.no_grad():
    ref = model(mix)
    x, xt = core(mix, mag)
    full = xt + model._ispec(model._mask(z, x), L)
print("max diff vs demucs:", float((full - ref).abs().max()))

torch.onnx.export(core, (mix, mag), out, input_names=["mix", "mag"], output_names=["spec", "wave"],
                  opset_version=17, dynamo=False)
print("exportado", out)
