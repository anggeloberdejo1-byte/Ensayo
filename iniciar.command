#!/bin/bash
# Doble clic para abrir Ensayo en el navegador.
# La primera vez instala lo necesario (unos minutos).
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Instalando Ensayo por primera vez…"
  python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt || exit 1
fi
(sleep 2 && open http://localhost:8765) &
.venv/bin/python -m uvicorn server:app --port 8765
