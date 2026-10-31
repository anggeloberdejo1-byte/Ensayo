# Guía para grabar la percusión del Modo salsa

La IA aprende a reconocer cada instrumento escuchándolo **solo**, muchas veces y de muchas formas.
Después yo lo mezclo de miles de maneras para que aprenda a encontrarlo dentro de una salsa.
Mientras más variado grabes, mejor separa.

## Qué grabar

Una carpeta por instrumento. El nombre de la carpeta es el nombre que tendrá la pista en Ensayo.

| Carpeta | Qué va adentro | Mínimo | Ideal |
|---|---|---|---|
| `tumbadoras` | Conga, tumba y quinto: tumbao, guaguancó, mambo, abierto, slap, manoteo, rellenos | 15 min | 40 min |
| `bongo` | Martillo, rellenos, repiques (sin la campana) | 10 min | 25 min |
| `timbal` | Cueros, cáscara, rim, abanicos, rellenos (sin las campanas) | 10 min | 25 min |
| `campana` | Campana de bongó (mambo), campana de timbal, bloque | 10 min | 20 min |
| `guiro_maracas` | Güiro, maracas, claves, chekeré | 5 min | 15 min |
| `bateria` | Cajón y batería (si tienes), cualquier ritmo | 10 min | 25 min |

## Cómo grabar

- **Un instrumento por archivo, sin nada más sonando.** Nada de música de fondo. Si usas metrónomo, que sea con audífonos para que no se grabe.
- **Con el celular sirve.** Ponlo a 1 o 2 metros, en un cuarto sin mucho eco. Si tienes micrófono e interfaz, mejor.
- **Varía todo:** tempos lentos y rápidos (de 80 a 220 BPM), fuerte y suave, distintas afinaciones,
  distintos tambores si tienes, distintos cuartos, sentado y parado. La variedad es lo que más ayuda.
- **Archivos de 1 a 10 minutos.** No importa el formato: wav, m4a, mp3 o el de las notas de voz.
- **Nombra los archivos como quieras**; lo que importa es en qué carpeta están.
- Los silencios al principio y al final no molestan: el programa los recorta solo.

## Dónde ponerlas

```
Ensayo-grabaciones/
  tumbadoras/
  bongo/
  timbal/
  campana/
  guiro_maracas/
  bateria/
```

Si un instrumento no lo tienes, deja la carpeta vacía o bórrala: el modelo se arma con las carpetas que tengan audio.

## Qué pasa después

1. `preparar.py` recorta silencios, iguala volúmenes y corta todo en pedazos de 10 segundos.
2. `entrenar.py` arma mezclas al azar de esos pedazos (con cambios de volumen, tono y ecualización)
   y entrena el modelo partiendo del modelo de batería que ya usa Ensayo.
3. `exportar_onnx.py` lo convierte para el navegador y se agrega a Ensayo como **Modo salsa**.
