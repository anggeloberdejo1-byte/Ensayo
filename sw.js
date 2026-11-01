/* Service Worker — Ensayo
 *
 * Hace que la app funcione sin conexión una vez abierta con internet.
 * Todo el HTML/CSS/JS está en index.html, así que basta con cachear ese
 * archivo, el manifest, el sello y los íconos. Los modelos de IA (models/ y los de
 * la letra en Hugging Face) no pasan por aquí: la app los guarda aparte.
 *
 * IMPORTANTE: cada vez que actualices index.html, sube también el número de
 * versión de CACHE (v1 -> v2 ...) para que los usuarios reciban la versión
 * nueva en vez de la vieja cacheada.
 */
const CACHE = "ensayo-v6";
const ARCHIVOS = [
  "./",
  "./index.html",
  "./manifest.json",
  "./icon-192.png",
  "./icon-512.png",
  "./apple-touch-icon.png",
  "./sello.webp"
];

// Instalación: guardar los archivos base en el caché.
self.addEventListener("install", (evento) => {
  evento.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(ARCHIVOS))
      .then(() => self.skipWaiting())
  );
});

// Activación: borrar cachés viejos de versiones anteriores
// (sin tocar "ensayo-modelos", donde vive el modelo de IA).
self.addEventListener("activate", (evento) => {
  evento.waitUntil(
    caches.keys().then((claves) =>
      Promise.all(claves.filter((k) => k !== CACHE && k !== "ensayo-modelos").map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Aislamiento: con estas cabeceras el navegador deja usar varios núcleos del
// procesador para la IA (3 a 4 veces más rápido en celulares sin GPU).
// GitHub Pages no permite ponerlas, así que las agrega el service worker.
function conAislamiento(respuesta) {
  if (!respuesta || respuesta.status === 0 || respuesta.type === "opaque") return respuesta;
  const cabeceras = new Headers(respuesta.headers);
  cabeceras.set("Cross-Origin-Opener-Policy", "same-origin");
  cabeceras.set("Cross-Origin-Embedder-Policy", "require-corp");
  return new Response(respuesta.body, { status: respuesta.status, statusText: respuesta.statusText, headers: cabeceras });
}

// Fetch: "network-first" (ves la versión más nueva si hay internet),
// con respaldo en el caché cuando estás sin conexión.
self.addEventListener("fetch", (evento) => {
  const req = evento.request;
  if (req.method !== "GET") return;
  // Los modelos (cientos de MB) los guarda la app; no duplicarlos aquí.
  if (req.url.includes("/models/") || /(^|\.)(huggingface\.co|hf\.co)$/.test(new URL(req.url).hostname)) return;

  const propia = new URL(req.url).origin === self.location.origin;
  evento.respondWith(
    fetch(req)
      .then((respuesta) => {
        const copia = respuesta.clone();
        caches.open(CACHE).then((cache) => cache.put(req, copia));
        return propia ? conAislamiento(respuesta) : respuesta;
      })
      .catch(() => caches.match(req).then((c) => c || caches.match("./index.html")).then((c) => (propia ? conAislamiento(c) : c)))
  );
});
