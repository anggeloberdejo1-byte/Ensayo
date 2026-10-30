/* Service Worker — Ensayo
 *
 * Hace que la app funcione sin conexión una vez abierta con internet.
 * Todo el HTML/CSS/JS está en index.html, así que basta con cachear ese
 * archivo, el manifest y los íconos. El modelo de IA (models/) no pasa por
 * aquí: la app lo guarda aparte la primera vez que separa una canción.
 *
 * IMPORTANTE: cada vez que actualices index.html, sube también el número de
 * versión de CACHE (v1 -> v2 ...) para que los usuarios reciban la versión
 * nueva en vez de la vieja cacheada.
 */
const CACHE = "ensayo-v1";
const ARCHIVOS = [
  "./",
  "./index.html",
  "./manifest.json",
  "./icon-192.png",
  "./icon-512.png",
  "./apple-touch-icon.png"
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

// Fetch: "network-first" (ves la versión más nueva si hay internet),
// con respaldo en el caché cuando estás sin conexión.
self.addEventListener("fetch", (evento) => {
  const req = evento.request;
  if (req.method !== "GET") return;
  // El modelo (90 MB) lo maneja la app; no duplicarlo aquí.
  if (req.url.includes("/models/")) return;

  evento.respondWith(
    fetch(req)
      .then((respuesta) => {
        const copia = respuesta.clone();
        caches.open(CACHE).then((cache) => cache.put(req, copia));
        return respuesta;
      })
      .catch(() => caches.match(req).then((c) => c || caches.match("./index.html")))
  );
});
