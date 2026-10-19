import time
import cv2
import requests
import numpy as np

class ClienteReco:
    def __init__(self, url, api_key, canal = None, timeout=3.0, calidad_jpeg=90):
        self.url = url.rstrip("/")
        self.canal = canal
        self.timeout = timeout
        self.calidad = calidad_jpeg
        self.sesion = requests.Session()
        self.sesion.headers.update({"X-API-Key": api_key})
        self._fallos = 0
        self._pausa_hasta = 0.0

    def recorte(self, box, tam, margen_x=0.3, prop_cabeza=0.45):
        if time.time() < self._pausa_hasta:
            return None
        cuerpo = {"canal": self.canal, "box": [float(v) for v in box[:4]],
                 "ancho": int(tam[0]), "alto": int(tam[1]),
                 "margen_x": margen_x, "prop_cabeza": prop_cabeza}
        try:
            r = self.sesion.post(f"{self.url}/recorte-hd", json=cuerpo, timeout=self.timeout)
            if r.status_code == 422:
                print(f"[cliente-capturaHD] detalle 422: {r.text}")
                return None
            r.raise_for_status()
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            self._fallos = 0
            return img
        except requests.RequestException as e:
            self._fallos += 1
            self._pausa_hasta = time.time() + min(0.5 * 2 ** self._fallos, 10.0)
            if self._fallos == 1 or self._fallos % 10 == 0 :
                print(f"[cliente-capturaHD] -> no hubo respuesta del servicio ({e})")
            return None

    def disponible(self):
        try:
            r = self.sesion.get(f"{self.url}/salud", timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def reconocer(self, img, modo="vivo", tid=None):
        if time.time() < self._pausa_hasta:
            return None
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.calidad])
        if not ok:
            return None
        params = {"modo": modo}
        if tid is not None:
            params["tid"] = tid
        try:
            r = self.sesion.post(f"{self.url}/reconocer", params=params, data=buf.tobytes(), timeout=self.timeout, headers={"Content-Type": "image/jpeg"})
            r.raise_for_status()
            if self._fallos:
                print("[cliente-reconocimiento] servidor de reconocimiento de vuelta")
            self._fallos = 0
            return r.json()
        except (requests.RequestException, ValueError) as e:
            self._fallos += 1
            espera = min(0.5 * 2 ** self._fallos, 10.0)
            self._pausa_hasta = time.time() + espera
            if self._fallos == 1 or self._fallos % 10 == 0:
                print(f"[cliente-reconocimiento] sin respuesta del servidor ({e})...")
            return None

class ClienteDesconocidos:
    def __init__(self, url, api_key, timeout=3.0):
        self.url = url.rstrip("/")
        # segundos máximos de espera para la respuesta HTTP antes de lanzar una excepción timeout.
        self.timeout = timeout
        # deja la conexión TCP abierta entre peticiones, permite fijar headers por defecto de una sola vez.
        self.sesion = requests.Session()
        # agrega este header a todas las peticiones.
        self.sesion.headers.update({"X-API-Key": api_key})
        # contador de fallos consecutivos.
        self._fallos = 0
        # marca de tiempo (timestamp) hasta la cual el cliente debe estar en pausa sin intentar peticiones nuevas
        self._pausa_hasta = 0.0

    def registrar(self, embedding, canal, ruta_crop=None):
        if time.time() < self._pausa_hasta:
            return None
        cuerpo = {"embedding": embedding, "canal": canal, "ruta_crop": ruta_crop}
        try:
            r = self.sesion.post(f"{self.url}/registrar-desconocido", json=cuerpo, timeout=self.timeout)
            r.raise_for_status()
            self._fallos = 0
            return r.json()
        except (requests.RequestException, ValueError) as e:
            self._fallos += 1
            self._pausa_hasta = time.time() + min (0.5 * 2 ** self._fallos, 10.0)
            if self._fallos == 1 or self._fallos % 10 == 0:
                print(f"[desconocidos] sin respuesta del servicio ({e})")
                return None
