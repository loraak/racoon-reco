import json
import re
import subprocess
import threading
import time
from collections import deque
import numpy as np


# Este archivo es para hacer una reconexión automática, vigilancia de congelamientos.

# Uso en otros archivos:
# fuente = FuenteVideo(STREAMING, escala=(W, H))
# fuente.start()
# fuente, fid = fuente.ultimo()

# Se abren 3 hilos -> lectura, vigilante y ultimo.

# evita que los logs impriman todo. Busca lo que hay después de // y antes de @
def _ocultar_credenciales(texto):
    return re.sub(r"(?<=//)[^/@\s]+@", "***@", str(texto))

# estudiando Python mientras xd __init__ es el constructor de la clase.
# Python lo ejecuta automáticamente cada vez que crea un objeto.
# Por ejemplo -> fuente = FuenteVideo(STREAMING, escala=(W,H)) -> Python hace dos cosas: crea un objeto vacío y llama a FuenteVideo.__init__(ese_objeto, STREAMING, escala)(640, 360))
# self es el objeto concreto ene lq ue se está trabajando. Primera cosa que recibe cualquier método de una clase.
class FuenteVideo:
    # fuente -> URL o ruta a un archivo.
    # reconectar puede quitarse -> es por sí hay un video (para que no intente reconectarse al video xd)
    # espera_reconexion -> segundos entre intentos.
    # timeout_sinframes -> segundos sin recibir imagen antes de considerar que el stream se congeló.
    # escala -> para que FFMpeg reescale.
    # fps -> limita los cuadros por segundo.
    # hwaccel -> aceleración de decodificación -> "Forma de hacer que otra parte del hardware decodifique el video, en vez de CPU"
    # timeout_sondeo -> tiempo límite para que ffprobe responda.
    # ver_errores_ffmpg -> muestra o silencia los errores de ffmpeg
    # nombre -> etiqueta para ver los logs.

    # cosas de estado interno...
    # _lock -> protegerá las variables compartidas entre hilos.
    # _frame _fid -> último frame y su contador.
    # _t_ultimo -> momento en el que llega el último frame.
    # _ancho, _alto -> resolución real.
    # _marcas: deque -> últimas marcas de horas de los frames para calcular FPS.
    # _conectado, _terminado -> banderas de estado
    # _reconexiones -> contador de las reconexiones.
    # _proc -> el proceso de ffmpeg actual. Literal el que crea el proceso.
    # _parar -> evento que funciona como interruptor de apagado.
    # _hilo y _vigilante son las referencias a los hilos.
    def __init__(self, fuente, *, reconectar=None, espera_reconexion=3.0, timeout_sin_frames=10.0, escala=None, fps=None, hwaccel=None, timeout_sondeo=15.0, ver_errores_ffmpeg=False, nombre="video"):
        self.fuente = str(fuente)
        self.es_rtsp = self.fuente.lower().startswith("rtsp")
        self.reconectar = self.es_rtsp if reconectar is None else reconectar
        self.espera_reconexion = espera_reconexion
        self.timeout_sin_frames = timeout_sin_frames
        self.escala = tuple(escala) if escala else None
        self.fps = fps
        self.hwaccel = hwaccel
        self.timeout_sondeo = timeout_sondeo
        self.ver_errores_ffmpeg = ver_errores_ffmpeg
        self.nombre = nombre
        self._lock = threading.Lock()
        self._frame = None
        self._fid = 0
        self._t_ultimo = 0.0
        self._ancho = 0
        self._alto = 0
        self._marcas = deque(maxlen=60)
        self._conectado = False
        self._terminado = False
        self._reconexiones = 0
        self._proc = None
        self._parar = threading.Event()
        self._hilo = None
        self._vigilante = None

    def start(self, esperar=True, timeout=30.0):
        # si hilo ya está corriendo, no hace otro hilo.
        if self._hilo is not None and self._hilo.is_alive():
            return self.conectado
        # reinicia banderas, así arranca después de un stop.
        # crea el hilo lector (_correr) y si timeout_sin_frames es distinto de 0 crea el vigilante.
        # daemon = true -> indica que ese hilo debe hacerse en segundo plano y morir automáticamente cuando el hilo principal termine.
        self._parar.clear()
        self._terminado = False
        self._hilo = threading.Thread(target=self._correr, name="FuenteVideo", daemon=True)
        self._hilo.start()
        # por defecto bloeque hasta que llegue el primer frame o hasta que pasen 30 segundos. Si el hilo muere antes, sale de inmediato.
        if self.timeout_sin_frames:
            self._vigilante = threading.Thread(target=self._vigilar, name="FuenteVideo-vigilante", daemon=True)
            self._vigilante.start()
        if not esperar:
            return True
        limite = time.monotonic() + timeout
        while time.monotonic() < limite and not self._terminado:
            if self.conectado:
                return True
            time.sleep(0.05)
        return self.conectado

    # activa _parar, toma el proceso bajo el candado y hace kill(). Necesario para matarlo bien xd -> espera 5s a que el hilo termine.
    def stop(self):
        self._parar.set()
        with self._lock:
            proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        if self._hilo is not None:
            self._hilo.join(timeout=5)

    # devuelve (frame, fid) bajo el candado. No hace falta hacer copia ya que cada uno es un arreglo nuevo.
    def ultimo(self):
        with self._lock:
            return self._frame, self._fid

    #@property -> "La forma pythonica" xd de trabajar con atributos y POO.

    # si el hilo no terminó y nadie pide parar, sirve para unas banderas del deteccion-worker.
    @property
    def activa(self):
        return not self._terminado and not self._parar.is_set()

    # si hay frames recibiendose.
    @property
    def conectado(self):
        with self._lock:
            return self._conectado

    # (ancho y alto).
    @property
    def resolucion(self):
        with self._lock:
            return self._ancho, self._alto

    # ¿cuánto se ha reconectado?
    @property
    def reconexiones(self):
        return self._reconexiones

    # fps reales de llegada. Toma 60 marcas. Si hay menos de 2 , o la última tiene más de 2ss, devuelve 0. Si no, calcula (n-1) / (-ultima-primera).
    @property
    def fps_recibidos(self):
        with self._lock:
            marcas = list(self._marcas)
        if len(marcas) < 2 or time.monotonic() - marcas[-1] > 2.0:
            return 0.0
        dt = marcas[-1] - marcas[0]
        return (len(marcas) - 1) / dt if dt > 0 else 0.0

    # tiempo desde el último frame.
    @property
    def segundos_sin_frame(self):
        with self._lock:
            if self._fid == 0:
                return float("inf")
            return time.monotonic() - self._t_ultimo

    # En Python, cuando se escribe with ... as variable, Python busca en ese objeto dos métodos con nombres exactos.
    # __enter__ y __exit__. Si existen, los llama solo en momentos precisos.
    # equivale a:
    # obj = FuenteVideo(Streaming)
    # fuente = obj.__enter__()
    # try:
    # ...
    # finally:
    # obj.__exit__(...)
    # aquí arranca los hilos y la conexión (start()).
    # devuelve sef, el propio objeto para que as fuente recib ala fuente de video.
    def __enter__(self):
        self.start()
        return self

    # llama a stop. Mata a FFmpeg y espera los hilos.
    def __exit__(self, *_):
        self.stop()

    # imprime [nombre] mensaje.
    def _log(self, msg):
        print(f"[{self.nombre}] {_ocultar_credenciales(msg)}", flush=True)

    # arma el comando -> [v:0] solo el primer stream de video, la salida va en JSON.
    # Si esto no devuelve nada, se lanza un error con el mensaje. El error es atrapado por _correr y provoca un reintento.
    # sirve para averiguar el ancho y alto del stream.
    def _sondear(self):
        cmd = ["ffprobe", "-v", "error"]
        if self.es_rtsp:
            cmd += ["-rtsp_transport", "tcp"]
        cmd += ["-select_streams", "v:0", "-show_entries", "stream=width,height",
                "-of", "json", self.fuente]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=self.timeout_sondeo)
        streams = (json.loads(r.stdout or "{}")).get("streams") or []
        if not streams:
            raise RuntimeError(r.stderr.strip() or "ffprobe no encontró ningún stream de video")
        return int(streams[0]["width"]), int(streams[0]["height"])

    # arma el comando chido para conectarse.
    # -hide_banner, log-level solo muestra errores.
    # si es rtsp -> minima latencia y TCP...
    # i fuente es la entrada.
    # filtros, si no hay se omiten.
    # tiene una salida cruda, en BGR (el orden de opencv).
    def _comando(self):
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if self.hwaccel:
            cmd += ["-hwaccel", self.hwaccel]
        if self.es_rtsp:
            cmd += ["-fflags", "nobuffer", "-flags", "low_delay", "-rtsp_transport", "tcp"]
        else:
            cmd += ["-re"]
        cmd += ["-i", self.fuente]
        filtros = []
        if self.fps:
            filtros.append(f"fps={self.fps}")
        if self.escala:
            filtros.append(f"scale={self.escala[0]}:{self.escala[1]}")
        if filtros:
            cmd += ["-vf", ",".join(filtros)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-"]
        return cmd

    # apaga el processo haciendo 3 acciones: kill, cerrar stdout y wait(timeout=3).
    def _cerrar(self, proc):
        if proc is None:
            return
        for accion in (proc.kill, proc.stdout.close, lambda: proc.wait(timeout=3)):
            try:
                accion()
            except Exception:
                pass
        with self._lock:
            if self._proc is proc:
                self._proc = None

    # el hilo principal de la clase.
    # bucle de conexión -> lectura -> limpieza -> espera -> reintento.
    def _correr(self):
        try:
            while not self._parar.is_set():
                proc = None
                try:
                    # ve la resolución. Si hay escala, la que cuenta es la salida de ffmpeg, no la original.
                    ancho, alto = self._sondear()
                    if self.escala:
                        ancho, alto = self.escala
                    # lanza ffmpeg, bufsize=0 quita el búfer de python.
                    proc = subprocess.Popen(
                        self._comando(), stdout=subprocess.PIPE,
                        stderr=None if self.ver_errores_ffmpeg else subprocess.DEVNULL, bufsize=0)
                    # registra proceso y resolución. Da al vigilante un margen antes de considerar que no llegan frames.
                    with self._lock:
                        self._proc = proc
                        self._ancho, self._alto = ancho, alto
                        self._t_ultimo = time.monotonic()
                        self._marcas.clear()
                    self._leer(proc, ancho, alto)
                    # se queda bloqueado leyendo hasta que el stream se corte.
                except subprocess.TimeoutExpired:
                    self._log("ffprobe no respondió a tiemp")
                except FileNotFoundError:
                    self._log("no se encontró ffmpeg/ffprobe")
                    return
                except Exception as e:
                    self._log(f"error: {e}")
                finally:
                    self._cerrar(proc)
                    with self._lock:
                        self._conectado = False

                # si se pide parar, se sale.
                if self._parar.is_set():
                    break
                # si no hay reconexión, termina.
                if not self.reconectar:
                    self._log("fin de video")
                    break
                # si sí, cuenta la reconexión, avisa y espera 3s.
                self._reconexiones += 1
                self._log(f"sin señal; reintentando en {self.espera_reconexion:.0f} s"
                          f"(reconexión #{self._reconexiones})")
                self._parar.wait(self.espera_reconexion)
        finally:
            self._terminado = True

    #
    def _leer(self, proc, ancho, alto):
        # bytes por frame y una bandera para loguear solo el primer frame.
        tam = ancho * alto * 3
        primero = True
        # crea un arreglo vacío y una vista en bytes para que FFmpeg escriba ahí.
        # repite readinto hasta completar tam.
        # con el frame completo y bajo candado, publica el frame, incrmeenta _fid.
        # _conectado se activa solo cuando llega un frame real, no cuando arranca el proceso.
        while not self._parar.is_set():
            frame = np.empty((alto, ancho, 3), np.uint8)
            mv = memoryview(frame).cast("B")
            leidos = 0
            while leidos < tam:
                n = proc.stdout.readinto(mv[leidos:])
                if not n:
                    return
                leidos += n
            ahora = time.monotonic()
            with self._lock:
                self._frame = frame
                self._fid += 1
                self._t_ultimo = ahora
                self._conectado = True
                self._marcas.append(ahora)
            if primero:
                primero = False
                self._log(f"conectado: {ancho}x{alto}")

    def _vigilar(self):
        # cada segundo revisa.
        # toma el proceso actual y calcula cuánto lleva sin frames. Si hay prceso y pasó el timeout:
        # loguea y hace kill() a ffmpeg, eso hace que reainto regrese 0 en _leer, que devuelve el control a _correr, y reconecta.
        while not self._parar.wait(1.0):
            with self._lock:
                proc = self._proc
                sin_frames = time.monotonic() - self._t_ultimo
            if proc is not None and sin_frames > self.timeout_sin_frames:
                self._log(f"sin frames desde hace {sin_frames:.0f} s: se reinicia")
                try:
                    proc.kill()
                except Exception:
                    pass
                with self._lock:
                    self._t_ultimo = time.monotonic()



