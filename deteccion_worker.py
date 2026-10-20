import os
import queue
import threading
import time
from collections import Counter, deque
import cv2
import uvicorn

# nuevos imports para crearle una URL a este streaming ¿?
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

from cliente_reconocimiento import ClienteReco, ClienteDesconocidos
from fuente_video import FuenteVideo
from utils.segmentacion import cuadro_cabeza

# COSAS PA' QUITAR DSPS.
CARPETA = "capturas_debug_capturasHD"
os.makedirs(CARPETA, exist_ok=True)

from dotenv import load_dotenv
import os

load_dotenv()

VIDEO = False
MAIN_CHANNEL = int(os.getenv("MAIN_CHANNEL", 101))
NVR_USER = os.getenv("NVR_USER_POES")
NVR_PASS = os.getenv("NVR_PASS_POES")
NVR_IP = os.getenv("NVR_IP_POES")
STREAMING = f"rtsp://{NVR_USER}:{NVR_PASS}@{NVR_IP}:554/Streaming/Channels/{MAIN_CHANNEL}01"
RECO_URL = "http://localhost:8000"
RECO_API_KEY = os.getenv("API_KEY_RECONOCIMIENTO")
HD_URL = os.getenv("URL_HD")
HD_API_KEY = os.getenv("API_KEY_HD")
#DESCONOCIDOS_URL = os.getenv("URL_DESCONOCIDOS")
#DESCONOCIDOS_API_KEY = os.getenv("API_KEY_DESCONOCIDOS")

CAP_ACTIVO = True
ESPERA_ENTRE_CAPS = 2.0
MODELO_PERSONAS = "yolov8n.pt"
DEVICE_PERSONAS = os.getenv("DEVICE")
IMGSZ = 640
TRACKER = "bytetrack.yaml"
CONF_TRACK = 0.4
CONF_BUSCAR_CARA = 0.5 # umbral para pedir la cara.
PERSONA = 0
FPS_WORKER = 25
MAX_CAP_INTENTOS = 3

# parámetros de reconocimiento.
PROP_CABEZA = 0.5
INTERVALO_BUSQUEDA = 0.5
INTERVALO_REVERIFICAR = 10.0
VENTANA_VOTOS = 7
VOTOS_CONFIRMAR = 3
OLVIDAR_SEG = 5.0
PESO_CAP = 3

# cosas streaming.
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8010
STREAM_FPS = 25
STREAM_CALIDAD = 50

# estado por persona, LO CHIDOOO.
personas = {}
personas_lock = threading.Lock()
cola_hd = queue.Queue(maxsize=4)
cola_reco = queue.Queue(maxsize=8)
cola_desconocidos = queue.Queue(maxsize=4)
latest_dib = []
dib_lock = threading.Lock()
running = True
cliente = None
cliente_hd = None
#cliente_desc = None
fuente_global = None

CARPETA_DESCONOCIDOS = "capturas_desconocidos"
os.makedirs(CARPETA_DESCONOCIDOS, exist_ok=True)

def nuevo_estado():
    return {"votos": deque(maxlen=VENTANA_VOTOS), "nombre": "...", "score": 0.0,
            "confirmado": False, "pendiente": False, "ultimo_intento": 0.0, "visto": time.time(),
            "hd_ok": False, "hd_pendiente": False, "hd_intentos": 0, "hd_ultimo": 0.0,
            "desconocido_guardado": False,
            "mejor_embedding": None, "mejor_frontalidad": float("inf"), "mejor_crop": None, "mejor_calidad": None}

# PARA ENTENDERME. Proceso hasta llegar a este método.
# YOLO procesa frame, encuentra persona con tid = 17 + confianza 0.85 y no hay petición pendiente para ese tid.
# recorta cabeza del frame y mete a la cola -> cola_reco.put_nowait((17,crop)) -> st["pendiente"] = True
# reco_worker toma el crop y pide el reconocimiento -> tid, crop = cola_reco.get() > r = cliente.reconocer(crop, "vivo")
# el crop se codifica a JPEG y se manda por HTTP a la API de reconocimiento.
# La API procesa la imagen. Ahí dentro, procesar(img, modo) detecta la cara más grande, saca embedding y lo compara contra la base.
# API regresa un JSON y cliente.reconocer lo devuelve tal cual (ej. cara: true, nombre: juan, score:0.78, bbox:[], ms:10.0)
# si es un desconocido real la respuesta trae el vector completo (ya quitpe eso xd)
# agregar_voto (17, "Juan", 0.78, peso=1, origen="vivo") -> se agrega una vez al deque de votos del tid=17.
# se llama decidir_nombre(st), que agrupa por "Juan" (quitando un #2 dependiendo de qué devuelva xd), ve si ese grupo
# ya tiene VOTOS_CONFIRMAR votos acumulador entre varios frames, y si sí, fija st["nombre"] = "Juan#x" ( la variante más votada dentro del grupo).
# en paralelo -> proceso de petición a API.
def decidir_nombre(st):
    if not st["votos"]:
        return
    # toma un nombre como "Juan#2" y devuelve solo "Juan". Para tratar distintos embeddings registrados de la misma persona
    # como un solo grupo.
    base = lambda n: n.split("#")[0]
    # v[0] nombre de cada voto (cada voto es una tupla -> (nombre, score).
    # counter -> cuántos votos cayeron en cada grupo (persona base).
    # most_common -> toma el grupo más votado -> grupo es el nombre base ganador, n es cuántos votos tiene el grupo.
    grupo, n = Counter(base(v[0]) for v in st["votos"]).most_common(1)[0]
    # filtra solo los votos que pertenecen al grupo ganador.
    votos_grupo = [v for v in st["votos"] if base(v[0]) == grupo]
    # dentro del grupo, cuenta los nombres exactos (sin quitar #) y toma el más común.
    # esto recupera la variante específica más votada.
    nombre, _ = Counter(v[0] for v in votos_grupo).most_common(1)[0]
    # se confirma la identidad si el grupo tiene al menos VOTOS_CONFIRMAR votos.
    # n es del grupo (Juan, Juan#2, Juan, Juan#2, Desconocido) Juan -> 4 (n).
    st["confirmado"] = n >= VOTOS_CONFIRMAR
    # si está confirmado, guarda el nombre tal cual; si no, lo marca con un ?
    st["nombre"] = nombre if st["confirmado"] else f"{nombre}?"
    # guarda el score más alto entre los votos que coinciden con la variante elegida nombre dentro del grupo ganador.
    st["score"] = max(s for m, s in votos_grupo if m == nombre)

def agregar_voto(tid, nombre, score, embedding=None, frontalidad=None, crop=None, calidad=None, peso=1, origen="vivo"):
    with personas_lock:
        st = personas.get(tid)
        if st is None:
            return
        # se guarda el nombre que tenía antes, para detectar si cambió.
        anterior = st["nombre"]
        # range(peso) genera la secuencia de números. Si peso -> 3 -> 0, 1, 2.
        # _ solo sirve para repetir algo un número de veces.
        # entonces, peso = 3 es equivalente a escribir -> st["votos"].append((nombre, score)) x3
        # st["votos"] es un deque.
        # deque -> tiene dos propiedades que ayudan al procesamiento. maxlen (ventana deliszante automática) ->
        # con esto fijo, cuando ya está lleno y agregas un elemento nuevo por un lado, automáticamente elimina el más viejo
        # sin que se tenga que escribir un pop() cómo una lista.
        for _ in range(peso):
            st["votos"].append((nombre, score))
        # llama al método para recalcular nombre, confirmado y score con el voto ya incluido.
        decidir_nombre(st)
        # si el nombre resultante cambió respecto al que tenía antes, lo imprime en consola con el origen.
        if st["nombre"] != anterior:
            print(f"[{origen}] ID {tid}: {st['nombre']} (score {score:.3f})")
        if embedding is not None and frontalidad is not None and frontalidad < st["mejor_frontalidad"]:
            # están separados porque c/u tiene su propia comparación.
            st["mejor_embedding"] = embedding
            st["mejor_frontalidad"] = frontalidad
            # se va a tomar el crop con el mejor puntaje en frontalidad.
            # la condición es < porque va al revés ¿? o sea, ahorita si sale 1 significa que hay mucha discrepancia en un rostro frontal.
            st["mejor_crop"] = crop
            st["mejor_calidad"] = calidad

        # condición para guardar a un desconocido -> que ya esté confirmado, que el nombre sea literalmente "Desconocido".
        # que no se haya guardado para este track,.
        if (st["confirmado"] and st["nombre"] == "Desconocido"
                and not st["desconocido_guardado"]
                and st["mejor_embedding"] is not None
                and st["mejor_frontalidad"] <= 0.25):
            try:
                print(f"frontalidad -> {st["mejor_frontalidad"]}")
                cola_desconocidos.put_nowait((tid, st["mejor_embedding"], st["mejor_crop"], st["mejor_calidad"]))
                st["desconocido_guardado"] = True
            except queue.Full:
                pass

COLOR_CONFIRMADO = (0, 200, 0)
COLOR_DESCONOCIDO = (0, 0, 255)
COLOR_PROVISIONAL = (0, 200, 255)
COLOR_SIN_CARA = (200, 200, 200)

def color_para(st):
    if not st["votos"]:
        return COLOR_SIN_CARA
    if not st["confirmado"]:
        return COLOR_PROVISIONAL
    return COLOR_DESCONOCIDO if st["nombre"] == "Desconocido" else COLOR_CONFIRMADO

def hd_worker():
    while running:
        try:
            tid, box, tam = cola_hd.get(timeout=0.5)
        except queue.Empty:
            continue

        resultado = None
        embedding = None

        # box son las coordenadas de la caja que se van a tener que redimensionar.
        try:
            crop_hd = cliente_hd.recorte(box, tam, margen_x=0.1, prop_cabeza=PROP_CABEZA)
            print(f"[capturaHD] resultado: {'OK, shape=' + str(crop_hd.shape) if crop_hd is not None else 'None (HD falló -> )'}")

            if crop_hd is not None:
                r = cliente.reconocer(crop_hd, "vivo", tid = tid)
                tag = "cara" if (r and r.get("cara")) else "nocara"
                nombre_archivo = f"{CARPETA}/{int(time.time()*1000)}_id{tid}_{tag}.jpg"
                cv2.imwrite(nombre_archivo, crop_hd)

                # Si la API respondió y trae un "nombre" (incluso "Desconocido" guarda -> resultado (nombre, score).
                # ver si quitamos frontalidad.
                if r and r.get("nombre"):
                    resultado = (r["nombre"], r["score"])
                    embedding = r.get("embedding")
                    frontalidad = r.get("frontalidad")

        except Exception as e:
            print(f"[hd] error: {e}")

        with personas_lock:
            # con el trackeo -> marca que ya no hay pendiente para este TID.
            st = personas.get(tid)
            if st is not None:
                st["hd_pendiente"] = False
                if resultado:
                    st["hd_ok"] = True

        if resultado:
            agregar_voto(tid, *resultado, embedding=embedding, frontalidad=frontalidad, crop=crop_hd, peso=PESO_CAP, origen="captura")


def reco_worker():
    while running:
        try:
            tid, crop = cola_reco.get(timeout=0.5)
        except queue.Empty:
            continue
        resultado = None
        embedding = None
        try:
            r = cliente.reconocer(crop, "vivo", tid = tid)
            if r and r.get("nombre"):
                resultado = (r["nombre"], r["score"])
                embedding = r.get("embedding")
                frontalidad = r.get("frontalidad")
                calidad = r.get("calidad")
        except Exception as e:
            print(f"[reconocimiento] error: {e}")
        with personas_lock:
            st = personas.get(tid)
            if st is not None:
                st["pendiente"] = False
        if resultado:
            agregar_voto(tid, *resultado, embedding=embedding, crop=crop, frontalidad=frontalidad, calidad=calidad, peso=1, origen="vivo")

def desconocido_worker():
    while running:
        try:
            tid, embedding, crop, calidad = cola_desconocidos.get(timeout=0.5)
        except queue.Empty:
            continue
        ruta_crop = None
        if crop is not None:
            try:
                ruta_crop = f"{CARPETA_DESCONOCIDOS}/{int(time.time()*1000)}_id{tid}.jpg"
                cv2.imwrite(ruta_crop, crop)
                print("Desconocido guardado")
            except Exception as e:
                print(f"[desconocido] no se pudo guardar el crop -> error: {e}")
        try:
            r = cliente.registrar(embedding, canal=MAIN_CHANNEL, calidad=calidad, ruta_crop=ruta_crop)
            if r:
                print(f"[desconocido] ID {tid} -> {r}")
        except Exception as e:
            print(f"[desconocido] error: {e}")

def tracker_worker(fuente, detector):
    global latest_dib
    intervalo_min = 1.0/FPS_WORKER
    ultimo_fid = -1
    vueltas, acum = 0, 0.0

    while running:
        t0 = time.perf_counter()
        frame, fid = fuente.ultimo()
        if frame is None or fid == ultimo_fid:
            time.sleep(0.005)
            continue
        ultimo_fid = fid
        H, W = frame.shape[:2]

        res = detector.track(frame, persist=True, tracker=TRACKER, classes=[PERSONA], conf=CONF_TRACK, imgsz=IMGSZ, device=DEVICE_PERSONAS, verbose=False)[0]
        ahora = time.time()
        dib = []
        b = res.boxes
        if b is not None and len(b) > 0 and b.id is not None:
            cajas = b.xyxy.cpu().numpy()
            confs = b.conf.cpu().numpy()
            ids = b.id.cpu().numpy().astype(int)

            with personas_lock:
                for box, c, tid in zip(cajas, confs, ids):
                    tid = int(tid)
                    st = personas.setdefault(tid, nuevo_estado())
                    st["visto"] = ahora

                    intervalo = INTERVALO_REVERIFICAR if st["confirmado"] else INTERVALO_BUSQUEDA
                    if (c >= CONF_BUSCAR_CARA and not st["pendiente"] and ahora - st["ultimo_intento"] >= intervalo):
                        rx1, ry1, rx2, ry2 = cuadro_cabeza(box, W, H)
                        crop = frame[ry1:ry2, rx1:rx2].copy()
                        if crop.size > 0:
                            try:
                                cola_reco.put_nowait((tid, crop))
                                st["pendiente"] = True
                                st["ultimo_intento"] = ahora
                            except queue.Full:
                                pass

                    if (CAP_ACTIVO and st["votos"] and not st["hd_ok"] and not st["hd_pendiente"]
                            and st["hd_intentos"] < MAX_CAP_INTENTOS
                            and ahora - st["hd_ultimo"] >= ESPERA_ENTRE_CAPS):
                        try:
                            cola_hd.put_nowait((tid, box.copy(), (W, H)))
                            st["hd_pendiente"] = True
                            st["hd_intentos"] += 1
                            st["hd_ultimo"] = ahora
                        except queue.Full:
                            pass
                    x1, y1, x2, y2 = box.astype(int)
                    dib.append(((x1, y1), (x2, y2), f"ID {tid} {st['nombre']}", color_para(st)))

        with personas_lock:
            for tid in [t for t, s in personas.items() if ahora - s["visto"] > OLVIDAR_SEG]:
                del personas[tid]
        with dib_lock:
            latest_dib = dib

        vueltas += 1
        acum += time.perf_counter() - t0
        time.sleep(max(0.0, intervalo_min - (time.perf_counter() - t0)))


app_stream = FastAPI()


def generar_stream():
    # define los cuadros por segundo. El tiempo que debería durar cada vuelta del ciclo.
    intervalo = 1.0/STREAM_FPS
    ultimo_fid = -1

    while running:
        t0 = time.perf_counter()
        frame, fid = fuente_global.ultimo() if fuente_global else (None, -1)

        if frame is None or fid == ultimo_fid:
            time.sleep(0.01)
            continue

        ultimo_fid = fid

        # latest_dib obtenido por parte de tracker_worker.
        with dib_lock:
            dib = latest_dib

        # copia del frame crudo para no afectarlo si otro proceso lo está usando.
        vis = frame.copy()

        # recorre cada detección guardada (p1, p2 = esquinas del rectángulo; txt="ID 9 DESCONOCIDO"; color = ..."
        for p1, p2, txt, color in dib:
            cv2.rectangle(vis, p1, p2, color, 2)
            cv2.putText(vis, txt, (p1[0], max(15, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # comprime la imagen vis (la copia que ya tiene las cajas dibujadas) a JPEG.
        # STREAM_CALIDAD = 80 -> nivel de comprensión JPEG que se le pide a OpenCV.
        # El formato JPEG es "con pérdida" para ocupar menos espacio, descarta información visual que se nota menos. La escala va de 0 a 100.
        # entre más -> mejor calidad y mayor peso.
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, STREAM_CALIDAD])

        # La b antes de las comillas es para indicarle que hay bytes crudos. Se usa aquí porque se está mezclando texto (como "Content-Type..." con los bytes
        # de una foto JPEG (buf.tobytes()), y todo tiene que viajar por la red como el mismo tipo de datos: puros bytes.
        # \r\n es para indicar un "Enter".
        # de una forma muy banal, la estructura sería:
        # -- frame
        # Content-Type: image/jpeg
        # (línea vacía, aquí termina la información de la "carta").
        # [bytes de foto]
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

        time.sleep(max(0.0, intervalo - (time.perf_counter() - t0)))

@app_stream.get("/stream")
def stream():
    return StreamingResponse(generar_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

def main():
    global running, cliente, cliente_hd, fuente_global
    if not RECO_API_KEY:
        raise SystemExit("Define la variable API_KEY")
    cliente = ClienteReco(RECO_URL, RECO_API_KEY)
    cliente_hd = ClienteReco(HD_URL, HD_API_KEY, canal = MAIN_CHANNEL)
    #cliente_desc = ClienteDesconocidos(DESCONOCIDOS_URL, DESCONOCIDOS_API_KEY)
    print(f"Servidor de reconocimiento: {RECO_URL} -> "
          f"{'responde' if cliente.disponible() else 'NO responde por ahora (se seguirá intentando)'}")

    detector = YOLO(MODELO_PERSONAS)
    fuente = FuenteVideo(STREAMING, nombre="camara")
    fuente_global = fuente
    print("Conectando...")
    if not fuente.start(timeout=30) :
        print("No llegó video en 30s... :'v")

    threading.Thread(target=tracker_worker, args=(fuente, detector), daemon=True).start()
    #threading.Thread(target=hd_worker, daemon=True).start()
    threading.Thread(target=reco_worker, daemon=True).start()
    threading.Thread(target=desconocido_worker, daemon=True).start()

    try:
        print(f"Streaming en: http://{STREAM_HOST}:{STREAM_PORT}/stream")
        uvicorn.run(app_stream, host=STREAM_HOST, port=STREAM_PORT, log_level="info")
    except KeyboardInterrupt:
        print("Deteniendo...")
    finally:
        running = False
        fuente.stop();

    try:
        if VIDEO:
            cv2.namedWindow("STREAMING", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("STREAMING", 1280, 720)
            ultimo_fid = -1

            #writer = None

            while fuente.activa:
                frame, fid = fuente.ultimo()
                if frame is None or fid == ultimo_fid:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                ultimo_fid = fid

                #if writer is None:
                    #h, w = frame.shape[:2]
                    #writer = cv2.VideoWriter("debug_salida.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

                with dib_lock:
                    dib = latest_dib
                vis = frame.copy()
                for p1, p2, txt, color in dib:
                    cv2.rectangle(vis, p1, p2, color, 2)
                    cv2.putText(vis, txt, (p1[0], max(15, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                #writer.write(vis)

                cv2.imshow("STREAMING", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print("Corriendo sin video...")
            while fuente.activa:
                time.sleep(5)
                with personas_lock:
                    print(f"[estado] { {t: s ['nombre'] for t, s in personas.items() } }")

    except KeyboardInterrupt:
        print("Deteniendo")

    finally:
        running = False
        fuente.stop()
        if VIDEO:
            #if writer is not None:
                #writer.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()