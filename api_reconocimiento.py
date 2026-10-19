import os

try:
    import openvino
    os.add_dll_directory(os.path.join(os.path.dirname(openvino.__file__), "libs"))
except Exception:
    print("Hubo un error intentando usar openvino :'v")
    pass

import secrets
import time
from contextlib import asynccontextmanager
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Query, Header
from starlette.concurrency import run_in_threadpool

import onnxruntime as ort
ort.set_default_logger_severity(3)
from uniface.detection import SCRFD
from uniface.recognition import ArcFace
from uniface.constants import SCRFDWeights, ArcFaceWeights
from psycopg2.pool import ThreadedConnectionPool

from dotenv import load_dotenv
import os

load_dotenv()

HOST = "0.0.0.0"
PORT = 8000

DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("API_KEY_RECONOCIMIENTO")
RECO_DEVICE = os.getenv("DEVICE")

UMBRAL = 0.5
MIN_LADO_CARA = 35
# ?
MAX_BYTES = 4 * 1024 * 1024
face_det = None
recognizer = None
pool_conocidos = None

"""
def forzar_openvino(modelo, dispositivo):
    import onnxruntime as ort
    if dispositivo == "CPU":
        return
    for nombre, sesion in vars(modelo).items():
        if isinstance(sesion, ort.InferenceSession):
            opciones = { "device_type": dispositivo, "precision": "FP16", "cache_dir" : "ov_cache" }
            nueva = ort.InferenceSession(session._model_path, providers = [("OpenVINOExecutionProvider", opciones), "CPUExecutionProvider"])
            setattr(modelo, nombre, nueva)
            print(f"{type(modelo).__name__}: {nueva.get_providers()}")
            return
    print(f"{type(modelo).__name__}: no se encontró la sesión, se queda en CPU :P")
"""

def cargar_modelos():
    global pool_conocidos, face_det, recognizer
    print("Cargando modelos...")
    face_det = SCRFD(model_name=SCRFDWeights.SCRFD_10G_KPS, confidence_threshold=0.5, input_size=(320, 320), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    recognizer = ArcFace(model_name=ArcFaceWeights.RESNET, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    pool_conocidos = ThreadedConnectionPool(1, 5, DATABASE_URL)
    print("Listo")

def cara_mas_grande(caras):
    return max(caras, key=lambda c: (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]))

def comparar(emb, confianza_cara, tid=None):
    emb = emb / np.linalg.norm(emb)
    emb_texto = str(emb.tolist())

    con = pool_conocidos.getconn()
    try:
        # primer select -> tabla rostros_conocidos, cada fila es una persona conocida con su embedding guardado.
        # persona se renombra como nombre, y se marca origen = "conocido".
        # segundo select -> tabla rostros_desconocidos, cada fila es un desconocido guardado previamente (los que se
        # mandan por cola_desconocidos). Se le arma un nombre usando el id de la fila y se marca origen = "desconocido".
        with con.cursor() as cur:
            cur.execute(
                "SELECT nombre, similitud, origen FROM ("
                "  SELECT persona AS nombre, 1 - (embedding <=> %s::vector) AS similitud, 'conocido' AS origen"
                "  FROM rostros_conocidos"
                "  UNION ALL"
                "  SELECT 'Desconocido#' || id AS nombre, 1 - (embedding <=> %s::vector) AS similitud, 'desconocido' AS origen"
                "  FROM rostros_desconocidos"
                ") t ORDER BY similitud DESC LIMIT 1",
                (emb_texto, emb_texto))
            fila = cur.fetchone()

            if fila is not None:
                nombre, similitud, origen = fila
                umbral = UMBRAL
                paso = similitud >= UMBRAL
                cur.execute(
                    "INSERT INTO auditoria_comparaciones "
                    "(tipo, similitud, resultado, paso_umbral, confianza_cara, tid) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (origen, float(similitud), nombre, paso, confianza_cara, tid))
                con.commit()

    finally:
        pool_conocidos.putconn(con)

    if fila is None:
        return None, 0.0
    nombre, similitud, origen = fila
    umbral = UMBRAL if origen == "conocido" else UMBRAL
    return (nombre if similitud >= umbral else None), float(similitud)

UMBRAL_CONFIANZA_EMBEDDING = 0.7
# se recibe una imagen, busca la cara + grande, la reconoce contra la base de datos y devuelve un diccionario con el resultado y CUÁNTO tardó.
# se convierte en un JSON para quien hizo la petición.
# recibe la imagen en un Numpy en BGR y el parámetro de MODO.
def procesar(img, modo, tid=None):
    t0 = time.perf_counter()
    caras = face_det.detect(img)
    if not caras:
        return {"cara": False, "ms": round((time.perf_counter() - t0) * 1000, 1)}
    cara = cara_mas_grande(caras)
    # toma las 4 coordenass y las convierte en float de python.
    # Importante para la API ya que Numpy no se puede convertir a JSON directamente.
    x1, y1, x2, y2 = [float(v) for v in cara.bbox[:4]]
    lado = min(x2 - x1, y2 - y1)
    # variable que tendrá el diccionario de respuesta.
    base = {"cara": True, "lado": round(lado, 1),
            "confianza_cara": round(float(cara.confidence), 4),
            "bbox": [x1, y1, x2, y2]}
    if lado < MIN_LADO_CARA:
        # se arma el diccionario con los datos de la cara y luego update agrega Nombre = None, score=0.0 y motivo.
        base.update(nombre=None, score=0.0, motivo="cara_pequena")
    elif cara.confidence < UMBRAL_CONFIANZA_EMBEDDING:
        base.update(nombre=None, score=0.0, motivo="confianza_baja")
    else:
        # alinea la cara con los 5 landmarks, pasa por arcface y obtiene el vector de 512 números.
        emb = np.asarray(recognizer.get_normalized_embedding(img, cara.landmarks),
                         dtype=np.float32).ravel()
        # acá compara contra la base.
        n, score = comparar(emb, float(cara.confidence), tid)
        # devuelve el nombre o None y score de similitud.
        base.update(nombre=n or "Desconocido", score=round(score, 4))
        base["frontalidad"] = round(float(max(
            desvio_nariz(cara.landmarks),
            desvio_vertical(cara.landmarks)
        )), 4)
        if n is None:
            base["embedding"] = emb.tolist()
    # devuelve el tiempo y devuelve el diccionario.
    base["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return base

def desvio_nariz(landmarks):
    ojo_izq, ojo_der, nariz = landmarks[0], landmarks[1], landmarks[2]
    # [0] en ojo izq y der es la coordenada x ([0] solo está tomando lo horizontal, ignorando [1] que sería vertical).
    # entre dos ya obtenemos la coordenada del centro.
    centro_ojos_x = (ojo_izq[0] + ojo_der[0]) / 2
    # abs es el valor absoluto, se saca la distancia entre las dos coordenadas de los ojos.
    dist_ojos = abs(ojo_der[0] - ojo_izq[0])
    # regresamos el 1.0 para tratarlo cómo el peor caso, obviamente no es frontal.
    if dist_ojos < 1e-3:
        return 1.0
    # mide que tan lejos está la nariz del punto medio
    return abs(nariz[0] - centro_ojos_x)

def desvio_vertical(landmarks):
    ojo_izq, ojo_der = landmarks[0], landmarks[1]
    boca_izq, boca_der = landmarks[3], landmarks[4]
    nariz = landmarks[2]

    centro_ojos_y = (ojo_izq[1] + ojo_der[1]) / 2
    centro_boca_y = (boca_izq[1] + boca_der[1]) / 2
    dist_ojos_boca = abs(centro_boca_y - centro_ojos_y)
    if dist_ojos_boca < 1e-3:
        return 1.0

    punto_medio_y = (centro_ojos_y - centro_boca_y) / 2
    return abs(nariz[1] - punto_medio_y) / dist_ojos_boca

# API.
# Carga los modelos una sola vez al arrancar.
# async convierte una función con un yield en un gestor de contexto. Lo que está antes de yield es la entrada; lo que esté después será la salida.
@asynccontextmanager
async def lifespan(app):
    cargar_modelos()
    yield

# lifespan indica que función ejecutar al arrancar y apagar.
app = FastAPI(title="Reconocimiento", lifespan=lifespan)

# Verifica que haya clave configurada y que coincida.
def _verificar_clave(clave):
    if not API_KEY or not clave or not secrets.compare_digest(clave, API_KEY):
        raise HTTPException(status_code=401, detail="clave inválida")

# registra la función.
# le dice a fastapi que ese parámetro no viene de la URL, sino de un encabezado HTTTP.
@app.get("/salud")
def salud(x_api_key: str = Header(default="")):
    _verificar_clave(x_api_key)
    con = pool_conocidos.getconn()
    try:
        with con.cursor() as cur:
            cur.execute("SELECT count(*) FROM rostros_conocidos")
            total = cur.fetchone()[0]
    finally:
        pool_conocidos.putconn(con)
    return {"ok": True, "fotos_en_base": total, "umbral": UMBRAL}

# este es el que recibe una imagen, la manda a procesar y devuelve el resultado.
# para entender los parámetros -> request es una clase de starlette (librería sobre la que se construye FastAPI).
# esta representa toda la petición HTTP entrante. Al declarar este parámetro pides el acceso al objeto completo de la petición.
# se ocupa solo aquí porque este endpoint recibe una imagen como bytes crudos en el cuerpo (await request.body(), no como un JSON.
# hemos usado el modelo Pydantic porque este parsea JSON automáticamente, acá no se puede. Por eso se ocupa esta otra librería.
# Query es una función de FastAPI que dice explícitamente "este parámetro viene de la cadena de consulta de la URL -> la parte después del ? "
# reconocer?modo=vivo.
# Query permite agregar validaciones extra -> el primer argumento de Query(...) es el valor por defecto si el cliente no lo manda.
# pattern es una validación: que el valor recibido coincida exactamente con el string "vivo".
@app.post("/reconocer")
async def reconocer(request: Request, modo: str = Query("vivo", pattern="vivo"), tid: int = Query(None), x_api_key: str = Header(default="")):
    _verificar_clave(x_api_key)
    datos = await request.body()
    if not datos or len(datos) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="imagen vacía o demasiada grande")
    img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="no se pudo decodificar la imagen")
    return await run_in_threadpool(procesar, img, modo, tid)


if __name__ == "__main__":
    import uvicorn
    if not API_KEY:
        raise SystemExit("Define la variable de entorno API_KEY")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")