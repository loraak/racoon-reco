import os

from psycopg2.extras import Json
from pydantic import BaseModel, Field

try:
    import openvino
    os.add_dll_directory(os.path.join(os.path.dirname(openvino.__file__), "libs"))
except Exception:
    print("Hubo un error intentando usar openvino :'v")
    pass

import secrets
import time
from contextlib import asynccontextmanager, contextmanager
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

MIN_LADO_CARA = 35

# ?
MAX_BYTES = 4 * 1024 * 1024
face_det = None
recognizer = None
LIMITE_EMBEDDING = 5
CREADO_POR = "apiReconocimiento"

# postgres maqneja candados "de aviso". Mientras una transacción tiene este candado, cualquier otra que lo pida espera a que termine.
# así, dos registros no ocurren al mismo tiempo y no se crean dos personas para la misma cara.
LOCK_REGISTRO = 7201
MODELO = "arcface_resnet" # para insertar dentro de la base de datos.
DIMENSION = 512

# comparación
UMBRAL_CONOCIDO = 0.5
UMBRAL_DESCONOCIDO = 0.5
MARGEN_MINIMO = 0.05 # ventaja del 1er lugar al 2do.
VECINOS = 20 # x cantidad de embeddings cercanos revisa antes de elegir a la persona.
pool_argos = None
DATABASE_ARGOS = os.getenv("DATABASE_ARGOS")

# convierte la función en algo que se usa con with "with .. as".
@contextmanager
def conexion():
    # pide una conexión al pool.
    con = pool_argos.getconn()
    try:
        # entrega la conexión. La función se pausa mientras se ejecuta el código de adentro.
        yield con
        # si el bloque with terminó bien, guarda los cambios.
        con.commit()
    except Exception:
        con.rollback()
        # vuelve a lanzar el error para que no se oculte. FasAPI lo convierte en una respuesta 500.
        raise
    finally:
        pool_argos.putconn(con)

# convierte el embedding al formato que entiende pgvector.
def a_texto(emb):
    # el embedding puede llegar como lista o como arreglo, esto lo convierte siempre a arreglo de numpy
    # ravel lo aplana a una sola fila.
    emb = np.asarray(emb, dtype=np.float32).ravel()
    emb = emb / np.linalg.norm(emb)
    return str(emb.tolist())

# PASO 1.
# with, as... define tablas temporales que solo existen mientras corre la consulta.
# cada una usa el resultado de la anterior: vecinos -> ordenados -> por_persona -> final.
# k es el límite de embeddings, toma el valor de "VECINOS" (o sea 20).
# embedding <=> es la distancia coseno. 0 -> idénticas.
# 1 - x es la similitud: 1 significa que son idénticas.
# "k" se queda con las k fotos más parecidas de toda la tabla, sin importar de quién sean.
# emb se trata de los 512 vectores.

# PASO 2.
# De la tabla que acabamos de crear (vecinos).
# patition by pid separa las filas en grupos, uno por persona.
# order by ordena cada grupo de la foto más parecida a la menos parecida.
# row... las numera 1, 2, 3, dentro de cada grupo.
# pid -> sim -> rn
# Tenna -> 0.71 -> 1
# Tenna -> 0.66 -> 2
# Otro -> 0.69 -> 1
# Otro -> 0.45 -> 2

# PASO 3.
# por persona.
# junta todas las filas de cada persona en una sola. Para c/u calcula tres cosas:
# mejor -> la foto más parecida.
# prom_top3 -> el promedio de las 3 fotos más parecidas. FILTER (WHERE rn <= 3) toma solo las filas numeradas. Si hay menos de 3, promedia las que haya.
# votos -> cuántas de las k fotos cercanas son suyas.
# pid -> mejor -> prom_top3 -> votos
# Tenna -> 0.71 -> 0.67 (ejemplo) -> 2
# Otro -> 0.69 -> 0.57 (ejemplo) -> 2
# para evitar coincidencias y tenerlo todo promediado mejor.

# PASO 4.
# JOIN une esos puntajes con la tabla de personas para obtener nombre y tipo.
# CASE arma el nombre
# order by... regresa al primer y segunda lugar. El segundo sirve para medir el margen en Python.
SQL_BUSCAR = """
WITH vecinos as (
    SELECT persona_identificada_id AS pid, 1 - (embedding <=> %(emb)s::vector) AS sim 
    FROM vector.rostro_persona 
    ORDER BY embedding <=> %(emb)s::vector 
    LIMIT %(k)s
), 
ordenados AS (
    SELECT pid, sim, row_number() OVER (PARTITION BY pid ORDER BY sim DESC) AS rn 
    FROM vecinos
), 
por_persona AS (
    SELECT pid, max(sim) AS mejor, avg(sim) FILTER (WHERE rn <=3) AS prom_top3, count(*) AS votos 
    FROM ordenados 
    GROUP BY pid  
)
SELECT p.id, 
    CASE WHEN p.tipo = 'conocido' 
         THEN concat_ws(' ', p.nombre) 
         ELSE 'Desconocido #' || p.id 
    END AS nombre, 
    p.tipo, pp.mejor, pp.prom_top3, pp.votos 
FROM por_persona pp 
JOIN argos.persona_identificada p ON p.id = pp.pid 
ORDER BY pp.prom_top3 DESC 
LIMIT 2
"""

def buscar(cur, emb_texto):
    cur.execute(SQL_BUSCAR, {"emb": emb_texto, "k": VECINOS})
    return cur.fetchall()

def cargar_modelos():
    global face_det, recognizer, pool_argos
    print("Cargando modelos...")
    face_det = SCRFD(model_name=SCRFDWeights.SCRFD_10G_KPS, confidence_threshold=0.5, input_size=(320, 320), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    recognizer = ArcFace(model_name=ArcFaceWeights.RESNET, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    pool_argos = ThreadedConnectionPool(1, 5, DATABASE_ARGOS)
    print("Listo")

def cara_mas_grande(caras):
    return max(caras, key=lambda c: (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]))

# para confirmar que se reconoce a alguien, se deben cumplir dos condiciones:
# puntaje >= umbral: se parece lo suficiente.
# margen >= MARGEN_MINIMO: le gana con ventaja clara. Es preferible que no sepa a que de malos resultados ¿? preguntar.
def comparar(emb):
    # pide una conexión, convierte el embedding a texto y busca.
    with conexion() as con, con.cursor() as cur:
        filas = buscar(cur, a_texto(emb))

    # si no hay resultados, no hay a quién reconocer, regresa nadie con similitud 0.
    if not filas:
        return None, 0.0

    # toma la primera fila (primer lugar) y reparte los 6 valores en variables. _ significa que se van a ignorar.
    _, nombre, tipo, _, prom_top3, _ = filas[0]
    puntaje = float(prom_top3)
    segundo = float(filas[1][4] if len(filas) > 1 else 0.0)

    umbral = UMBRAL_CONOCIDO if tipo == "conocido" else UMBRAL_DESCONOCIDO

    # paso es verdad solo si se pasan dos condiciones:
    # se parece lo suficiente (puntaje >= umbral)
    # saca vnetaja clara al segundo lugar.
    paso = puntaje >= umbral and (puntaje - segundo) >= MARGEN_MINIMO

    return (nombre if paso else None), puntaje

def agregar_embedding (cur, persona_id, emb_texto, calidad, ruta_crop, metadata):
    # for update bloquea la fila de la persona hasta que la transacción termine.
    cur.execute("SELECT tipo FROM argos.persona_identificada WHERE id = %s FOR UPDATE", (persona_id, ))

    # devuelve el id.
    cur.execute(
        "INSERT INTO vector.rostro_persona "
        "(persona_identificada_id, embedding, imagen_url, modelo_embedding, dimension, "
        "status, calidad, metadata, created_at, updated_at, created_by, updated_by) "
        "VALUES (%s, %s::vector, %s, %s, %s, 'activo', %s, %s, now(), now(), %s, %s) " 
        "RETURNING id",
        (persona_id, emb_texto, ruta_crop or "", MODELO, DIMENSION, calidad, Json(metadata), CREADO_POR, CREADO_POR))
    nuevo_id = cur.fetchone()[0]

    # ¿qué embeddings sobran?
    # where -> solo los embeddings que no son "fijos" -> probablemente se quite, no sée.....
    # order by -> ordenador de mejor a peor calidad.
    # null -> lo que notienen calidad van al final.
    # created at... -> si dos tienen la misma calidad, el más reciente va primero.
    # offset -> se salta los primeros N y devuelve el resto. Estos son los sobrantes.
    cur.execute(
        "DELETE FROM vector.rostro_persona e "
        "USING ( "
        "       SELECT id FROM vector.rostro_persona "
        "       WHERE persona_identificada_id = %s AND NOT fijo "
        "       ORDER BY calidad DESC NULLS LAST, created_at DESC "
        "       OFFSET %s "
        ") sobrantes "
        "WHERE e.id = sobrantes.id "
        "RETURNING e.id",
        (persona_id, LIMITE_EMBEDDING)
    )
    borrados = {f[0] for f in cur.fetchall()}

    # regresa dos valores: nuevo_id: el id del embedding insertado. nuevo_id not in borrados: t/f -> si el nuevo sobrevivió/ false si era el peor y se borró.
    return nuevo_id, nuevo_id not in borrados

def registrar_desconocido(emb, canal, calidad=None, ruta_crop=None):
    emb_texto = a_texto(emb)

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
        n, score = comparar(emb)

        front = frontalidad(cara.landmarks)

        # devuelve el nombre o None y score de similitud.
        base.update(nombre=n or "Desconocido",
                    score=round(score, 4),
                    frontalidad=round(float(front), 4),
                    calidad=round(calcular_calidad(cara.confidence, front), 4))

        if n is None or n.startswith("Desconocido#"):
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
    return abs(nariz[0] - centro_ojos_x) / dist_ojos

def desvio_vertical(landmarks):
    ojo_izq, ojo_der = landmarks[0], landmarks[1]
    boca_izq, boca_der = landmarks[3], landmarks[4]
    nariz = landmarks[2]

    centro_ojos_y = (ojo_izq[1] + ojo_der[1]) / 2
    centro_boca_y = (boca_izq[1] + boca_der[1]) / 2
    dist_ojos_boca = abs(centro_boca_y - centro_ojos_y)
    if dist_ojos_boca < 1e-3:
        return 1.0

    posicion = (nariz[1] - centro_ojos_y) / dist_ojos_boca
    return abs(posicion - 0.5)

def frontalidad(landmarks):
    return max(desvio_nariz(landmarks), desvio_vertical(landmarks))

def calcular_calidad(confianza, frontalidad):
    return float(confianza) * (1.0 - min(float(frontalidad), 1.0))

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

class PeticionDesconocido(BaseModel):
    embedding: list[float] = Field(min_length=512, max_length=512)
    canal: int = Field(ge=1)
    calidad: float | None = None
    ruta_crop: str | None = None

# registra la función.
# le dice a fastapi que ese parámetro no viene de la URL, sino de un encabezado HTTTP.
@app.get("/salud")
def salud(x_api_key: str = Header(default="")):
    _verificar_clave(x_api_key)
    con = pool_argos.getconn()
    try:
        with con.cursor() as cur:
            cur.execute("SELECT count(*) FROM persona_identificada")
            total = cur.fetchone()[0]
    finally:
        pool_argos.putconn(con)
    return {"ok": True, "fotos_en_base": total, "umbral": UMBRAL_CONOCIDO}

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

@app.post("/registrar-desconocido")
def registrar(p: PeticionDesconocido, x_api_key: str = Header(default="")):
    _verificar_clave(x_api_key)
    return registrar_desconocido(p.embedding, p.canal, p.calidad, p.ruta_crop)

if __name__ == "__main__":
    import uvicorn
    if not API_KEY:
        raise SystemExit("Define la variable de entorno API_KEY")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")