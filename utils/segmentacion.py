import numpy as np

def cuadro_cabeza(box, ancho, alto, prop_cabeza=0.45, margen_x=0.15, margen_arriba=0.05):
    """Devuelve x1, y1, x2, y2 (zona de la cabeza), dentro de la imagen."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    w, h = x2 - x1, y2 - y1
    rx1 = max(0, int(x1 - w * margen_x))
    rx2 = min(ancho, int(x2 + w * margen_x))
    ry1 = max(0, int(y1 - h * margen_arriba))
    ry2 = min(alto, int(y1 + h * prop_cabeza))
    return rx1, ry1, rx2, ry2


def recortar(img, bbox, margen=0.3):
    """Recorta bbox con un margen proporcional. Devuelve una copia."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * margen)); y1 = max(0, int(y1 - bh * margen))
    x2 = min(w, int(x2 + bw * margen)); y2 = min(h, int(y2 + bh * margen))
    return img[y1:y2, x1:x2].copy()


def escalar_caja(box, de_tam, a_tam):
    """Convierte una caja entre resoluciones. de_tam y a_tam son (ancho, alto)."""
    sx, sy = a_tam[0] / de_tam[0], a_tam[1] / de_tam[1]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)


def cara_mas_grande(caras):
    """Devuelve la cara de mayor área."""
    return max(caras, key=lambda c: (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]))