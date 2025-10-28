# -*- coding: utf-8 -*-
"""

"""

import pymcprotocol
import cv2
import numpy as np
import glob
import sys
import os
import json
import time

# ---------------------------------------------
# Configuración y rutas
# ---------------------------------------------
carpeta = sys.argv[1] if len(sys.argv) > 1 else "CAM"
config_path = os.path.join(carpeta, "config.json")
patrones_arriba_dir = os.path.join(carpeta, "patrones_arriba")
patrones_arriba2_dir = os.path.join(carpeta, "patrones_arriba2")
patrones_medio_dir = os.path.join(carpeta, "patrones_medio")
patrones_abajo_dir = os.path.join(carpeta, "patrones_abajo")

# Conexion al PLC
mc = pymcprotocol.Type3E()
mc.connect('192.168.100.120', 5007)

# Buscar imágenes
patrones = ["*.jpg", "*.png", "*.jpeg"]
lista_imgs = []
for pat in patrones:
    lista_imgs.extend(glob.glob(os.path.join(carpeta, pat)))
lista_imgs.sort()

if not lista_imgs:
    print(f"❌ No se encontraron imágenes en {carpeta}")
    sys.exit(1)

# ---------------------------------------------
# Conexión a cámara IP
# ---------------------------------------------
usuario = "mizkan"
contrasena = "mizkan"
canal = "stream1"
url_camara_ip = f"rtsp://{usuario}:{contrasena}@192.168.100.125:554/{canal}"

cap = cv2.VideoCapture(url_camara_ip)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara IP.")
    sys.exit(1)

# Capturar una imagen para inicializar
ret, frame = cap.read()
if not ret:
    print("❌ No se pudo capturar imagen desde la cámara.")
    sys.exit(1)

#lista_imgs = ["camaraframe"]  # marcador ficticio para iterar una vez

# ---------------------------------------------
# Cargar o definir ROI
# ---------------------------------------------
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    roi_arriba = tuple(config["roi_arriba"])
    roi_arriba2 = tuple(config["roi_arriba2"])
    roi_medio = tuple(config["roi_medio"])
    roi_abajo = tuple(config["roi_abajo"])
    roi_filas1 = tuple(config["roi_filas1"])
    roi_filas2 = tuple(config["roi_filas2"])
    x_minimo = config.get("x_minimo", 0)
    max_distancia_x = config.get("max_distancia_x", 250)
    umbral_validacion = config.get("umbral_validacion", 0.85)
    max_1x = config.get("max_1x", None)
    max_2x = config.get("max_2x", None)
    max_3x = config.get("max_3x", None)
else:
    print("🖼️ Mostrando primera imagen para definir los ROIs...")
    img = cv2.imread(lista_imgs[0])
    if img is None:
        print("❌ No se pudo cargar la imagen inicial.")
        sys.exit(1)

    print("🟦 Seleccione ROI superior")
    roi_arriba = cv2.selectROI("ROI superior", img, False, False)
    print("🟪 Seleccione ROI superior alternativo")
    roi_arriba2 = cv2.selectROI("ROI superior alternativo", img, False, False)
    print("🟨 Seleccione ROI medio")
    roi_medio = cv2.selectROI("ROI medio", img, False, False)
    print("🟩 Seleccione ROI inferior")
    roi_abajo = cv2.selectROI("ROI inferior", img, False, False)
    print("🖼️ Seleccione ROI general 1 para detección de filas")
    roi_filas1 = cv2.selectROI("ROI para filas", img, False, False)
    print("🖼️ Seleccione ROI general 2 para detección de filas")
    roi_filas2 = cv2.selectROI("ROI para filas", img, False, False)

    x_minimo = int(input("🔧 Ingrese el valor mínimo de X para considerar detección (en píxeles): "))
    max_distancia_x = int(input("🔧 Ingrese la distancia máxima en X entre los patrones (en píxeles): "))
    umbral_validacion = int(input("🔧 Ingrese el umbral de validación: "))
    cv2.destroyAllWindows()

    with open(config_path, "w") as f:
        json.dump({
            "roi_arriba": list(roi_arriba),
            "roi_arriba2": list(roi_arriba2),
            "roi_medio": list(roi_medio),
            "roi_abajo": list(roi_abajo),
            "roi_filas1": list(roi_filas1),
            "roi_filas2": list(roi_filas2),
            "x_minimo": x_minimo,
            "max_distancia_x": max_distancia_x,
            "umbral_validacion": umbral_validacion
        }, f, indent=4)
    print("✅ ROIs guardados en config.json")


zona_max_delta = ("ninguna", 0.0)
delta_max_mm = 0.0
zona_min_delta = ("ninguna", 0.0)
delta_min_mm = 0.0

# ---------------------------------------------
# Recortar ROI
# ---------------------------------------------  
def recortar_roi(img, roi):
    x, y, w, h = roi
    return img[y:y+h, x:x+w]

# ---------------------------------------------
# Procesamiento de imagenes
# ---------------------------------------------  
def preprocesar_imagen(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(2, 2))
    eq = clahe.apply(img)

    blur = cv2.GaussianBlur(eq, (41, 41), 2.0)
    
    return blur

# ---------------------------------------------
# Procesamiento de imagenes para deteccion de filas general
# ---------------------------------------------  
def preprocesar_imagen2(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(20, 20))
    eq = clahe.apply(img)

    blur = cv2.GaussianBlur(eq, (11, 11), 2.0)
    
    return blur

# ---------------------------------------------
# Cargar múltiples patrones desde carpetas
# ---------------------------------------------
def cargar_patrones(carpeta, fila):
    patrones = []
    path_fila = os.path.join(carpeta, f"fila{fila}")
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for path in glob.glob(os.path.join(path_fila, ext)):
            img = cv2.imread(path, 0)
            if img is not None:
                #nombre = os.path.basename(path)
                #patrones.append((img, nombre))
                img_proc = preprocesar_imagen(img)
                nombre = os.path.basename(path)
                patrones.append((img_proc, nombre))
    return patrones

# ---------------------------------------------
# ROI progresivo según filas
# --------------------------------------------- 
def calcular_roi_x_progresivo(roi_filas, fila, borde_x=20):
    x, y, w, h = roi_filas
    ancho_util = w - 2 * borde_x
    paso = (ancho_util / 8)

    ancho_roi_fila = int(paso * 2)
    inicio_x = int(x + borde_x + (paso * (fila - 1.5)))

    return (inicio_x, y, ancho_roi_fila, h)

# ---------------------------------------------
# Detectar fials progresivamente
# --------------------------------------------- 
def detectar_fila_por_sectores(gray, roi_filas, referencias_por_filas, umbral=0.70, borde_x=10):
    """
    Evalúa progresivamente si hay presencia de filas a partir de cada sector (1 a 8).
    Devuelve la fila más baja (más cercana al top) en la que hay detección.
    """
    umbral_vacio = 0.80
    x, y, w, h = roi_filas
    ancho_util = w - 2 * borde_x
    paso = ancho_util / 8
    ancho_sector = int(paso * 2.0)
    ancho_patron = int(paso * 1.5)

    mejor_fila_detectada = 9  # Por defecto: fila 9 = sin filas
    mejor_score = -1
    mejor_nombre = "N/A"
    mejor_roi = (0, 0, 0, 0)

    # Referencias de F9 = sin fila
    referencias_f9 = referencias_por_filas.get(9, [])

    for fila in range(8, 0, -1):  # de 8 a 1
        #print("fila:" + str(fila))
        inicio_x_sector = int(borde_x + (paso * (fila - 1.5 - 0.6)))
        inicio_x_patron = int(borde_x + (paso * (fila - 1.25 - 0.6)))
        roi_actual = gray[0:h, max(0,inicio_x_sector):inicio_x_sector+ancho_sector]

        if fila not in referencias_por_filas:
            #print("sin referencias")
            continue

        for patron, nombre in referencias_por_filas[fila]:
            patron_actual = patron[0+5:h-10, max(0,inicio_x_patron):inicio_x_patron+ancho_patron]

            if roi_actual.shape[0] < patron_actual.shape[0] or roi_actual.shape[1] < patron_actual.shape[1]:
                print("Error tamaños")
                continue

            res = cv2.matchTemplate(roi_actual, patron_actual, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)
            if fila <= 2:
                umbral_final =0.80
                umbral_vacio = 0.85
            else:
                umbral_final = umbral
            if score > umbral_final:  # Umbral de validación
                # Validar contra patrones F9 (sin fila)
                es_valido = True
                for patron_f9, _ in referencias_f9:
                    patron_f9_crop = patron_f9[0+5:h-10, max(0, inicio_x_patron):inicio_x_patron + ancho_patron]
                    if (roi_actual.shape[0] >= patron_f9_crop.shape[0] and roi_actual.shape[1] >= patron_f9_crop.shape[1]):
                        res_f9 = cv2.matchTemplate(roi_actual, patron_f9_crop, cv2.TM_CCOEFF_NORMED)
                        _, score_f9, _, _ = cv2.minMaxLoc(res_f9)
                        if score_f9 > umbral_vacio:
                            es_valido = False
                            break
                if es_valido and fila < mejor_fila_detectada:
                    mejor_fila_detectada = fila
                    mejor_score = score
                    mejor_nombre = nombre
                    mejor_roi = (max(0,inicio_x_patron)+x, y,ancho_patron,h)
                    #print(mejor_fila_detectada)
                    #print(mejor_score)
                    #print(mejor_nombre)
                break  # ya detectamos una fila en esta zona

    return mejor_fila_detectada, mejor_score, mejor_nombre, mejor_roi

# ---------------------------------------------
# Cargar patrones de filas
# ---------------------------------------------  
def cargar_referencias_por_filas(base_folder, roi_filas):
    referencias = {}
    roi_filas_final = {}
    for i in range(1, 10):
        carpeta = os.path.join(base_folder, f"F{i}")
        refs = []
        #roi_filas_x = calcular_roi_x_progresivo(roi_filas,i)
        roi_filas_x = roi_filas
        for img_path in glob.glob(os.path.join(carpeta, "*.jpg")) + glob.glob(os.path.join(carpeta, "*.png")):
            img = cv2.imread(img_path, 0)
            if img is not None:
                ref_img = recortar_roi(img, roi_filas_x)
                #refs.append(ref_img)
                img_proc = preprocesar_imagen2(ref_img)
                nombre = os.path.basename(img_path)
                refs.append((img_proc, nombre))
        if refs:
            referencias[i] = refs
            roi_filas_final[i] = roi_filas_x
    return referencias, roi_filas_final

# ---------------------------------------------
# Detectar mejor patrón considerando posición
# ---------------------------------------------
def detectar_mejor_patron(roi, patrones, x_global_offset, peso_izquierda=0.2, umbral_validacion=0.8, x_maximo=None):
    resultados = []
    for patron, nombre in patrones:
        if roi.shape[0] < patron.shape[0] or roi.shape[1] < patron.shape[1]:
            continue
        res = cv2.matchTemplate(roi, patron, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        x_global = max_loc[0] + x_global_offset

        centerX = int((max_loc[0] + patron.shape[1] // 2) + x_global_offset)

        # Filtros por posición global
        if centerX < x_minimo:
            continue
        if (x_maximo is not None) and (centerX > x_maximo):
            continue
        if (x_maximo is not None) and (centerX < (x_maximo - 150)):
            continue
        # Filtro por umbral de coincidencia
        if max_val < umbral_validacion:
            continue

        x_norm = max_loc[0] / roi.shape[1]
        score = max_val - peso_izquierda * x_norm
        center = (max_loc[0] + patron.shape[1] // 2, max_loc[1] + patron.shape[0] // 2)
        resultados.append((center, max_val, score, nombre))

    resultados.sort(key=lambda x: x[2], reverse=True)
    return resultados

# ---------------------------------------------
# Detectar cantidad de filas
# ---------------------------------------------
def detectar_cantidad_filas(imagen_actual, referencias, alpha = 0.0):
    mejor_score = -1
    mejor_cantidad = 0
    mejor_archivo = "N/A"
    area_maxima = 1

    h0, w0 = imagen_actual.shape[:2]
    area_maxima = w0 * h0

    for filas, patrones in referencias.items():
        for patron, nombre in patrones:
            if imagen_actual.shape[0] < patron.shape[0] or imagen_actual.shape[1] < patron.shape[1]:
                continue
            res = cv2.matchTemplate(imagen_actual, patron, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)
            h, w = patron.shape[:2]
            area = w * h
            score_ponderado = score * (1 + alpha * (area / area_maxima)) * (1 + ((9 - filas) * 0.0))
            if score_ponderado > mejor_score:
                mejor_score = score_ponderado
                mejor_cantidad = filas
                mejor_archivo = nombre
    return mejor_cantidad, mejor_score, mejor_archivo



# ---------------------------------------------
# Detectar rectas
# ---------------------------------------------
def detectar_y_dibujar_rectas(img, gray, cantidad_filas, roi_filas_x, umbral_validacion, fallback=True):
    """
    Intenta detectar los puntos y dibujar rectas para una fila. Si falla y fallback está habilitado, prueba con la siguiente fila.
    Devuelve True si tuvo éxito, False si no.
    """
    global nombre_arriba, nombre_medio, nombre_abajo, val1, val2, val3, zona_max_delta, delta_max_mm, zona_min_delta, delta_min_mm

    zona_max_delta = ("ninguna", 0.0)
    delta_max_mm = 0.0
    zona_min_delta = ("ninguna", 0.0)
    delta_min_mm = 0.0

    try:
        x1, y1, w1, h1 = roi_arriba
        x12, y12, w12, h12 = roi_arriba2
        x2, y2, w2, h2 = roi_abajo
        x3, y3, w3, h3 = roi_medio

        x0, y0, w0, h0 = roi_filas_x.get(cantidad_filas, (0, 0, 0, 0))

        distx_max1 = max_1x[cantidad_filas - 1]
        distx_max2 = max_2x[cantidad_filas - 1]
        distx_max3 = max_3x[cantidad_filas - 1]

        patrones_arriba = cargar_patrones(patrones_arriba_dir, cantidad_filas)
        patrones_arriba2 = cargar_patrones(patrones_arriba2_dir, cantidad_filas)
        patrones_medio = cargar_patrones(patrones_medio_dir, cantidad_filas)
        patrones_abajo = cargar_patrones(patrones_abajo_dir, cantidad_filas)

        if not patrones_arriba or not patrones_arriba2 or not patrones_abajo or not patrones_medio:
            return -1

        roi1 = gray[y1:y1+h1, x1:x1+w1]
        roi12 = gray[y12:y12+h12, x12:x12+w12]
        roi2 = gray[y2:y2+h2, x2:x2+w2]
        roi3 = gray[y3:y3+h3, x3:x3+w3]

        resultados1 = detectar_mejor_patron(roi1, patrones_arriba, x1, 0.2, umbral_validacion, distx_max1)
        if not resultados1:
            resultados1 = detectar_mejor_patron(roi12, patrones_arriba2, x12, 0.2, umbral_validacion, distx_max1)

        resultados2 = detectar_mejor_patron(roi2, patrones_abajo, x2, 0.2, umbral_validacion, distx_max3)
        resultados3 = detectar_mejor_patron(roi3, patrones_medio, x3, 0.2, umbral_validacion/2, distx_max2)

        combos = []
        for res1 in resultados1:
            for res2 in resultados2:
                distancia_x = abs((x1 + res1[0][0]) - (x2 + res2[0][0]))
                if distancia_x <= max_distancia_x:
                    scorecombo = res1[2] + res2[2]
                    combos.append((res1[0], res1[1], res2[0], res2[1], scorecombo, (res1[3], res2[3])))

        if not combos:
            raise Exception("No se encontraron combos válidos")

        mejor_combo = max(combos, key=lambda c: c[4])
        loc1, val1, loc2, val2, scorecombo, nombrescombo = mejor_combo

        pt1 = (x1 + loc1[0], y1 + loc1[1])
        pt2 = (x2 + loc2[0], y2 + loc2[1])
        medio_ideal = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

        mejor_loc3 = None
        mejor_score_loc3 = -np.inf

        for res3 in resultados3:
            pt3 = (x3 + res3[0][0], y3 + res3[0][1])
            distancia = np.linalg.norm(np.array(pt3) - np.array(medio_ideal))
            if distancia > 250:
                continue
            dist_norm = distancia / 100.0
            score = res3[1] - 1.5 * dist_norm
            if score > mejor_score_loc3:
                mejor_score_loc3 = score
                mejor_loc3 = res3
                val3 = res3[1]

        #if mejor_loc3 is None:
        #    raise Exception("No se encontró punto medio adecuado")
        
        if mejor_loc3:
            pt3 = (x3 + mejor_loc3[0][0], y3 + mejor_loc3[0][1])
            #cv2.circle(img, pt3, 5, (0, 255, 0), -1)
            cv2.line(img, pt1, pt3, (0, 0, 255), 3)
            cv2.line(img, pt3, pt2, (0, 0, 255), 3)
            nombre_medio = mejor_loc3[3]
            #print(pt3)
        else:
            cv2.line(img, pt1, pt2, (255, 0, 255), 3)
            nombre_medio = "N/A"

        #pt3 = (x3 + mejor_loc3[0][0], y3 + mejor_loc3[0][1])
        #cv2.line(img, pt1, pt3, (0, 0, 255), 3)
        #cv2.line(img, pt3, pt2, (0, 0, 255), 3)
        nombre_arriba, nombre_abajo = nombrescombo
        ################################
        try:
            pos_ideal = config["posiciones_filas"].get(str(cantidad_filas), None)
            mm_por_pixel = config.get("mm_por_pixel", {})

            if pos_ideal:
                # Detectado
                pos_real_arriba = pt1[0]
                pos_real_abajo = pt2[0]
                pos_real_medio = pt3[0] if nombre_medio != "N/A" else None

                # Esperado
                pos_ideal_arriba = pos_ideal.get("arriba", pos_real_arriba)
                pos_ideal_medio = pos_ideal.get("medio", pos_real_medio) if pos_real_medio is not None else None
                pos_ideal_abajo = pos_ideal.get("abajo", pos_real_abajo)

                # Deltas en mm
                delta_arriba = (pos_ideal_arriba - pos_real_arriba) * mm_por_pixel.get("arriba", 0.0)
                delta_abajo = (pos_ideal_abajo - pos_real_abajo) * mm_por_pixel.get("abajo", 0.0)
                delta_medio = (
                    (pos_ideal_medio - pos_real_medio) * mm_por_pixel.get("medio", 0.0)
                    if pos_real_medio is not None and pos_ideal_medio is not None else 0.0
                )

                # Determinar el máximo
                zona_max_delta = max(
                    [("arriba", delta_arriba), ("medio", delta_medio), ("abajo", delta_abajo)],
                    key=lambda x: x[1]
                )
                delta_max_mm = zona_max_delta[1]
                zona_min_delta = min(
                    [("arriba", delta_arriba), ("medio", delta_medio), ("abajo", delta_abajo)],
                    key=lambda x: x[1]
                )
                delta_min_mm = zona_min_delta[1]



        except Exception as e:
            print(f"⚠️ Error al calcular delta: {e}")
            zona_max_delta = ("ninguna", 0.0)
            delta_max_mm = 0.0
            zona_min_delta = ("ninguna", 0.0)
            delta_min_mm = 0.0
        
        ################################
        # Mostrar
        print(f"📏 Máxima desviación: {delta_max_mm:.2f} mm en zona: {zona_max_delta[0]} (Fila {cantidad_filas})")
        cv2.putText(img, f"Delta max: {delta_max_mm:.2f} mm - {zona_max_delta[0]}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
        print(f"📏 Mínima desviación: {delta_min_mm:.2f} mm en zona: {zona_min_delta[0]} (Fila {cantidad_filas})")
        cv2.putText(img, f"Delta min: {delta_min_mm:.2f} mm - {zona_min_delta[0]}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

        return cantidad_filas

    except Exception as e:
        if fallback and cantidad_filas < 8:
            print(f"⚠️ Falló fila {cantidad_filas}, intentando fila {cantidad_filas + 1}")
            return detectar_y_dibujar_rectas(img, gray, cantidad_filas + 1, roi_filas_x, umbral_validacion, False)

        return -1

# Funciones auxiliares
def int32_to_words(n: int) -> list[int]:
    """Convierte un entero con signo de 32 bits a dos WORDs (low, high)."""
    n &= 0xFFFFFFFF  # complemento a dos para 32 bits
    return [n & 0xFFFF, (n >> 16) & 0xFFFF]

# ---------------------------------------------
# Procesar imágenes
# ---------------------------------------------
cv2.namedWindow("Resultado", cv2.WINDOW_NORMAL)
#cv2.namedWindow("Procesado", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Resultado", 1600, 900)
#cv2.resizeWindow("Procesado", 1600, 900)

# Cargar referencias de carpetas f1/ a f8/
referencias_filas1, roi_filas_x1 = cargar_referencias_por_filas(carpeta, roi_filas1)
referencias_filas2, roi_filas_x2 = cargar_referencias_por_filas(carpeta, roi_filas2)

x1, y1, w1, h1 = roi_arriba
x12, y12, w12, h12 = roi_arriba2
x2, y2, w2, h2 = roi_abajo
x3, y3, w3, h3 = roi_medio

x01, y01, w01, h01 = roi_filas1
x02, y02, w02, h02 = roi_filas2
x0, y0, w0, h0 = 0, 0, 0, 0

i = 0
intervalo_ms = 500
ultimo_tiempo = time.time()
ok_procesar = False
automatic = False

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    _ = cap.read()

    tiempo_actual = time.time()
    if (tiempo_actual - ultimo_tiempo) * 1000 >= intervalo_ms:
        ultimo_tiempo = tiempo_actual
        ok_procesar = True
    try:
        d28 = mc.batchread_wordunits(headdevice='D28', readsize=1)[0]
        if (d28 == 99) or (ok_procesar and automatic):
            print("🔍 Solicitud de inspección (D28 = 99)")
        
            ret, img = cap.read()
            if not ret:
                print("❌ No se pudo capturar imagen.")
                continue

            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = preprocesar_imagen(img)
            gray2 = preprocesar_imagen2(img)
            img_filas1 = recortar_roi(gray2, roi_filas1)
            img_filas2 = recortar_roi(gray2, roi_filas2)
            # Detectar cuántas filas hay
            #cantidad_filas1, score_filas1, patron_filas1 = detectar_cantidad_filas(img_filas1, referencias_filas1)
            #cantidad_filas2, score_filas2, patron_filas2 = detectar_cantidad_filas(img_filas2, referencias_filas2)
            cantidad_filas1, score_filas1, patron_filas1, patron_roi1 = detectar_fila_por_sectores(img_filas1, roi_filas1, referencias_filas1)
            cantidad_filas2, score_filas2, patron_filas2, patron_roi2 = detectar_fila_por_sectores(img_filas2, roi_filas2, referencias_filas2)

            xr1, yr1, wr1, hr1 = patron_roi1
            xr2, yr2, wr2, hr2 = patron_roi2

            if cantidad_filas1 < cantidad_filas2:
                cantidad_filas = cantidad_filas1
                score_filas = score_filas1
                patron_filas = patron_filas1
                roi_filas_x = roi_filas_x1
            else:
                cantidad_filas = cantidad_filas2
                score_filas = score_filas2
                patron_filas = patron_filas2
                roi_filas_x = roi_filas_x2

            nombre_arriba = "N/A"
            nombre_medio = "N/A"
            nombre_abajo = "N/A"
            val1 = val2 = val3 = -1

            #####################################################################################################
            nombre_arriba = "N/A"
            nombre_medio = "N/A"
            nombre_abajo = "N/A"
            val1 = val2 = val3 = -1

            if 2 < cantidad_filas < 9:
                fila_corregida = detectar_y_dibujar_rectas(img, gray, cantidad_filas, roi_filas_x, umbral_validacion)
                if fila_corregida > 0:
                    cantidad_filas = fila_corregida  # Actualizamos al valor corregido
                else:
                    print(f"❌ No se pudo trazar rectas ni con fallback desde fila {cantidad_filas}")

            #####################################################################################################
            # Interpretar resultado
            texto_resultado1 = "Sin filas" if cantidad_filas1 == 9 else f"{cantidad_filas1}"
            texto_resultado2 = "Sin filas" if cantidad_filas2 == 9 else f"{cantidad_filas2}"
            texto_resultado = "Sin filas" if cantidad_filas == 9 else f"{cantidad_filas}"

            if (cantidad_filas < 9) and (cantidad_filas > 0):
                valor_envio = int(round(-delta_min_mm * 100))  # en centésimas de mm
                words = int32_to_words(valor_envio)     # -> [low_word, high_word]
                mc.batchwrite_wordunits(headdevice='D29', values=words)
                mc.batchwrite_wordunits(headdevice='D14', values=[cantidad_filas])
                mc.batchwrite_wordunits(headdevice='D28', values=[88])
            else:
                words = int32_to_words(0)     # -> [low_word, high_word]
                mc.batchwrite_wordunits(headdevice='D29', values=words)
                mc.batchwrite_wordunits(headdevice='D14', values=[0])
                mc.batchwrite_wordunits(headdevice='D28', values=[77])


            # Mostrar en la imagen
            cv2.putText(img, f"Fila: {texto_resultado1} (score={score_filas1:.2f}) - {patron_filas1}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(img, f"Fila: {texto_resultado2} (score={score_filas2:.2f}) - {patron_filas2}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            # Obtener dimensiones de la imagen y del texto
            h_img, w_img = img.shape[:2]
            (text_w, text_h), _ = cv2.getTextSize(texto_resultado, cv2.FONT_HERSHEY_SIMPLEX, 10, 6)
            # Coordenadas para centrar horizontalmente y ponerlo a, por ejemplo, 50 px desde arriba
            x_text = (w_img - text_w) // 2
            cv2.putText(img, f"{texto_resultado}",
                        (x_text, 720), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 6)
            
            cv2.putText(img, f"ProbArriba: {val1:.2f} - {nombre_arriba}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
            cv2.putText(img, f"ProbMedio: {val3:.2f} - {nombre_medio}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
            cv2.putText(img, f"ProbAbajo: {val2:.2f} - {nombre_abajo}", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)

            cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
            cv2.rectangle(img, (x12, y12), (x12 + w12, y12 + h12), (255, 100, 0), 2)

            cv2.rectangle(img, (x3, y3), (x3 + w3, y3 + h3), (0, 255, 255), 2)
            cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

            cv2.rectangle(img, (xr1, yr1), (xr1+wr1, yr1+hr1), (255, 0, 255), 3)
            cv2.rectangle(img, (xr2, yr2), (xr2+wr2, yr2+hr2), (255, 0, 255), 3)
            cv2.rectangle(img, (x0, y0), (x0+w0, y0+h0), (0, 0, 0), 3)

            cv2.imshow("Resultado", img)
            #cv2.imshow("Procesado", gray2)
    
    except Exception as e:
        words = int32_to_words(0)     # -> [low_word, high_word]
        mc.batchwrite_wordunits(headdevice='D29', values=words)
        mc.batchwrite_wordunits(headdevice='D14', values=[0])
        mc.batchwrite_wordunits(headdevice='D28', values=[77])
        print(f"❌ Error: {e}")

cv2.destroyAllWindows()