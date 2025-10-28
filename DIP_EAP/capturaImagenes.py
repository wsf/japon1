import cv2
import os
import time
from datetime import datetime

# Configuración de la cámara IP
usuario = "mizkan"
contrasena = "mizkan"
canal = "stream1"
url_camara_ip = f"rtsp://{usuario}:{contrasena}@192.168.100.125:554/{canal}"

#url_camara_ip = f"rtsp://{usuario}:{contraseña}@192.168.3.184:554/{canal}"
#url_qr_camara = f"rtsp://{usuario}:{contraseña}@192.168.3.182:554/{canal}"

# Carpeta donde se guardarán las capturas
carpeta_salida = "capturas"
os.makedirs(carpeta_salida, exist_ok=True)

# Inicializar la cámara
cap = cv2.VideoCapture(url_camara_ip)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("📷 Cámara conectada. Presiona [ESPACIO] para capturar una imagen.")
print("Presiona [q] para salir.")

# --- Limitador de FPS ---
FPS_OBJETIVO = 1  # fotogramas por segundo
intervalo = 1.0 / FPS_OBJETIVO
ultimo_mostrado = 0.0
# ------------------------

contador = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Error capturando imagen.")
        continue

    # Mostrar video en vivo
    ahora = time.time()
    if ahora - ultimo_mostrado >= intervalo:
        # Mostrar video en vivo (1 fps)
        preview = cv2.resize(frame, (1600, 900))
        cv2.imshow("Captura desde cámara IP", preview)
        ultimo_mostrado = ahora

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # ESPACIO para capturar imagen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"captura_{timestamp}.jpg"
        path_completo = os.path.join(carpeta_salida, nombre_archivo)
        cv2.imwrite(path_completo, frame)
        print(f"✅ Imagen guardada: {path_completo}")
        contador += 1

    elif key == ord('q'):  # Q para salir
        print("👋 Saliendo del programa...")
        break

cap.release()
cv2.destroyAllWindows()
