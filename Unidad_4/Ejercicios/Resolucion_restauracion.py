import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carga en color; la gris se deriva de ella
img_color = cv2.imread("Unidad_4/Ejercicios/qr_and_phone.png")
img_gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Umbral 200: solo quedan píxeles muy brillantes, es decir, el fondo blanco del QR
_, binaria = cv2.threshold(img_gris, 200, 255, cv2.THRESH_BINARY)

# Contornos externos de las regiones blancas, ordenados de mayor a menor área
contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos_ordenados = sorted(contornos, key=cv2.contourArea, reverse=True)

# Verificación visual: el contorno 0 es el fondo blanco del QR
img_copy = img_color.copy()
cv2.drawContours(img_copy, contornos_ordenados, 0, (0, 255, 0), 6)
plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
plt.show()

contorno_qr = contornos_ordenados[0]
puntos = contorno_qr.reshape(-1, 2)  # de (N, 1, 2) a (N, 2) para operar por columnas

# Los 4 vértices del QR son los puntos extremos del contorno en cada dirección
arriba    = puntos[puntos[:, 1].argmin()]  # menor y
abajo     = puntos[puntos[:, 1].argmax()]  # mayor y
izquierda = puntos[puntos[:, 0].argmin()]  # menor x
derecha   = puntos[puntos[:, 0].argmax()]  # mayor x

print(f"Arriba:    {arriba}")
print(f"Abajo:     {abajo}")
print(f"Izquierda: {izquierda}")
print(f"Derecha:   {derecha}")

# Verificación visual de los 4 vértices detectados
img_copy = img_color.copy()
for punto, label, color in [
    (arriba,    'arriba',    (255, 0, 0)),
    (abajo,     'abajo',     (0, 255, 0)),
    (izquierda, 'izquierda', (0, 0, 255)),
    (derecha,   'derecha',   (255, 255, 0))
]:
    cv2.circle(img_copy, tuple(punto), 8, color, -1)
    cv2.putText(img_copy, label, (punto[0]+10, punto[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
plt.show()

# Tamaño de salida: distancia euclídea entre vértices opuestos del QR
ancho = int(np.linalg.norm(arriba - derecha))
alto  = int(np.linalg.norm(arriba - izquierda))
print(f"ancho={ancho}, alto={alto}")

# Homografía: mapea los 4 vértices del QR rotado a las esquinas de un rectángulo recto
pts_src = np.float32([arriba, derecha, abajo, izquierda])
pts_dst = np.float32([
    [0, 0],        # arriba    → esquina superior izquierda
    [ancho, 0],    # derecha   → esquina superior derecha
    [ancho, alto], # abajo     → esquina inferior derecha
    [0, alto]      # izquierda → esquina inferior izquierda
])

H = cv2.getPerspectiveTransform(pts_src, pts_dst)
qr_rectificado = cv2.warpPerspective(img_color, H, (ancho, alto))

plt.imshow(cv2.cvtColor(qr_rectificado, cv2.COLOR_BGR2RGB))
plt.show()