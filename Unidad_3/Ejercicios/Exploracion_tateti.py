import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)


img = cv2.imread("Unidad_3/Ejercicios/tateti_2.png")
imshow(img)

blur = cv2.GaussianBlur(img, (3,3), 0)
canny = cv2.Canny(blur, threshold1=80, threshold2=120)


plt.figure()
ax = plt.subplot(221)
imshow(img, new_fig=False, title="imagen original")
plt.subplot(222, sharex=ax, sharey=ax), imshow(blur, new_fig=False, title="")
plt.subplot(223, sharex=ax, sharey=ax), imshow(canny, new_fig=False, title="Canny")
plt.show(block=False)

lines = cv2.HoughLines(canny, rho=1, theta=np.pi/180, threshold=380)
print(lines.shape)

#imprimimos los rho (distancia desde el origen hasta la linea de pixeles) 
# y theta (angulo de la linea, 90=horizontal, 0=vertical.)
for i in range(len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1] * 180 / np.pi    # pasamos de radianes a grados.
    print(f"rho={rho:.1f}  theta={theta:.1f}°")

# lines[:,0,0] todos los rho
# lines[:,0,1] todos los theta
# lines[3,0,0] el rho de la linea 3

thetas = lines[:,0,1] * 180 / np.pi
horizontales = lines[thetas > 45]
verticales = lines[thetas <= 45]

rhos_h = np.sort(horizontales[:,0,0])
rhos_v = np.sort(verticales[:,0,0])

print("Horizontales:", rhos_h)
print("Verticales:", rhos_v)

def agrupar_lineas(rhos, gap=10):
    grupos, grupo_actual = [], [rhos[0]]
    for r in rhos[1:]:
        if r - grupo_actual[-1] < gap:
            grupo_actual.append(r)
        else:
            grupos.append(np.mean(grupo_actual))
            grupo_actual = [r]
    grupos.append(np.mean(grupo_actual))
    return np.array(grupos)

rhos_h = agrupar_lineas(np.sort(horizontales[:,0,0]))
rhos_v = agrupar_lineas(np.sort(np.abs(verticales[:,0,0])))

print("Horizontales:", rhos_h)
print("Verticales:", rhos_v)

img_lineas = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()

for rho in rhos_h:
    y = int(round(rho))
    cv2.line(img_lineas, (0, y), (img.shape[1], y), (255, 0, 0), 2)

for rho in rhos_v:
    x = int(round(rho))
    cv2.line(img_lineas, (x, 0), (x, img.shape[0]), (255, 0, 0), 2)

plt.figure()
imshow(img_lineas, new_fig=False, title="4 líneas detectadas", color_img=True)
plt.show(block=False)

#obtener lineas h y v, 4 cada una
#y: 0, rho_h[0], rhos_h[1], img.shape[0]
#x: 0, rho_v[0], rhos_v[1], img.shape[1]

#bordes para cortar celdas, inicio de imagen, 1ra linea, 2da linea, borde final de la imagen
bordes_h = np.array([0, rhos_h[0], rhos_h[1], img.shape[0]], dtype=int) #convertimos a int para poder hacer el crop de la imagen
bordes_v = np.array([0, rhos_v[0], rhos_v[1], img.shape[1]], dtype=int)

fig, axes = plt.subplots(3,3)
#hacemos crop para obtener las celdas.
for i in range(3):
    for j in range(3):
        y1, y2 = bordes_h[i], bordes_h[i+1]
        x1, x2 = bordes_v[j], bordes_v[j+1]
        crop = img[y1:y2,x1:x2]
        axes[i,j].imshow(crop, cmap='gray')
        axes[i,j].set_title(f"{i},{j}")
plt.show()

#Chequeo circulos
celda = img[bordes_h[0]:bordes_h[1], bordes_v[2]:bordes_v[3]]
celda_gray = cv2.cvtColor(celda, cv2.COLOR_BGR2GRAY)
celda_blur = cv2.GaussianBlur(celda_gray, (5,5), 1)
circles = cv2.HoughCircles(celda_blur, method=cv2.HOUGH_GRADIENT, dp=1, minDist=10, param2=40, minRadius=30)
print(circles)
#devuelve coordenadas (x, y, radio)

celda_color = cv2.cvtColor(celda_gray, cv2.COLOR_GRAY2BGR)

#Desempaqueto el primer circulo.
c = circles[0][0]
#dibujo el circulo coord, x,y como tupla y radio
cv2.circle(celda_color, (int(c[0]), int(c[1])), int(c[2]), (0,255,0), 2)

plt.figure()
imshow(celda_color, title='Circulo')
plt.show()

#Identificar lineas diagonales.
celda = img[bordes_h[1]:bordes_h[2], bordes_v[0]:bordes_v[1]]
celda_gray = cv2.cvtColor(celda, cv2.COLOR_BGR2GRAY)

plt.figure()
imshow(celda_gray)
plt.show()

w1 = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])   # +45°
w2 = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])   # -45°

f1 = cv2.filter2D(celda_gray, cv2.CV_64F, w1)
f2 = cv2.filter2D(celda_gray, cv2.CV_64F, w2)

f1 = np.abs(f1)
f2 = np.abs(f2)

print(f"Max filtro +45°: {f1.max():.1f}")
print(f"Max filtro -45°: {f2.max():.1f}")

#Funcion final.
fig, axes = plt.subplots(3, 3, figsize=(8,8))

w1 = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])   # +45°
w2 = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])   # -45°

for i in range(3):
    for j in range(3):
        y1, y2 = bordes_h[i], bordes_h[i+1]
        x1, x2 = bordes_v[j], bordes_v[j+1]
        celda = img[y1:y2, x1:x2]
        celda_gray = cv2.cvtColor(celda, cv2.COLOR_BGR2GRAY)
        celda_blur = cv2.GaussianBlur(celda_gray, (5,5), 1)

        # 1. Buscar círculo
        circles = cv2.HoughCircles(celda_blur, method=cv2.HOUGH_GRADIENT, dp=1, minDist=50, param2=30)
        if circles is not None:
            etiqueta = "Círculo"
        else:
            # 2. Buscar cruz
            f1 = np.abs(cv2.filter2D(celda_gray, cv2.CV_64F, w1))
            f2 = np.abs(cv2.filter2D(celda_gray, cv2.CV_64F, w2))
            if f1.max() > 500 and f2.max() > 500:
                etiqueta = "Cruz"
            else:
                etiqueta = "Vacío"

        axes[i,j].imshow(cv2.cvtColor(celda, cv2.COLOR_BGR2RGB))
        axes[i,j].set_title(etiqueta)
        axes[i,j].axis('off')

plt.tight_layout()
plt.show(block=False)

#-----------------------------tatetis con degrade -------------------------------------------

img = cv2.imread("Unidad_3/Ejercicios/tateti_8.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#vamos a convertir la imagen en escala de grises para utilizar otsu
thresh, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Umbral calculado: {thresh}")


plt.figure()
plt.subplot(121), plt.imshow(img_gray, cmap='gray'), plt.title("Original")
plt.subplot(122), plt.imshow(bw, cmap="gray"), plt.title("Otsu")
plt.show()


#Pruebo celda vacia (porque me devuelve cruz por error)
# probá con una celda que sepas que está vacía




