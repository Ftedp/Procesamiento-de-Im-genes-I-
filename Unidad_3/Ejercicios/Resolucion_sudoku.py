import cv2
import matplotlib.pyplot as plt 
import numpy as np

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

img = cv2.imread('Unidad_3/Ejercicios/sudoku.jpeg', cv2.IMREAD_GRAYSCALE)
imshow(img, title='Sudoku')

print(img.shape)
print(img.dtype)
print(img.min(),img.max())

blur = cv2.GaussianBlur(img, (3,3), 0)
gcan = cv2.Canny(blur, threshold1=80, threshold2=120)


plt.figure()
ax = plt.subplot(221)
imshow(img, new_fig=False, title="imagen original")
plt.subplot(222, sharex=ax, sharey=ax), imshow(blur, new_fig=False, title="")
plt.subplot(223, sharex=ax, sharey=ax), imshow(gcan, new_fig=False, title="Canny")
plt.show(block=False)

lines = cv2.HoughLines(gcan, rho=1, theta=np.pi/180, threshold=250)
print(lines.shape)

img_lineas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # paso a color para dibujar en azul

for i in range(len(lines)):
    rho   = lines[i][0][0]   # distancia
    theta = lines[i][0][1]   # ángulo
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img_lineas, (x1,y1), (x2,y2), (255,0,0), 2)

plt.figure()
imshow(img_lineas, new_fig=False, title="Lineas detectadas", color_img=True)
plt.show(block=False)

for i in range(len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1] * 180 / np.pi
    print(f"rho={rho:1f}    theta={theta:.1f}°")


thetas = lines[:,0,1] * 180 / np.pi
horizontales = lines[thetas > 45] #horizontales
verticales = lines[thetas < 45] #verticales

print(f"Horizontales: {len(horizontales)}")
print(f"verticales: {len(verticales)}")

rho_sort = np.sort(horizontales[:,0,0])
print(rho_sort)

grupos_h = []
grupos_actual = [rho_sort[0]]

for rho_h in rho_sort[1:]:
    if rho_h - grupos_actual[-1] < 10:
        grupos_actual.append(rho_h)
    else:
        grupos_h.append(np.mean(grupos_actual))
        grupos_actual = [rho_h]

grupos_h.append(np.mean(grupos_actual))

print(grupos_h)
rhos_h = np.array(grupos_h)

rho_sort = np.sort(verticales[:,0,0])
print(rho_sort)

grupos_v = []
grupos_actual = [rho_sort[0]]

for rho_v in rho_sort[1:]:
    if rho_v - grupos_actual[-1] < 10:
        grupos_actual.append(rho_v)
    else:
        grupos_v.append(np.mean(grupos_actual))
        grupos_actual = [rho_v]

grupos_v.append(np.mean(grupos_actual))

print(grupos_v)
rhos_v = np.array(grupos_v)

print(rhos_h)
print(rhos_v)


img_lineas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for rho in rhos_h:
    y = int(round(rho))
    cv2.line(img_lineas, (0, y), (img.shape[1], y), (255, 0, 0), 2)

for rho in rhos_v:
    x = int(round(rho))
    cv2.line(img_lineas, (x, 0), (x, img.shape[0]), (255, 0, 0), 2)

plt.figure()
imshow(img_lineas, new_fig=False, title="20 líneas detectadas", color_img=True)
plt.show(block=False)

celdas_vacias = 0
img_resultado = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for i in range(9):
    for j in range(9):
        y1, y2 = int(round(rhos_h[i])), int(round(rhos_h[i+1]))
        x1, x2 = int(round(rhos_v[j])), int(round(rhos_v[j+1]))
        celda = img[y1+3:y2-3, x1+3:x2-3]
        _, bw = cv2.threshold(celda, 127, 255, cv2.THRESH_BINARY_INV)
        n_labels, _, _, _ = cv2.connectedComponentsWithStats(bw)
        if n_labels == 1:
            celdas_vacias += 1
            img_resultado[y1:y2, x1:x2] = 128

print(f"Celdas vacías: {celdas_vacias}")
print(f"Celdas con número: {81 - celdas_vacias}")

plt.figure()
plt.subplot(121), plt.imshow(celda, cmap='gray'), plt.title("celda original")
plt.subplot(122), plt.imshow(bw, cmap='gray'), plt.title("umbralizada")
plt.show(block=False)

porcentaje = (81 - celdas_vacias) * 100 / 81
print(f"Porcentaje de avance: {porcentaje:.1f}%")

for rho in rhos_h:
    y = int(round(rho))
    cv2.line(img_resultado, (0, y), (img.shape[1], y), (255, 0, 0), 2)

for rho in rhos_v:
    x = int(round(rho))
    cv2.line(img_resultado, (x, 0), (x, img.shape[0]), (255, 0, 0), 2)
    
plt.figure()
imshow(img_resultado, new_fig=False, title=f"Avance: {porcentaje:.1f}%  |  Celdas vacías: {celdas_vacias}/81", color_img=True)
plt.show(block=False)


