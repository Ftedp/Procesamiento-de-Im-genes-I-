import matplotlib.pyplot as plt
import cv2
import numpy as np
import roipoly as roi

""""    1.1 Carga de la imagen de entrada 
        a) Cargar la imagen desde el archivo faces.jpg y mostrarla en una figura. """

img = cv2.imread("Unidad_2/Ejercicios/faces.jpg")
plt.imshow(img), plt.show(block=False)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img), plt.show(block=False)

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.figure(), plt.imshow(img, cmap='gray'), plt.show(block=False)
"""b) Obtener y mostrar la información básica de la imagen 
    (tipo de dato, dimensiones, 
    valores máximos y mínimos de intensidad en cada canal).  """

img.dtype
img.shape
h,w,c= img.shape

img.min()
img.max()
np.unique(img)
len(np.unique(img))

""" 1.2 Filtrado manual.
    c) Ubicar cada rostro de la imagen, encerrarlos en un rectángulo (bounding box) y 
    mostrarlos en una nueva figura (obtener las coordenadas de los rostros de forma 
    manual). """
coors = [
    (60, 122, 122, 162),
    (32, 82, 251, 290),
    (52, 91, 382, 418),
    (84, 130, 472, 514)
]
crops = []
for (y1, y2, x1, x2) in coors:
    crops.append(img[y1:y2, x1:x2].copy())
#ejes que marca x,y son y,x
# f1_crop=img[60:122,122:162].copy()
# f2_crop=img[32:82,251:290].copy()
# f3_crop=img[52:91,382:418].copy()
# f4_crop=img[84:130,472:514].copy()

img_output = img.copy()
#(x,y)ini, (x,y)fin
cv2.rectangle(img_output, (122, 60),  (162, 122), (0, 255, 0), 2) # Para f1
cv2.rectangle(img_output, (251, 32),  (290, 82),  (0, 255, 0), 2) # Para f2
cv2.rectangle(img_output, (382, 52),  (418, 91),  (0, 255, 0), 2) # Para f3
cv2.rectangle(img_output, (472, 84),  (514, 130), (0, 255, 0), 2) # Para f4
cv2.imshow("Rectangulos", img_output)

"""d)   Tomando como base la imagen original, recortar y mostrar los rostros en una única 
        figura utilizando subplots para cada rostro. """

plt.figure()
for i, cara in enumerate(crops):
    #subplot (filas,columnas,indice)
    plt.subplot(2,2,i+1)
    cara = cv2.cvtColor((cara), cv2.COLOR_BGR2RGB)
    plt.imshow(cara)
    plt.title(f"Caras {i+1}")

plt.show()

""""e)  Filtrar cada recorte con el método cv2.blur(), 
        utilizando los parámetros adecuados para 
        obtener el efecto de borrosidad deseado, y 
        mostrar cada resultado en una única figura 
        con el uso de subplots para cada rostro."""
k = 5
plt.figure()
for i, cara in enumerate(crops):
    cara_blur = cv2.blur(cara, (k,k))
    plt.subplot(2,2,i+1)
    cara_blur=cv2.cvtColor(cara_blur,cv2.COLOR_BGR2RGB)
    plt.imshow(cara_blur)
    plt.title(f"BLUR {i+1}")

plt.show(block=False)

""" 
f) 
    Pegar cada recorte con efecto de borrosidad en las posiciones 
    correspondientes de la imagen original y 
    mostrar el resultado en una nueva figura."""    

img_final = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
k=15
for (y1,y2,x1,x2) in coors:
    roi=img_final[y1:y2,x1:x2]

    img_blur = cv2.blur(roi, (k,k))

    img_final[y1:y2,x1:x2] = img_blur

plt.figure()
plt.imshow(img_final)
plt.title("caras blur")    
plt.show()

""" 1.3 Filtrado automático 
g)  A partir de la imagen original, convertir la misma a escala 
    de grises con el método cv2.cvtColor(), mostrarla en una
    figura y obtener su información básica (tipo de dato, 
    dimensiones, valores máximos y mínimos de intensidad)."""

img = cv2.imread("Unidad_2/Ejercicios/faces.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure, plt.imshow(img, cmap="gray"), plt.show(block=False)

img.dtype
img.shape
img.min()
img.max()

np.unique(img)
len(np.unique(img))

""") Ubicar cada rostro mediante el uso de un modelo pre-entrenado
     de detección de rostros basado en un clasificador en
     Haar Cascade (ver documentación): 
    i) Crear el objeto clasificador con cv2.CascadeClassifier(). 
    ii)Cargar el modelo cv2.data.haarcascades + 
    haarcascade_frontalface_alt.xml con el método load() del objeto creado. 
    iii) Aplicar el clasificador sobre la imagen en escala de grises 
        con el método detectMultiScale() del objeto creado. """

face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
faces = face_cascade.detectMultiScale(img)

subfaces=[]
img_result=img.copy()

if len(faces):
    for(x,y,w,h) in faces:
        cv2.rectangle(img_result, (x,y), (x+w,y+h), (0,255,255),2)    
        subface = img[y:y+h, x:x+w].copy()
        subfaces.append(subface)

plt.figure()
plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
plt.show()

"""
Tomando como base la imagen original, recortar y mostrar 
los rostros detectados en una única figura utilizando 
subplots para cada rostro"""
n_subfaces=len(subfaces)
if n_subfaces > 0:
    for i in range(n_subfaces):
        plt.subplot(2,2,i+1)
        plt.imshow(cv2.cvtColor(subfaces[i], cv2.COLOR_BGR2RGB))

plt.show()

"""k) Filtrar cada recorte con el método cv2.GaussianBlur(), utilizando los parámetros 
adecuados para obtener el efecto de borrosidad deseado y mostrar cada resultado en 
una única figura con el uso de subplots para cada rostro. """

subfaces_blur=[]

for i, subface in enumerate(subfaces):
    
    blur_face = cv2.GaussianBlur(subface,(15, 15),0)
    subfaces_blur.append(blur_face)
    
    plt.subplot(2,2,i+1)
    plt.imshow(blur_face, cmap='gray')

plt.show()

""""l) Pegar cada recorte con efecto de borrosidad en las posiciones correspondientes de la 
imagen original y mostrar el resultado en una nueva figura. """

img_final = img.copy()

for i, (x,y,w,h) in enumerate(faces):
    img_final[y:y+h,x:x+w] = subfaces_blur[i]

plt.figure(), plt.imshow(img_final, cmap='gray')
plt.show(block=False)

























