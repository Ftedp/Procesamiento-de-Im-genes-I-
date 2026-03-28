import matplotlib.pyplot as plt
import numpy as np
import cv2
import roipoly as roi

"""La Figura 2 (archivo john_canny_bio.png) corresponde a una 
biografía de John F. Canny, en este ejercicio se requiere que, 
a partir de las técnicas de PDI vistas en clase, se recupere de 
forma automática el dato perdido debajo de la “mancha” que se observa 
en el pie de página de la misma. Tener en cuenta que es necesario no
perder detalles ni información de la biografía en el proceso de
recuperación. Una vez obtenido el dato perdido, se pide insertarlo en
la imagen original y mostrar el resultado en una nueva figura."""

"""Idea:
Usar laplaciano para detectar bordes de la manche, sobel para delimitar region,
pasa altos para resaltar la diferencia.
Antes hay que aislar la region con mascara (manual o algun algoritmo?)
"""
img = cv2.imread("Unidad_2/Ejercicios/john_canny_bio.png",cv2.IMREAD_GRAYSCALE)
plt.figure(), plt.imshow(img, cmap='gray'), plt.show(block=False)

gray = img.copy()

y1, y2 = 1783, 1852
x1, x2 = 140, 464
crop = gray[y1:y2,x1:x2].copy()
plt.figure(), plt.imshow(crop, cmap='gray'), plt.show(block=False)

crop_eq = cv2.equalizeHist(crop)
plt.figure(), plt.imshow(crop_eq, cmap='gray'), plt.show(block=False)

crop_inv = 255 - crop_eq
plt.figure(), plt.imshow(crop_inv, cmap='gray'), plt.show(block=False)

A = 1.8
w2 = -np.ones((5,5), dtype=np.float32)/(5*5)  
w2[2,2] = (25*A-1)/25
crop_final = cv2.filter2D(crop_inv,-1,w2)

plt.figure(), plt.imshow(crop_final, cmap='gray'), plt.show(block=False)

crop_final_rgb = cv2.cvtColor(crop_final, cv2.COLOR_GRAY2RGB)
# crop_final = cv2.convertScaleAbs(crop_final_rgb, alpha=2, beta=0)
img = cv2.imread("Unidad_2/Ejercicios/john_canny_bio.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_final = img.copy()

img_final[y1:y2,x1:x2] = crop_final_rgb
plt.figure(), plt.imshow(img_final), plt.show(block=False)



#pasaalto no sirve para el caso.
w1 = -np.ones((3,3))/(3*3)
w1[1,1]=8/9
w2 = -np.ones((5,5))/(5*5)  
w2[2,2]=24/25
img1 = cv2.filter2D(crop_inv,-1,w1)
img2 = cv2.filter2D(crop_inv,-1,w2)
plt.figure(), plt.imshow(img1, cmap='gray'), plt.show(block=False)
plt.figure(), plt.imshow(img2, cmap='gray'), plt.show(block=False)


#Laplaciano no sirve para el caso.
laplacian = cv2.Laplacian(crop_inv, cv2.CV_64F)
lap_result= crop_inv - laplacian
lap_result = cv2.convertScaleAbs(lap_result)
plt.figure(), plt.imshow(lap_result, cmap='gray'), plt.show(block=False)
