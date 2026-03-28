import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

"""1.1 Carga de la imagen de entrada 
a) Cargar la imagen desde el archivo img_calculadora.tif y mostrarla en una figura. 
b) Determinar sus dimensiones y el tipo de dato con el cual se representa el valor de 
cada píxel. 
c) Determinar el valor mínimo y máximo del nivel de grises de la imagen. 
d) Hallar todos los valores de nivel de grises que tiene la imagen. ¿Cuántos son? 
e) ¿Cuál es el valor de nivel de gris con menor repetitividad? ¿Cuál es el valor de nivel 
de gris con mayor repetitividad? Considere que, en ambos casos, pueden ser más 
de uno. En tal caso, mostrarlos todos. """

img = cv2.imread("Unidad_1/Codigo-20260303/calculadora.tif", cv2.IMREAD_GRAYSCALE)
plt.figure(),plt.imshow(img, cmap='gray'), plt.show(block=False)




""".2 Segmentación con ROI 
f) 
Recortar las teclas con las etiquetas: ‘SIN’, ‘COS’ y ‘TAN’. Los tres recortes deben ser 
del mismo tamaño. Mostrar los recortes en una nueva figura utilizando subplots (uno 
para cada región recortada), con los títulos acordes. 
g) Pegar cada recorte en una copia de la imagen original, de manera tal que las teclas 
de la calculadora queden ordenadas con la secuencia: | ‘TAN’ |  | ‘COS’ |  | ‘SIN’ |. 
h) A partir de la imagen resultante, recortar la tecla con etiqueta ‘ENTER’ y pegarla en el 
lugar de la tecla con etiqueta ‘COS’ (tenga presente que ambas teclas tienen 
diferentes tamaños). 
i) 
Aplicar modificaciones en la imagen resultante de manera tal que las teclas ‘4’, ‘5’, ‘6’, 
‘7’, ‘8’ y ‘9’ posean sus etiquetas numéricas pintadas de gris. Por ejemplo, cada 
etiqueta numérica con valor = 170."""