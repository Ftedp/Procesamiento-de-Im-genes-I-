import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

""""Dada la imagen texto.png, que se muestra en la Figura 1, desarrollar el código 
necesario para generar una imagen que resalte con recuadros rojos las letras. 
Opcionalmente, puede generar 2 imágenes más: 
1. Una imagen que resalte con recuadros azules las palabras. 
2. Una imagen que resalte con recuadros verdes los párrafos"""

img = cv2.imread("Unidad_1/Codigo-20260303/texto.png", cv2.IMREAD_GRAYSCALE)
plt.figure(),plt.imshow(img, cmap='gray'), plt.show(block=False)

#imagen binaria
img_bin = img < 200
plt.figure(),plt.imshow(img_bin, cmap='gray'), plt.show(block=False)

#linea renglones
img_row_zeros = img_bin.any(axis=1) #aplasto columnas 
img_row_zeros_inds = np.argwhere(img_bin.any(axis=1)) # identifica los indices
plt.figure(),plt.plot(img_row_zeros), plt.show(block=False) 

#renglones en imagen
xr = img_row_zeros*(img.shape[1]-1) #multiplico el binario (donde hay letras) por el ancho de la imagen (img.shape[1]-1)
yr = np.arange(img.shape[0]) # genera lista de numeros (secuencial) por el alto de la imagen
plt.figure(),plt.imshow(img, cmap='gray'), plt.plot(xr,yr, c='r'), plt.show(block=False)

# inicio y final de cada renglon
x = np.diff(img_row_zeros) #np.diff detecta los cambios entre true y false de la lista.
renglones_inds=np.argwhere(x) #indices donde ocurrieron esos cambios (comienzo y fin de linea)
len(renglones_inds)

pos_ini = np.arange(0,len(renglones_inds),2) #selecciono posiciones de inicio
renglones_inds[pos_ini]+=1 # reacomodo (corrijo) posicion

'''shape[0] alto de la imagen (filas)
    shape[1] ancho de la imagen (columnas)'''
#Mascara visual
xri = np.zeros(img.shape[0]) #creo columna llena de ceros del alto de la imagen
xri[renglones_inds] = (img.shape[1]-1) #en los indices de los renglones agrego valor hasta el final del ancho de la imagen (en lugar de un 0)
yri = np.arange(img.shape[0]) # yri es el alto de la imagen
plt.figure(), plt.imshow(img, cmap='gray'), plt.plot(xri, yri, 'r'), plt.title("Renglones - Inicio y Fin"), plt.show(block=False)                    

#transformamos un vector en una matriz de 2 columnas
ind_fila = np.reshape(renglones_inds, (-1,2)) #en lugar de una lista de indices de renglon obtengo una matriz agrupando de a 2 (ini,fin)


renglones = []
for ind_r, inds in enumerate(ind_fila):
    renglones.append({ # lista de diccionarios
        "ind_r": ind_r+1, # contador
        "cord": inds, # par de indices verticales de renglon (ini,fin)
        "img": img[inds[0]:inds[1], :] # slicing de (ini,fin) alto, y todo el ancho 
    })

#Visualizacion renglones
plt.figure()
for renglon in renglones:
    plt.subplot(len(renglones), 1, renglon["ind_r"]) # (filas, columnas, pos_subplot_actual)
    plt.imshow(renglon["img"], cmap='gray') #accede al recorte de imagen guardado en renglones
    plt.xticks([])
    plt.yticks([])
plt.show(block=False)

#Deteccion de letras por renglon
# detecto donde hay tinta, ajusto verticalmente 

letras = []
ind_letra = -1

for renglon in renglones:
    renglon_bin = renglon['img'] < 200

    ren_col_zeros = renglon_bin.any(axis=0) # aplasto filas para notar donde hay letras(true)
    ren_col_zeros_inds = np.argwhere(ren_col_zeros) #obtengo los indices de donde hay letra

    x = np.diff(ren_col_zeros) #detecta el cambio (inicio y fin de cada letra)
    letras_inds = np.argwhere(x) # obtengo dicho indices

    ii = np.arange(0, len(letras_inds),2) #obtengo indices que representan inicios
    letras_inds[ii] +=1 # sumo uno a cada indice de inicio (los corrijo)

    letras_inds = letras_inds.reshape(-1,2) #-1 no se cuantas filas hay, pero quiero 2 columnas

    for ind_ren_letra, inds in enumerate(letras_inds):
        ind_letra +=1

        letra_tmp = renglon["img"][:, inds[0]:inds[1]] #Por todo el alto de renglon actual obtengo el ancho de la letra (slicing)
        letra_bin = letra_tmp < 200 # convierto el binario,

        filas_con_letra = letra_bin.any(axis=1) #aplasto columnas para ver en que filas hay letra
        filas_inds = np.argwhere(filas_con_letra) # obtengo los indices de dichas filas

        fila_ini = filas_inds[0,0] #defino techo (primer indice)
        fila_fin = filas_inds[-1,0]+1 #defino piso (ultimo indice), +1 para no dejar ultimo indice afuera.

        y_ini_global = renglon["cord"][0] + fila_ini #Inicio de letra vertical
        y_fin_global = renglon["cord"][0] + fila_fin #Fin de letra vertical
        x_ini_global = inds[0] #indice de inicio horizontal de letra
        x_fin_global = inds[1] #indice de final horizontal de letra

        letras.append({
            "ind_r": renglon["ind_r"],
            "ind_ren_letra": ind_ren_letra+1,
            "ind_letra": ind_letra,
            "cord": [y_ini_global, x_ini_global, y_fin_global, x_fin_global],
            "img": img[y_ini_global:y_fin_global, x_ini_global:x_fin_global]
        })

print("Cantidad de letras: ", len(letras))

#Visualizacion de letras
plt.figure(), plt.imshow(img, cmap='gray')
for letra in letras:
    yi = letra["cord"][0]
    xi = letra["cord"][1]
    H = letra["cord"][2] - letra["cord"][0]
    W = letra["cord"][3] - letra["cord"][1]
    rect = Rectangle((xi,yi), W, H, linewidth=1, edgecolor='r',facecolor='none') #Parametros(x,y),ancho,alto
    ax= plt.gca()
    ax.add_patch(rect)
plt.show(block=False)

#Agrupacion de letras en palabra
# Distancias horizontales entre letras consecutivas por renglón
for renglon in renglones:
    letras_renglon = [letra for letra in letras if letra["ind_r"] == renglon["ind_r"]]

    distancias_letras = []
    for i in range(len(letras_renglon) - 1):
        x_fin_actual = letras_renglon[i]["cord"][3]
        x_ini_sig = letras_renglon[i + 1]["cord"][1]
        dist = x_ini_sig - x_fin_actual
        distancias_letras.append(dist)

    print(f"Renglón {renglon['ind_r']} -> distancias entre letras: {np.unique(distancias_letras)}")

#umbral 9
palabras = []
ind_p = -1
umbral_palabra = 9
for ind_r, renglon in enumerate(renglones):

    renglon_bin = renglon["img"] < 200
    kernel = np.ones((1, umbral_palabra), np.uint8)
    renglon_palabras = cv2.dilate(renglon_bin.astype(np.uint8), kernel, iterations=1)

    ren_col_palabras = renglon_palabras.any(axis=0)
    ren_col_palabras_inds = np.argwhere(ren_col_palabras)

    x = np.diff(ren_col_palabras)
    palabras_inds = np.argwhere(x)

    ii = np.arange(0, len(palabras_inds), 2)
    palabras_inds[ii] +=1

    palabras_inds = palabras_inds.reshape((-1, 2))

    for ind_r_p, inds in enumerate(palabras_inds):
        ind_p +=1

        palabras_tmp = renglon["img"][:,inds[0]:inds[1]]
        palabras_bin = palabras_tmp < 200
        
        filas_con_palabra = palabras_bin.any(axis=1)
        fila_inds = np.argwhere(filas_con_palabra)

        fila_ini = fila_inds[0, 0]
        fila_fin = fila_inds[-1, 0] + 1

        y_ini_global = renglon["cord"][0] + fila_ini
        y_fin_global = renglon["cord"][0] + fila_fin
        x_ini_global = inds[0]
        x_fin_global = inds[1]

        palabras.append({
            "ind_p": ind_p + 1,
            "ind_r": renglon["ind_r"],
            "cord": [y_ini_global, x_ini_global, y_fin_global, x_fin_global],
            "img": img[y_ini_global:y_fin_global, x_ini_global:x_fin_global]
        })

plt.figure(), plt.imshow(img, cmap='gray')
for palabra in palabras:
    yi = palabra["cord"][0]
    xi = palabra["cord"][1]
    W = palabra["cord"][2] - palabra["cord"][0]
    H = palabra["cord"][3] - palabra["cord"][1]
    rect = Rectangle((xi, yi), H, W, linewidth=2, edgecolor='b', facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
plt.title("Palabras detectadas")
plt.show()

#distancia vertical entre renglones
distancias = []
for i in range (len(renglones) - 1):
    fin_actual = renglones[i]['cord'][1] #tomamos el final del renglon i
    inicio_sig = renglones[i+1]['cord'][0] # tomamos el inicio del renglon i+1
    distancias.append(inicio_sig - fin_actual) # medimos la distancia entre ambos.

print("Distancias entre renglones:", distancias)

#Identificacion de parrafos
umbral_parrafo = 40
parrafos = []
parrafo_actual = [renglones[0]] #inicio primer parrafo con primer linea

for i in range (len(renglones) - 1): # recorro hasta la anteultima para que no falle la comparacion
    if distancias[i] > umbral_parrafo: # comparo distancias entre lineas
        parrafos.append(parrafo_actual) #Cierro parrafo actual
        parrafo_actual = [renglones[i + 1]] # Creo un nuevo parrafo en la pos actual.
    else:
        parrafo_actual.append(renglones[i + 1]) #Agrego una linea al parrafo actual.
parrafos.append(parrafo_actual) # Cierro ultimo parrafo

parrafos_data = [] # Lista de diccionarios por parrafo

for ip, parrafo in enumerate(parrafos): 
    y_ini = parrafo[0]['cord'][0] # Accedo al 1er indice del primer renglon (coordenada vertical inicial)
    y_fin = parrafo[-1]['cord'][1] #Accedo al 2do indice del ultimo renglon (coordenada vertical final)

    parrafos_data.append({
        "ip": ip+1, #indice de parrafo
        "renglones": parrafo, # renglones que componen ese parrafo
        "cord": [y_ini,0,y_fin, img.shape[1]], #definimos el area rectangular del parrafo
        "img": img[y_ini:y_fin,:] #Crop del parrafo de la imagen.
    })

print("Cantidad de parrafos: ", len(parrafos_data))

# Visualizacion parrafos
plt.figure(),plt.imshow(img, cmap='gray')
for parrafo in parrafos_data:
    yi = parrafo["cord"][0] #posicion vertical inicial
    xi = parrafo["cord"][1] #pos horizontal inicial
    H = parrafo["cord"][2] - parrafo["cord"][0] # (y_fin - y_ini)
    W = parrafo["cord"][3] - parrafo["cord"][1] # img.shape[1] - 0 (ancho de imagen - 0)
    rect = Rectangle((xi,yi), W, H, linewidth=2, edgecolor='g', facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)

plt.show(block=False)



































