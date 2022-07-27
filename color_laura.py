import os
import cv2 as cv
import random
import numpy as np
from scipy.stats import pearsonr


def divide_image(image, n):

    img_array = [] # Array donde estará cada renglón de la imagen dividida
    height, width = image.shape[:2] # Medidas de la imagen en pixeles
    rowSize = height//n

    initialRow = 0 # Numero de fila donde se va a comenzar a dividir
    finalRow = rowSize # Donde terminará el primer recorte

    for x in range(0, n):

        imageOut = image[initialRow:finalRow, 0:width] # Recorte de la imagen

        imageOut = cv.cvtColor(imageOut,cv.COLOR_BGR2LAB)
        
        img_array.append(imageOut) # Se agrega al array

        initialRow += rowSize # Se actualiza la fila inicial para el siguiente recorte
        finalRow += rowSize # Se actualiza el la fila final para el siguiente recorte

    return img_array # Retorna el arreglo con cada parte de la imagen recortada

#Método que compara dos imágenes a través de la correlación de pearson
def compare_image(image1, image2, num_patches, xoffset, yoffset):

    #image1 = cv.cvtColor(image1,cv.COLOR_BGR2LAB)
    #image2 = cv.cvtColor(image2,cv.COLOR_BGR2LAB)

    height, width = image1.shape[:2] # Medidas de la imagen en pixeles

    image1 = image1[yoffset:height-yoffset, xoffset:width-xoffset] # Recorte de la imagen
    image2 = image2[yoffset:height-yoffset, xoffset:width-xoffset]

    if num_patches > 0:
        patches1 = divide_image(image1, num_patches)
        patches2 = divide_image(image2, num_patches)

        diffs = []

        for image1, image2 in zip(patches1, patches2):
            L_1, a_1, b_1 = cv.split(image1)

            a_flat1 = a_1.flatten()
            b_flat2 = b_1.flatten()
            L_flat3 = L_1.flatten()

            sum1 = a_flat1 + b_flat2

            L_2, a_2, b_2 = cv.split(image2)

            a2_flat1 = a_2.flatten()
            b2_flat2 = b_2.flatten()
            L2_flat3 = L_2.flatten()

            sum2 = a2_flat1 + b2_flat2

            corr, _ = pearsonr(sum1, sum2)

            diffs.append(corr)
        return np.mean(diffs)
    else:
        image1 = cv.cvtColor(image1,cv.COLOR_BGR2LAB)
        image2 = cv.cvtColor(image2,cv.COLOR_BGR2LAB)
    #Separa los canales de la imagen 1
        L_1, a_1, b_1 = cv.split(image1)

        #Colapsa las dimensiones de la imagen 1 para crear un vector
        a_flat1 = a_1.flatten()
        b_flat2 = b_1.flatten()
        L_flat3 = L_1.flatten()
        
        #Realiza una sumatoria de los canales A y B
        #sum1 = a_flat1 + b_flat2
        
        #Separa los canales de la imagen 2
        L_2, a_2, b_2 = cv.split(image2)
    
        #Colapsa las dimensiones de la imagen 2 para crear un vector
        a2_flat1 = a_2.flatten()
        b2_flat2 = b_2.flatten()
        L2_flat3 = L_2.flatten()
        
        #Realiza la sumatoria de los canales A y B
        #sum2 = a2_flat1 + b2_flat2 
        
        #Método de correlación en numpy
        #coeficiente = np.corrcoef(sum1,sum2)

        corr1, _ = pearsonr(a_flat1, a2_flat1)
        corr2, _ = pearsonr(b_flat2, b2_flat2)

        return np.mean([corr1, corr2])

        return pearsonr(sum1,sum2)[0] #Correlación de pearson con scipy



def comparacion_de_canales (imagen_1, imagen_2):
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.stats import pearsonr

    imagen_RGB_1 = cv2.cvtColor(imagen_1, cv2.COLOR_BGR2RGB)
    imagen_gris_1 = cv2.cvtColor(imagen_RGB_1, cv2.COLOR_BGR2GRAY)

    imagen_RGB_2 = cv2.cvtColor(imagen_2, cv2.COLOR_BGR2RGB)
    imagen_gris_2 = cv2.cvtColor(imagen_RGB_2, cv2.COLOR_BGR2GRAY)

    u_1, th_1 = cv2.threshold(imagen_gris_1,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    u_2, th_2 = cv2.threshold(imagen_gris_2,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    imagenes= [imagen_RGB_1, imagen_RGB_2]
    mascaras = [th_1, th_2]

    imagenes_procesadas = []    #Guardamos las imagenes resultantes
    for i in range (len(mascaras)):
        imagen = cv2.bitwise_and(imagenes[i], imagenes[i], mask= mascaras[i])
        imagenes_procesadas.append(imagen)
        
    imagenes_Lab = []   #Guardamos las imagenes en el modelo Lab
    for i in range(len(imagenes_procesadas)):
        imagenBGR = cv2.cvtColor(imagenes_procesadas[i], cv2.COLOR_RGB2BGR)
        imagenLab = cv2.cvtColor(imagenBGR, cv2.COLOR_BGR2Lab)
        imagenes_Lab.append(imagenLab)

    histogramas_a= []
    histogramas_b = []
    for i in range(len(imagenes_Lab)):
        imagen = imagenes_Lab[i]
        L, a, b = cv2.split(imagen)
        a_flat = a.flatten()
        b_flat = b.flatten()
        histogramas_a.append(a_flat)
        histogramas_b.append(b_flat)
        
    corr, _ = pearsonr(histogramas_a[0], histogramas_a[1]) 
    corr2, _ = pearsonr(histogramas_b[0], histogramas_b[1])

    return np.mean([corr,corr2])


