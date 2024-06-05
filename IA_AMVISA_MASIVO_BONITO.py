init_weights=True
weights=None
from imageai.Classification.Custom import CustomImageClassification
import os
from os import scandir, getcwd, listdir
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import easyocr
import cv2
import pandas as pd
import time
from datetime import date, time, datetime
import json
import warnings
import csv
import shutil
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

#Definicion rutas y variables generales
lista_raices_tiempo=[]
lista_material_acometida=[]
lista_material_raices=[]
lista_material_grieta=[]
lista_raices_probabilidad=[]
lista_acometida_tiempo=[]
lista_acometida_probabilidad=[]
lista_grieta_tiempo=[]
lista_grieta_probabilidad=[]
lista_video_raices=[]
lista_video_acometida=[]
lista_video_grieta=[]
tabla_conjunta=pd.DataFrame()
diametros=[]
dataframes_acometida = []
dataframes_raices = []
dataframes_grieta = []
num_decimales = 2
informe_csv_path = "./uploads/informe/FORMATO_INCIDENCIAS.csv"
informe_excel_path = "./uploads/informe/FORMATO_INCIDENCIAS.xlsx"
historial_csv_path = "./uploads/informe/Informe_Videos_Hechos.csv"
    
d = datetime.now()
tiempo=(d.strftime('%d-%m-%Y'))

with open('diametros.txt', 'r') as f:
    diametros = f.readlines()
    
diametros = [diametro.strip() for diametro in diametros]

#Funcion para limpiar las carpetas cada vez que se ejecuta el codigo de nuevo
def limpiar_directorio(ruta_directorio):
    archivos = os.listdir(ruta_directorio)
    for archivo in archivos:
        os.remove(os.path.join(ruta_directorio, archivo))

limpiar_directorio("./uploads/final/")

limpiar_directorio("./static/resultados/acometida_probable/")

limpiar_directorio("./static/resultados/raices_probable/")

limpiar_directorio("./static/resultados/grieta_probable/")

limpiar_directorio("./static/resultados/acometidas/")

limpiar_directorio("./static/resultados/raices/")

limpiar_directorio("./static/resultados/grieta/")

limpiar_directorio("./static/resultados/acometidas_cropped/")

limpiar_directorio("./static/resultados/raices_cropped/")

limpiar_directorio("./static/resultados/ultimo_frame/")

limpiar_directorio("./static/resultados/grieta_cropped/")
        
archivo=listdir("./video")

#Informe de videos mensuales procesados
nombres_videos_procesados = [archivos for archivos in archivo if archivos.endswith(".mp4")]

try:
    historial_df = pd.read_csv(historial_csv_path)
except pd.errors.EmptyDataError:
    historial_df = pd.DataFrame(columns=['Nombre del Video'])

nuevos_videos = pd.DataFrame({'Nombre del Video': nombres_videos_procesados, 'Fecha de ejecución': tiempo})
historial_df = pd.concat([historial_df, nuevos_videos], ignore_index=True)

historial_df.to_csv(historial_csv_path, index=False)

informe_df = pd.read_csv(informe_csv_path)
num_filas=informe_df.count(axis=1)
for i in range(len(num_filas)):
    informe_df.drop([i],axis=0,inplace=True)
    
informe_df.to_csv(informe_csv_path, index=False) #Si se quieren guardar los resultados en el tiempo, comentar esta linea

#Comienza el bucle de analisis de todos los videos
for prueba in range(len(archivo)):
    vidcap = cv2.VideoCapture("./video/"+archivo[prueba])
    success,image = vidcap.read()
    i = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(i*1000))
        cv2.imwrite("./images_prueba_cv/frame"+str(i)+".jpg", image)
        success,image = vidcap.read()
        i=i+1
    #Definicion de variables y carga de modelo, predicción de los resultados
    vidcap=0
    prediction = CustomImageClassification()
    prediction.setModelTypeAsResNet50()
    execution_path=os.getcwd()
    prediction.setModelPath(os.path.join(execution_path, "./models/resnet50-AMVISA-test_acc_0.77104_epoch-70.pt")) 
    prediction.setJsonPath(os.path.join(execution_path, "./models/AMVISA_model_classes.json"))
    prediction.loadModel()
    predictions=[]
    ultimo_frame=0
    probabilidad_acometida=[]
    probabilidad_raices=[]
    metros_acometida=[]
    metros_raices=[]
    probabilidad_grieta=[]
    segundos_grieta=[]
    segundos_acometida=[]
    segundos_raices=[]
    segundos_acometida_posible=[]
    segundos_raices_posible=[]
    segundos_grieta_posible=[]
    result_raices=[]
    result_grieta=[]
    result_acometida=[]
    probabilities=[]
    res_aco=[]
    res_gr=[]
    res_raiz=[]
    size = listdir("./images_prueba_cv/")
    for i in range(1,len(size)):
        predictions , probabilities = prediction.classifyImage(os.path.join(execution_path, "./images_prueba_cv/frame"+str(i)+".jpg"), result_count=4)
        for eachPrediction, eachProbability in zip(predictions, probabilities):
            if eachPrediction=='Acometidas' and eachProbability >=65.0000:
                probabilidad_aco=round(eachProbability, num_decimales)
                probabilidad_acometida.append(probabilidad_aco)
                segundos_acometida.append(i)
            elif eachPrediction=='Raices' and eachProbability >=65.0000:
                probabilidad_rai=round(eachProbability, num_decimales)
                probabilidad_raices.append(probabilidad_rai)
                segundos_raices.append(i)
            elif eachPrediction=='Grieta' and eachProbability >=65.0000:
                probabilidad_gr=round(eachProbability, num_decimales)
                probabilidad_grieta.append(probabilidad_gr)
                segundos_grieta.append(i)
            elif eachPrediction=='Acometidas' and (40.0000 < eachProbability < 65.0000):
                segundos_acometida_posible.append(i)
            elif eachPrediction=='Raices' and (40.0000 < eachProbability < 65.0000):
                segundos_raices_posible.append(str(i))
            elif eachPrediction=='Grieta' and (40.0000 < eachProbability < 65.0000):
                segundos_grieta_posible.append(str(i))
    
    #Definicion de variables y carga de modelo para predecir el material
    prediction_material = CustomImageClassification()
    prediction_material.setModelTypeAsResNet50()
    execution_path_material=os.getcwd()
    prediction_material.setModelPath(os.path.join(execution_path_material, "./Entrenamiento_Material/models/resnet50-MyDrive-test_acc_0.97397_epoch-90.pt"))
    prediction_material.setJsonPath(os.path.join(execution_path_material, "./Entrenamiento_Material/models/MyDrive_model_classes.json"))
    prediction_material.loadModel()
    predictions_material=[]
    probabilities_material=[]
    pvc=0
    hormigon=0
    pead=0
    size = listdir("./images_prueba_cv/")
    
    #Se analizan todas las fotos, se suma uno a cada material, el material mayor es el material seleccionado
    for i in range(1,len(size)):
        predictions_material , probabilities_material = prediction_material.classifyImage(os.path.join(execution_path_material, "./images_prueba_cv/frame"+str(i)+".jpg"), result_count=3)
        for eachPrediction, eachProbability in zip(predictions_material, probabilities_material):
            if eachPrediction=='Hormigon' and eachProbability >=50.0000:
                hormigon=hormigon+1
            elif eachPrediction=='PEAD' and eachProbability >=50.0000:
                pead=pead+1
            elif eachPrediction=='PVC' and eachProbability >=50.0000:
                pvc=pvc+1
        
    if hormigon>pvc and hormigon>pead:
        pvc=0
        pead=0
        material_definitivo='Hormigon'
    elif pead>hormigon and pead>pvc:
        hormigon=0
        pvc=0
        material_definitivo='PEAD'
    elif pvc>pead and pvc>hormigon:
        pead=0
        hormigon=0
        material_definitivo='PVC'
    
    for j in probabilidad_acometida:
        lista_acometida_probabilidad.append("Probabilidad: "+ str(j) + "%")
        
    for j in probabilidad_raices:
        lista_raices_probabilidad.append("Probabilidad: "+ str(j) + "%")
    
    for j in probabilidad_grieta:
        lista_grieta_probabilidad.append("Probabilidad: "+ str(j) + "%")

    #Guardado de fotos para enseñarlas
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    j=0
    lista_duplic_aco=[]
    for i in range(len(segundos_acometida)-1):
        if (segundos_acometida[j]+1)!=segundos_acometida[j+1]:
            lista_duplic_aco.append(segundos_acometida[i])
        j=j+1
    if len(segundos_acometida)>0:
        lista_duplic_aco.append(segundos_acometida[-1])
    
    #Aqui comienza el bucle de guardado de repeticion de fotos, si se detectan 4 imagenes seguidas, solo se coge la última
    aco_rep_1=[]
    for i in range(len(lista_duplic_aco)-1):
        if (lista_duplic_aco[i]+2)!=lista_duplic_aco[i+1]:
            aco_rep_1.append(lista_duplic_aco[i])
    if len(lista_duplic_aco)>0:
        aco_rep_1.append(lista_duplic_aco[-1])
    
    aco_rep_2=[]
    for i in range(len(aco_rep_1)-1):
        if (aco_rep_1[i]+3)!=aco_rep_1[i+1]:
            aco_rep_2.append(aco_rep_1[i])
    if len(aco_rep_1)>0:
        aco_rep_2.append(aco_rep_1[-1])
        
    aco_rep_3=[]
    for i in range(len(aco_rep_2)-1):
        if (aco_rep_2[i]+4)!=aco_rep_2[i+1]:
            aco_rep_3.append(aco_rep_2[i])
    if len(aco_rep_2)>0:
        aco_rep_3.append(aco_rep_2[-1])
        
    aco_rep_4=[]
    for i in range(len(aco_rep_3)-1):
        if (aco_rep_3[i]+5)!=aco_rep_3[i+1]:
            aco_rep_4.append(aco_rep_3[i])
    if len(aco_rep_3)>0:
        aco_rep_4.append(aco_rep_3[-1])
    
    sum_aco=0
    for i in aco_rep_4:
        lista_acometida_tiempo.append("Tiempo transcurrido: "+str(i)+ "s")
        lista_video_acometida.append("Resultados correspondientes al video: "+archivo[prueba])
        lista_material_acometida.append("El material de la tubería es: "+material_definitivo)
        img=cv2.imread("./images_prueba_cv/frame"+str(i)+".jpg")
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./static/resultados/acometidas/"+archivo[prueba]+"_"+str(sum_aco)+"_frame"+str(i)+".jpg",img)
        sum_aco=sum_aco+1
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    j=0
    lista_duplic_rai=[]
    for i in range(len(segundos_raices)-1):
        if (segundos_raices[j]+1)!=segundos_raices[j+1]:
            lista_duplic_rai.append(segundos_raices[i])
        j=j+1
    if len(segundos_raices)>0:
        lista_duplic_rai.append(segundos_raices[-1])
        
    #Aqui comienza el bucle de guardado de repeticion de fotos, si se detectan 4 imagenes seguidas, solo se coge la última    
    rai_rep_1=[]
    for i in range(len(lista_duplic_rai)-1):
        if (lista_duplic_rai[i]+2)!=lista_duplic_rai[i+1]:
            rai_rep_1.append(lista_duplic_rai[i])
    if len(lista_duplic_rai)>0:
        rai_rep_1.append(lista_duplic_rai[-1])
    
    rai_rep_2=[]
    for i in range(len(rai_rep_1)-1):
        if (rai_rep_1[i]+3)!=rai_rep_1[i+1]:
            rai_rep_2.append(rai_rep_1[i])
    if len(rai_rep_1)>0:
        rai_rep_2.append(rai_rep_1[-1])
        
    rai_rep_3=[]
    for i in range(len(rai_rep_2)-1):
        if (rai_rep_2[i]+4)!=rai_rep_2[i+1]:
            rai_rep_3.append(rai_rep_2[i])
    if len(rai_rep_2)>0:
        rai_rep_3.append(rai_rep_2[-1])
        
    rai_rep_4=[]
    for i in range(len(rai_rep_3)-1):
        if (rai_rep_3[i]+5)!=rai_rep_3[i+1]:
            rai_rep_4.append(rai_rep_3[i])
    if len(rai_rep_3)>0:
        rai_rep_4.append(rai_rep_3[-1])
    
    sum_rai=0
    for i in rai_rep_4:
        lista_raices_tiempo.append("Tiempo transcurrido: "+str(i)+ "s")
        lista_video_raices.append("Resultados correspondientes al video: "+archivo[prueba])
        lista_material_raices.append("El material de la tubería es: "+material_definitivo)
        img=cv2.imread("./images_prueba_cv/frame"+str(i)+".jpg")
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./static/resultados/raices/"+archivo[prueba]+"_"+str(sum_rai)+"_frame"+str(i)+".jpg",img)
        sum_rai=sum_rai+1
        
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    m=0
    lista_duplic_gr=[]
    for i in range(len(segundos_grieta)-1):
        if (segundos_grieta[m]+1)!=segundos_grieta[m+1]:
            lista_duplic_gr.append(segundos_grieta[i])
        m=m+1
    if len(segundos_grieta)>0:
        lista_duplic_gr.append(segundos_grieta[-1])
    
    #Aqui comienza el bucle de guardado de repeticion de fotos, si se detectan 4 imagenes seguidas, solo se coge la última  
    gr_rep_1=[]
    for i in range(len(lista_duplic_gr)-1):
        if (lista_duplic_gr[i]+2)!=lista_duplic_gr[i+1]:
            gr_rep_1.append(lista_duplic_gr[i])
    if len(lista_duplic_gr)>0:
        gr_rep_1.append(lista_duplic_gr[-1])
        
    gr_rep_2=[]
    for i in range(len(gr_rep_1)-1):
        if (gr_rep_1[i]+3)!=gr_rep_1[i+1]:
            gr_rep_2.append(gr_rep_1[i])
    if len(gr_rep_1)>0:
        gr_rep_2.append(gr_rep_1[-1])
        
    gr_rep_3=[]
    for i in range(len(gr_rep_2)-1):
        if (gr_rep_2[i]+4)!=gr_rep_2[i+1]:
            gr_rep_3.append(gr_rep_2[i])
    if len(gr_rep_2)>0:
        gr_rep_3.append(gr_rep_2[-1])
        
    gr_rep_4=[]
    for i in range(len(gr_rep_3)-1):
        if (gr_rep_3[i]+5)!=gr_rep_3[i+1]:
            gr_rep_4.append(gr_rep_3[i])
    if len(gr_rep_3)>0:
        gr_rep_4.append(gr_rep_3[-1])
    
    sum_gr=0
    for i in gr_rep_4:
        lista_grieta_tiempo.append("Tiempo transcurrido: "+str(i)+ "s")
        lista_video_grieta.append("Resultados correspondientes al video: "+archivo[prueba])
        lista_material_grieta.append("El material de la tubería es: "+material_definitivo)
        img=cv2.imread("./images_prueba_cv/frame"+str(i)+".jpg")
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./static/resultados/grieta/"+archivo[prueba]+"_frame"+str(i)+"_"+str(sum_gr)+".jpg",img)
        sum_gr=sum_gr+1
     
    #Guardado de imagenes probables de ser algún elemento
    for i in segundos_acometida_posible:
        img=cv2.imread("./images_prueba_cv/frame"+str(i)+".jpg")
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./static/resultados/acometida_probable/"+archivo[prueba]+"_frame"+str(i)+".jpg",img)
   
    for i in segundos_raices_posible:
        img=cv2.imread("./images_prueba_cv/frame"+str(i)+".jpg")
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./static/resultados/raices_probable/"+archivo[prueba]+"_frame"+str(i)+".jpg",img)
        
    for i in segundos_grieta_posible:
        img=cv2.imread("./images_prueba_cv/frame"+str(i)+".jpg")
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./static/resultados/grieta_probable/"+archivo[prueba]+"_frame"+str(i)+".jpg",img)
    
    #Lectura de distancia
    #Punto a mejorar, se recorta la imagen para coger la zona donde se muestra la distancia
    #Se limpian las variables recogidas y se muestran por pantalla
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    reader = easyocr.Reader(['es'])
    jpg_acometidas = os.listdir("./static/resultados/acometidas/")
    for i in jpg_acometidas:
        img = cv2.imread("./static/resultados/acometidas/"+str(i))
        cropped_image = img[475:550, 545:750]
        cv2.imwrite("./static/resultados/acometidas_cropped/Cropped Image_"+str(i), cropped_image)
    
    temp_aco_dist=[]
    string_aco=0
    pendiente_aco=[]
    jpg_acometidas_cropped = os.listdir("./static/resultados/acometidas_cropped/")
    for acometidas in jpg_acometidas_cropped:
        img = cv2.imread("./static/resultados/acometidas_cropped/"+str(acometidas))
        res_aco.append(reader.readtext(img, detail=0))
        
        if len(res_aco[0])==2:
        
            for x in res_aco[0]:
                
                j=x.replace(' ','')
                temp_aco_dist.append(j)
            
            string_aco='% ; '.join([str(item) for item in temp_aco_dist])
        
            #string = string.replace(" ", "")
            string_aco = string_aco.replace(".", ",")      
            string_aco = string_aco.replace("~", "-")
        
            result_acometida.append("Pendiente y Distancia recorrida: " +str(string_aco)+ "m")
            
            temp_aco_dist=[]
            
        elif len(res_aco[0])==3:
            
            for x in res_aco[0]:
                
                j=x.replace(' ','')
                temp_aco_dist.append(j)
            
            string_aco=','.join([str(item) for item in res_aco[0]])
        
            string_aco = string_aco.replace("-", "")        
            string_aco = string_aco.replace("~", "")    
            string_aco = string_aco.replace(".", ",")
            
            if len(string_aco)==9:
                for i in range(0,4):
                    pendiente_aco.append(string_aco[i])
                    
            elif len(string_aco)==10:
                for i in range(0,5):
                    pendiente_aco.append(string_aco[i])
                
            elif len(string_aco)==11:
                for i in range(0,5):
                    pendiente_aco.append(string_aco[i])
            
            string_aco2=''.join([str(item) for item in pendiente_aco])
            string_aco=string_aco[len(string_aco2):]

            result_acometida.append("Pendiente y Distancia recorrida: " +str(string_aco2)+ "% " +str(string_aco) + "m" )
            
            temp_aco_dist=[]
            pendiente_aco=[]
        
        res_aco = []
        
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    reader = easyocr.Reader(['es'])
    jpg_raices = os.listdir("./static/resultados/raices/")
    for i in jpg_raices:
        img = cv2.imread("./static/resultados/raices/"+str(i))
        cropped_image = img[475:550, 545:725]
        cv2.imwrite("./static/resultados/raices_cropped/Cropped Image_"+str(i), cropped_image)
    
    temp_raiz_dist=[]
    pendiente_raiz=[]
    string_raiz=0
    jpg_raices_cropped = os.listdir("./static/resultados/raices_cropped/")
    for raices in jpg_raices_cropped:
        img = cv2.imread("./static/resultados/raices_cropped/"+str(raices))
        res_raiz.append(reader.readtext(img, detail = 0))
        
        if len(res_raiz[0])==2:
        
            for x in res_raiz[0]:
                
                j=x.replace(' ','')
                temp_raiz_dist.append(j)
            
            string_raiz='% ; '.join([str(item) for item in temp_raiz_dist])
        
            #string = string.replace(" ", "")
            string_raiz = string_raiz.replace(".", ",")      
            string_raiz = string_raiz.replace("~", "-")
        
            result_raices.append("Pendiente y Distancia recorrida: " +str(string_raiz)+ "m")
            
            temp_raiz_dist=[]
            
        elif len(res_raiz[0])==3:
            
            for x in res_raiz[0]:
                
                j=x.replace(' ','')
                temp_raiz_dist.append(j)
            
            string_raiz=','.join([str(item) for item in res_raiz[0]])
        
            string_raiz = string_raiz.replace("-", "")        
            string_raiz = string_raiz.replace("~", "")    
            string_raiz = string_raiz.replace(".", ",")
            
            if len(string_raiz)==9:    
                for i in range(0,4):
                    pendiente_raiz.append(string_raiz[i])
                    
            elif len(string_raiz)==10:
                for i in range(0,5):
                    pendiente_raiz.append(string_raiz[i])
                
            elif len(string_raiz)==11:
                for i in range(0,5):
                    pendiente_raiz.append(string_raiz[i])
            
            string_raiz2=''.join([str(item) for item in pendiente_raiz])
            string_raiz=string_raiz[len(string_raiz2):]

            result_raices.append("Pendiente y Distancia recorrida: " +str(string_raiz2)+ "% " +str(string_raiz) + "m" )
            
            temp_raiz_dist=[]
            pendiente_raiz=[]
            
        res_raiz=[]
        
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    reader = easyocr.Reader(['es'])
    jpg_grieta = os.listdir("./static/resultados/grieta/")
    for i in jpg_grieta:
        img = cv2.imread("./static/resultados/grieta/"+str(i))
        cropped_image = img[475:550, 545:725]
        cv2.imwrite("./static/resultados/grieta_cropped/Cropped Image_"+str(i), cropped_image)
        
    temp_gr_dist=[]
    string_gr=0
    pendiente_gr=[]
    jpg_grieta_cropped = os.listdir("./static/resultados/grieta_cropped/")
    for grieta in jpg_grieta_cropped:
        img = cv2.imread("./static/resultados/grieta_cropped/"+str(grieta))
        res_gr.append(reader.readtext(img, detail = 0))
        
        if len(res_gr[0])==2:
        
            for x in res_gr[0]:
                
                j=x.replace(' ','')
                temp_gr_dist.append(j)
            
            string_gr='% ; '.join([str(item) for item in temp_gr_dist])
        
            #string = string.replace(" ", "")
            string_gr = string_gr.replace(".", ",")      
            string_gr = string_gr.replace("~", "-")
        
            result_grieta.append("Pendiente y Distancia recorrida: " +str(string_gr)+ "m")
            
            temp_gr_dist=[]
            
        elif len(res_gr[0])==3:
            
            for x in res_gr[0]:
                
                j=x.replace(' ','')
                temp_gr_dist.append(j)
            
            string_gr=','.join([str(item) for item in res_gr[0]])
        
            string_gr = string_gr.replace("-", "")        
            string_gr = string_gr.replace("~", "")    
            string_gr = string_gr.replace(".", ",")
            
            if len(string_gr)==9:
                for i in range(0,4):
                    pendiente_gr.append(string_gr[i])
                    
            elif len(string_gr)==10:
                for i in range(0,5):
                    pendiente_gr.append(string_gr[i])
                
            elif len(string_gr)==11:
                for i in range(0,5):
                    pendiente_gr.append(string_gr[i])
            
            string_gr2=''.join([str(item) for item in pendiente_gr])
            string_gr=string_gr[len(string_gr2):]

            result_grieta.append("Pendiente y Distancia recorrida: " +str(string_gr2)+ "% " +str(string_gr) + "m" )
            
            temp_gr_dist=[]
            pendiente_gr=[]
            
        res_gr=[]
        
    #Lectura distancia total, se coge la ultima imagen y se lee la distancia
    directory_path = "./images_prueba_cv/"
    most_recent_file = None
    most_recent_time = 0
    for entry in os.scandir(directory_path):
        if entry.is_file():
            mod_time = entry.stat().st_mtime_ns
            if mod_time > most_recent_time:
                most_recent_file = entry.name
                most_recent_time = mod_time
             
    dist_final=[]
    img_final=cv2.imread("./images_prueba_cv/"+str(most_recent_file))
    cropped_image = img_final[475:550, 545:750]
    cv2.imwrite("./static/resultados/ultimo_frame/final.jpg", cropped_image)
    img_dist_final_cropped=cv2.imread("./static/resultados/ultimo_frame/final.jpg")
    dist_final.append(reader.readtext(img_dist_final_cropped, detail = 0))
    
    total_aco=0
    total_rai=0
    total_gr=0
    sec_aco=[]
    sec_rai=[]
    sec_gr=[]
    
    for i in aco_rep_4:
        sec_aco.append(str(i))
    for i in rai_rep_4:
        sec_rai.append(str(i))
    for i in gr_rep_4:
        sec_gr.append(str(i))
        
    total_aco=len(sec_aco)
    total_rai=len(sec_rai)
    total_gr=len(sec_gr)
    

    informe_df = pd.read_csv(informe_csv_path)
    
    if dist_final[0]:
        distancia_total=dist_final[0][1]
        distancia_total = distancia_total.replace(" ", "")
        distancia_total = distancia_total.replace(",", ".") 
        distancia_total=float(distancia_total)
    else:
        distancia_total=0 #Esto es un apaño para analizar los videos de coruña
    distancia_total_list_aco=[]
    distancia_total_list_rai=[]
    distancia_total_list_gr=[]
    for i in range(len(gr_rep_4)):
        distancia_total_list_gr.append("La distancia total recorrida: "+str(distancia_total)+"m")
    for i in range(len(rai_rep_4)):
        distancia_total_list_rai.append("La distancia total recorrida: "+str(distancia_total)+"m")
    for i in range(len(aco_rep_4)):
        distancia_total_list_aco.append("La distancia total recorrida: "+str(distancia_total)+"m")
    
    limpiar_directorio("./images_prueba_cv/")
    
    nuevos_entrada = pd.DataFrame({'Inspeccion': archivo[prueba], 'Fecha': tiempo,'Diametro (mm)' : diametros[prueba] , 'Metros totales (m)' : distancia_total, 'Agua': total_gr, 'Nº Raices': total_rai, 'Nº Grietas': total_gr, 'Nº Acometida': total_aco},index=[0])

    informe_df = pd.concat([informe_df, nuevos_entrada], ignore_index=False)

    informe_df.to_csv(informe_csv_path, index=False)
    
    #Se genera un diccionario para pasar las variables al framework
    resultados = {"lista_1": lista_acometida_tiempo, "lista_2": lista_acometida_probabilidad,"lista_3": lista_raices_tiempo, "lista_4": lista_raices_probabilidad,"lista_5": lista_video_raices, "lista_6": lista_video_acometida,"lista_7": lista_grieta_probabilidad, "lista_8": lista_grieta_tiempo, "lista_9": lista_video_grieta,"lista_10":result_acometida,"lista_11":result_raices,"lista_12":result_grieta,"lista_13":lista_material_acometida, "lista_14": lista_material_raices, "lista_15":lista_material_grieta,"lista_16":distancia_total_list_aco,"lista_17":distancia_total_list_rai,"lista_18":distancia_total_list_gr}
    with open('resultados.json', 'w') as f:
        json.dump(resultados, f)
    
#Se mueven los videos ejecutados por el servicio a otra carpeta para analizarlos posteriormente
destination_folder = "./BD_video"
source_folder = "./video"

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for archivos in archivo:
    source_path = os.path.join(source_folder, archivos)
    destination_path = os.path.join(destination_folder, archivos)

    # Mover el archivo
    shutil.move(source_path, destination_path)    

limpiar_directorio("./video/")
        
#tabla_acometida_final = pd.concat(dataframes_acometida, ignore_index=True)
#tabla_raices_final = pd.concat(dataframes_raices, ignore_index=True)
#tabla_grieta_final = pd.concat(dataframes_grieta, ignore_index=True)
#tabla_conjunta_final = pd.concat([tabla_acometida_final,tabla_raices_final,tabla_grieta_final], ignore_index=True)

#Funcion para generar el reporte
def generate_pdf_report(file_path, logo_path, company_name, report_content):
     # Crear un lienzo para el PDF
    pdf = SimpleDocTemplate(file_path, pagesize=letter)

    # Estilos de párrafo
    styles = getSampleStyleSheet()
    header_style = ParagraphStyle('Header', fontSize=10)
    cuerpo=ParagraphStyle('Body',
                            alignment=1
                            )
    titulo=ParagraphStyle('Title',
                            alignment=0,
                            fontSize=16,
                            borderWidth=2
                            )
    # Contenido del informe
    content = []
    
    # Agregar el logo y el nombre de la empresa a la cabecera
    logo = Image(logo_path, width=2*inch, height=0.5*inch)  # Ajusta el tamaño según sea necesario
    company_name_paragraph = Paragraph(company_name, header_style)

    # Alinear el logo y el nombre de la empresa en la cabecera
    header_content = [[logo, Spacer(1, 1), company_name_paragraph]]

    # Construir la cabecera
    header_table = Table(header_content, colWidths=[1.5*inch, 4*inch, 1.5*inch])
    content.append(header_table)
    content.append(Spacer(0, 0.5*inch))  # Espacio entre la cabecera y el contenido del informe

    table = Table(table_data)
    content.append(table)
    content.append(Spacer(0, 1*inch)) #Espacio entre tabla e imágenes
    
    jpg_aco_pdf = os.listdir("./static/resultados/acometidas/")
    total_acometidas=len(jpg_aco_pdf)
    if total_acometidas>0:
        texto_acometidas="ACOMETIDAS"
        content.append(Paragraph(texto_acometidas,titulo))
        content.append(Spacer(0, 0.5*inch))
    i=0
    for aco in jpg_aco_pdf:
        if i<=total_acometidas:
            img = "./static/resultados/acometidas/"+str(aco)
            image = Image(img, width=3*inch, height=2.5*inch,hAlign='CENTER')
            
            # Crea el texto que deseas colocar a la derecha de la imagen
            tiempo_aco = Paragraph(lista_acometida_tiempo[i],cuerpo)
            prob_aco=Paragraph(lista_acometida_probabilidad[i],cuerpo)
            vid_aco=Paragraph(lista_video_acometida[i],cuerpo)
            res_aco=Paragraph(result_acometida[i],cuerpo)
            mat_aco=Paragraph(lista_material_acometida[i],cuerpo)


            # Agrupa la imagen y el texto en una lista
            fila = [image,  tiempo_aco,prob_aco,vid_aco,res_aco,mat_aco]

            # Agrega la fila a la tabla de contenido
            content.extend(fila)  # Utiliza extend() para agregar los elementos individuales de la fila a la lista content
            content.append(Spacer(0, 0.5*inch))
            i=i+1
    
    jpg_rai_pdf = os.listdir("./static/resultados/raices/")
    total_raices=len(jpg_rai_pdf)
    if total_raices>0:
        texto_raices="RAICES"
        content.append(Paragraph(texto_raices,titulo))
        content.append(Spacer(0, 0.5*inch))
    j=0
    for rai in jpg_rai_pdf:
        if j<=total_raices: 
            img = "./static/resultados/raices/"+str(rai)
            image = Image(img, width=3*inch, height=2.5*inch,hAlign='CENTER')
            
            # Crea el texto que deseas colocar a la derecha de la imagen
            tiempo_rai = Paragraph(lista_raices_tiempo[j],cuerpo)
            prob_rai=Paragraph(lista_raices_probabilidad[j],cuerpo)
            vid_rai=Paragraph(lista_video_raices[j],cuerpo)
            res_rai=Paragraph(result_raices[j],cuerpo)
            mat_rai=Paragraph(lista_material_raices[j],cuerpo)


            # Agrupa la imagen y el texto en una lista
            fila = [image,  tiempo_rai,prob_rai,vid_rai,res_rai,mat_rai]

            # Agrega la fila a la tabla de contenido
            content.extend(fila)  # Utiliza extend() para agregar los elementos individuales de la fila a la lista content
            content.append(Spacer(0, 0.5*inch))
            j=j+1
            
    jpg_gr_pdf = os.listdir("./static/resultados/grieta/")
    total_grietas=len(jpg_gr_pdf)
    if total_grietas>0:
        texto_grietas="GRIETAS"
        content.append(Paragraph(texto_grietas,titulo))
        content.append(Spacer(0, 0.5*inch))
    k=0
    for gr in jpg_gr_pdf:
        if k<=total_grietas:
            img = "./static/resultados/grieta/"+str(gr)
            image = Image(img, width=3*inch, height=2.5*inch,hAlign='CENTER')
                                                         
            # Crea el texto que deseas colocar a la derecha de la imagen
            tiempo_gr = Paragraph(lista_grieta_tiempo[k],cuerpo)
            prob_gr=Paragraph(lista_grieta_probabilidad[k],cuerpo)
            vid_gr=Paragraph(lista_video_grieta[k],cuerpo)
            res_gr=Paragraph(result_grieta[k],cuerpo)
            mat_gr=Paragraph(lista_material_grieta[k],cuerpo)


            # Agrupa la imagen y el texto en una lista
            fila = [image,  tiempo_gr,prob_gr,vid_gr,res_gr,mat_gr]

            # Agrega la fila a la tabla de contenido
            content.extend(fila)  # Utiliza extend() para agregar los elementos individuales de la fila a la lista content
            content.append(Spacer(0, 0.5*inch))
            k=k+1
            
    # Construir el PDF
    pdf.build(content)
    
#index=[]
#index.append(["Numero de pozo","Fecha","Metros totales(m)","Agua","Raices","Grietas","Acometidas","Observaciones"])
ruta_csv = "./uploads/informe/FORMATO_INCIDENCIAS.csv"
data = pd.read_csv(ruta_csv)
column_names = data.columns.tolist()
table_data = []
table_data = [column_names] 
for i, row in data.iterrows():
    table_data.append(list(row))
table = Table(table_data)

jpg_grieta_pdf = os.listdir("./static/resultados/acometidas/")

    # Ejemplo de uso
if __name__ == "__main__":
    file_path = "informe.pdf"
    logo_path = "./static/sacyr_agua_esp.png"  # Ruta al archivo de imagen del logo
    company_name = "Informe "+tiempo
    report_content = [table]

generate_pdf_report(file_path, logo_path, company_name, report_content)

#tabla_acometida_final.to_csv("./uploads/final/resultado_acometidas_"+str(tiempo)+".csv", index=False)
#tabla_raices_final.to_csv("./uploads/final/resultado_raices_"+str(tiempo)+".csv", index=False)
#tabla_grieta_final.to_csv("./uploads/final/resultado_grieta_"+str(tiempo)+".csv", index=False)
#tabla_conjunta_final.to_csv("./uploads/final/resultado_conjunto_"+str(tiempo)+".csv", index=False)