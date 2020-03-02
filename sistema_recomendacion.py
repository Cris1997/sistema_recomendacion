"""
Author: Cristian Rosales Deloya

"""

#Pandas, librería para el manejo de los dataframe dentro del programa
import pandas as pd
#Librerias de funciones matemáticas
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


# Ponderaciones para cada atributo de las 

w_variedad = 0.20
w_pais = 0.05
w_region = 0.15
w_tipo = 0.10
w_sabor = 0.10
w_aroma = 0.25
w_precio = 0.15



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum() 


def distancia_euclidiana(x,y):
    return 1/(1+np.sqrt(np.sum((x - y) ** 2)))


# Leer el Dataframe de vinos
def abrir_datafrane():
    vinos_df = pd.read_csv('data_vinos.csv')
    # print(vinos_df['aroma'])
    return vinos_df


# Eliminar las columnas que no se requieren para hacer una recomendación 

def drop_columns():
    print("Eliminando columnas no usadas")


def clean_data(vinos_df):
    print("Limpiando Datos")
    # Eliminar espacios en blanco para no tener redundancias en las categoria
    vinos_df['variedad'] = vinos_df.variedad.str.replace(" ","")
    vinos_df['pais'] = vinos_df.pais.str.replace(" ","")
    vinos_df['region'] = vinos_df.region.str.replace(" ","")
    vinos_df['color'] = vinos_df.color.str.replace(" ","")
    vinos_df['sabor'] = vinos_df.sabor.str.replace(" ","")
    vinos_df['aroma'] = vinos_df.aroma.str.replace(" ","")
    vinos_df['precio'] = vinos_df.precio.str.replace(" ","")
    return vinos_df

def estructurar_datos(vinos_df):
    print("Estructurando datos a listas")
    vinos_df['variedad'] = vinos_df.variedad.str.split(',')
    vinos_df['aroma'] = vinos_df.aroma.str.split(',')
    vinos_df['pais'] = vinos_df.pais.str.split(',')
    vinos_df['region'] = vinos_df.region.str.split(',')
    vinos_df['color'] = vinos_df.color.str.split(',')
    vinos_df['sabor'] = vinos_df.sabor.str.split(',')
    vinos_df['precio'] = vinos_df.precio.str.split(',')
    # print(vinos_df['aroma'])
    return vinos_df
    
    

# Descriptor Variedad
def generar_descriptor(vinos_df,aspecto):
    for index, row in vinos_df.iterrows():
        for aspect in row[aspecto]:
            vinos_df.at[index, aspect] = 1
    vinos_df= vinos_df.fillna(0)
    vinos_df = vinos_df.drop(['nombre', 'variedad','porcen_alch','pais','region','guarda','temp_consumo','color','aroma','maridaje','sabor','precio'], axis=1)
    
    descriptor = vinos_df.drop(['id'],axis=1)
    # descriptor.to_csv('out.csv')
    # rept = []
    # for col in descriptor.columns: 
    #     rept.append(col)
    # print((rept))
    # Aplicar Softmax a cada fila del descriptor de vinnos
    
    arreglos_numpy = descriptor.to_numpy()
    for array in arreglos_numpy:
        temporal = []
        for item in array:
            if item == 1:
               temporal.append(item)     
        y = softmax(temporal)
        for i in range(len(array)):
            if array[i] == 1:
               array[i] = y[0]
        # print(array)
    
    
    descriptor[:] = arreglos_numpy
    descriptor.to_csv(aspecto + ".csv")
    # return arreglos_numpy


    


# Funcion Main
if __name__ == "__main__":
    # Abrir dataframe
    vinos_df = abrir_datafrane()
    # Para el precio es necesario convertir todos los datos a String primero 
    vinos_df['precio'] = vinos_df['precio'].astype(str)
    # print(vinos_df.dtypes)
    
    # Eliminar espacios en blanco para evitar la redundancia
    vinos_df = clean_data(vinos_df)
    # Convertir a lista los campos que son multivalor (aroma y variedad)
    vinos_df = estructurar_datos(vinos_df)


    aspectos = ['variedad','pais','region','color','sabor','aroma','precio']
    
    for aspecto in aspectos:
        df = vinos_df.copy()
        generar_descriptor(df ,aspecto)
        
    print("Recomendación en proceso")
    
    desc_variedad = pd.read_csv('variedad.csv', index_col=0).to_numpy()
    desc_pais = pd.read_csv('pais.csv', index_col=0).to_numpy()
    desc_region = pd.read_csv('region.csv',  index_col=0).to_numpy()
    desc_tipo = pd.read_csv('color.csv', index_col=0 ).to_numpy()
    desc_sabor = pd.read_csv('sabor.csv',  index_col=0).to_numpy()
    desc_aroma = pd.read_csv('aroma.csv',  index_col=0).to_numpy()
    desc_precio = pd.read_csv('precio.csv',  index_col=0).to_numpy()
    
    identificador = 17
    
    dist_variedad = []
    dist_pais = []
    dist_region = []
    dist_tipo = []
    dist_sabor = []
    dist_aroma = []
    dist_precio = []

    
    for i in range(len(desc_variedad)):
        # Distancia variedad
        dist_variedad.append(distancia_euclidiana(desc_variedad[identificador],desc_variedad[i])* w_variedad)
         # Distancia pais
        dist_pais.append(distancia_euclidiana(desc_pais[identificador],desc_pais[i]) * w_pais)
         # Distancia region
        dist_region.append(distancia_euclidiana(desc_region[identificador],desc_region[i]) *  w_region)
         # Distancia tipo
        dist_tipo.append(distancia_euclidiana(desc_tipo[identificador],desc_tipo[i]) * w_tipo )
         # Distancia sabor
        dist_sabor.append(distancia_euclidiana(desc_sabor[identificador],desc_sabor[i]) * w_sabor )
         # Distancia aroma
        dist_aroma.append(distancia_euclidiana(desc_aroma[identificador],desc_aroma[i]) * w_aroma)
         # Distancia precio
        dist_precio.append(distancia_euclidiana(desc_precio[identificador],desc_precio[i]) * w_precio )
    
   # Sumatoria de los arreglos
    uno = np.asarray(dist_variedad, dtype=np.float32)
    dos = np.asarray(dist_pais, dtype=np.float32)
    tres = np.asarray(dist_region, dtype=np.float32)
    cuatro = np.asarray(dist_tipo, dtype=np.float32)
    cinco = np.asarray(dist_sabor, dtype=np.float32)
    seis = np.asarray(dist_aroma, dtype=np.float32)
    siete = np.asarray(dist_precio, dtype=np.float32)
    
    scores = uno + dos + tres + cuatro + cinco + seis + siete
    print(scores)
    
    diccionario = { i : scores[i] for i in range(0, len(scores) ) }
    for item in diccionario:
        print(diccionario[item])
    # for i in range(len(arreglos_numpy)):
    #     eee = distancia_euclidiana(arreglos_numpy[0],arreglos_numpy[i])
    #     print(eee*w_variedad)
    


