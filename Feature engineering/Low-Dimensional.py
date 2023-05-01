import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits import mplot3d
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


file  = pd.read_csv('Datos_pred.csv')
 
#Clasificación variables categrocicas y numericas 
categoricas = []
numericas = []
for col in file.columns:
    tipo = type(file[col][1])
    if (tipo == str): 
        categoricas.append(col)
    else: 
        numericas.append(col)
        
numericas.remove('price')
numericas.remove('Unnamed: 0')

#Preprocesamiento de Datos 
scaler = MinMaxScaler()
Price = np.array(file['price']).reshape(-1,1)
Price = scaler.fit_transform(Price)


##Frame Variables Numericas
        
Var_numericas = file[numericas]
scaler = MinMaxScaler()
Var_numericas_stand = scaler.fit_transform(Var_numericas)


#Frame Variables Categoricas 

Var_categoricas = file[categoricas]    
enc = OrdinalEncoder()
enc.fit(Var_categoricas)
Var_categoricas_enc = enc.transform(Var_categoricas)
Var_categoricas_stand = scaler.fit_transform(Var_categoricas_enc)


#Variables en total 
carros  = np.concatenate((Var_numericas_stand,Var_categoricas_stand),axis = 1)
nombres = ['kms', 'model', 'power', 'shift' , 'year', 'version']
carros_pand = pd.DataFrame(carros)
carros_pand = carros_pand.rename(columns = {0: 'kms', 1: 'model', 2:'power', 3:'shift', 4:'year', 
                            5: 'version'})
#    
#----------------------- Descoposición en componentes principales  

pca = PCA()
pca.fit(carros_pand)

datos_pca = pd.DataFrame(data    = pca.components_, columns = carros_pand.columns, 
                         index   = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])

#Mapa de Calor 
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
componentes = pca.components_
plt.imshow(componentes.T, cmap='magma', aspect='auto')
plt.yticks(range(len(carros_pand.columns)), carros_pand.columns)
plt.xticks(range(len(carros_pand.columns)), np.arange(pca.n_components_) + 1)
plt.grid(False)
plt.colorbar();

#Varianza explicada por cada componente 

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.bar(x = np.arange(pca.n_components_) + 1, height = pca.explained_variance_ratio_)

for x, y in zip(np.arange(len(carros_pand.columns)) + 1, pca.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center')

ax.set_xticks(np.arange(pca.n_components_) + 1)
ax.set_ylim(0, 1.1)
ax.set_title('Porcentaje de varianza explicada por cada componente')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza explicada');

#Proyecciones de los datos 

proyecciones = pca.transform(carros_pand)

#Proyeccion en dos dimensiones 
fig = plt.figure(figsize = (10, 7))
plt.scatter(proyecciones[:,1], proyecciones[:,2])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


#Proyección en 3 dimensiones 
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
p = ax.scatter3D(proyecciones[:,1], proyecciones[:,2], proyecciones[:,3])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()
#
proyecciones_frame = pd.DataFrame(proyecciones, columns = carros_pand.columns )

proyecciones_frame.to_csv('PCA.csv')

#-------------------------------- Descomposición en valores singulares 
#Dos dimensiones 
svd = TruncatedSVD(n_components=5)
svd.fit(carros_pand)
descomposicion = svd.transform(carros_pand)

fig = plt.figure(figsize = (10, 7))
plt.scatter(descomposicion[:,1], descomposicion[:,2])
plt.xlabel('SVD1')
plt.ylabel('SVD2')
plt.show()

## Tres dimensiones 
#fig = plt.figure(figsize = (10, 7))
#ax = plt.axes(projection ="3d")
#p = ax.scatter3D(descomposicion[:,1], descomposicion[:,2], descomposicion[:,3])
#plt.show()
#
#descomposicion_frame = pd.DataFrame(descomposicion)







