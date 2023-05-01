import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.kernel_approximation import RBFSampler

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
    
##Mapeo Basico multiplicando variables 
#nombres_aux = nombres
#datos_altasDim =   carros_pand  
#for i in nombres_aux:
#    print(i)
#    for j in reversed(nombres_aux):
#        multplicacion = carros_pand.loc[:,i]*carros_pand.loc[:,j]
#        datos_altasDim = pd.concat([datos_altasDim,multplicacion], axis = 1)
#    nombres_aux.remove(i)
#    
##datos_altasDim.to_csv('AltasDimensiones.csv')
#
##Mapeo usando Kernels de transformación     
#rbf_feature = RBFSampler(gamma=1, random_state=1,n_components = 12)
#carros_kern = rbf_feature.fit_transform(carros_pand)
#
#pd.DataFrame(carros_kern).to_csv('DatosKernel.csv')
#
#
        
