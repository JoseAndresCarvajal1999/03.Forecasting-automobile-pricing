import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from scipy.stats import pearsonr, spearmanr 


 
file = pd.read_csv('Datos_No_Nan.csv')
 
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

#Coeficiente de Correlación variables numericas 

for i in range(len(numericas)):
    print(numericas[i])
    r,p  = pearsonr(Var_numericas_stand[:,i],Price[:,0])
    print(f"Correlación Pearson: r={r}, p-value={p}")
    r,p = spearmanr(Var_numericas_stand[:,i],Price[:,0])
    print(f"Correlación Spearman: r={r}, p-value={p}")
    
    
#Coeficiente de Correlacion Variables Categoricas 
    
for i in range(len(categoricas)):
    print(categoricas[i])
    r,p  = pearsonr(Var_categoricas_stand[:,i],Price[:,0])
    print(f"Correlación Pearson: r={r}, p-value={p}")
    r,p = spearmanr(Var_categoricas_stand[:,i],Price[:,0])
    print(f"Correlación Spearman: r={r}, p-value={p}")
    
    
#Variables a usar 
    
variables = ['kms', 'model', 'power', 'shift' , 'year', 'version', 'price']
datos_pred = file[variables]
datos_pred.to_csv('Datos_pred.csv')
