import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as ss

file  = pd.read_csv('Datos_pred.csv')
categoricas = []
numericas = []
for col in file.columns:
    tipo = type(file[col][1])
    if (tipo == str): 
        categoricas.append(col)
    else: 
        numericas.append(col)
numericas.remove('Unnamed: 0')
numericas.remove('price')

Var_numericas = file[numericas]
Var_categoricas = file[categoricas]    
scaler = MinMaxScaler()
Price = np.array(file['price']).reshape(-1,1)
Price = scaler.fit_transform(Price)
Var_numericas = file[numericas]
scaler = MinMaxScaler()
Var_numericas_stand = scaler.fit_transform(Var_numericas)
Var_categoricas = file[categoricas]    
enc = OrdinalEncoder()
enc.fit(Var_categoricas)
Var_categoricas_enc = enc.transform(Var_categoricas)
Var_categoricas_stand = scaler.fit_transform(Var_categoricas_enc)

carros  = np.concatenate((Var_numericas_stand,Var_categoricas_stand),axis = 1)

#Se crea un nuevo DataFrame para analisisar los datos 
nombres = ['kms', 'model', 'power', 'shift' , 'year', 'version']
carros_pand = pd.DataFrame(carros)
carros_pand = carros_pand.rename(columns = {0: 'kms', 1: 'model', 2:'power', 3:'shift', 4:'year', 
                            5: 'version'})
    
#Se hace un analisis de correlación entre las variables de los datos 
corr = carros_pand.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True)
plt.show()

# ------------ Analisis Estadistico de las variables de entrada
    
#Se hace histogramas sobre los datos var ver a ojo las distribuciones 

plt.figure()
carros_pand.diff().hist(color="r", alpha=0.5, bins=50,figsize=(10, 8), grid = False)

#Boxplot para las varialbes de entrada 

color = {"boxes": "Red",
"whiskers": "Orange",
"medians": "Blue",
"caps": "Gray"}
carros_pand.plot.box(color = color)

#Estadistica descriptiva 
estadistica = carros_pand.describe()

#Pruebas de Bondad de ajuste confianza del 99%
distribucion = [ss.norm, ss.maxwell, ss.pareto, ss.t, ss.uniform, ss.beta,
               ss.logistic, ss.laplace]
distribucion_names =['norm', 'maxwell', 'pareto', 't', 'uniform', 'beta',
               'logistic', 'laplace']

for dist in range(len(distribucion)):
    print('----------------------- ' + distribucion_names[dist])
    for nombre in nombres:
        param = distribucion[dist].fit(carros_pand[nombre])
        print(param)
        d, pvalor = ss.kstest(carros_pand[nombre],
                              distribucion_names[dist], param)
        print(nombre)      

#Calculo de Entropias 
def entropia(datos):
    H = [x*np.log(x) for x in datos]
    return -sum(H)


#Entropia para la variable kilometros 
data = ss.laplace.pdf(carros_pand['kms'])
entrop = entropia(data)
print('Entropia kilometros bajo la medida de laplace ', entrop)
data = ss.logistic.pdf(carros_pand['kms'])
entrop = entropia(data)
print('Entropia kilometros bajo la medida de la distribución logistica ', entrop )

#Entropia para la variable Modelo 
data = ss.laplace.pdf(carros_pand['model'])
entrop = entropia(data)
print('Entropia modelo bajo la medida de laplace ', entrop)
data = ss.logistic.pdf(carros_pand['model'])
entrop = entropia(data)
print('Entropia modelo bajo la medida de la distribución logistica ', entrop)

#Entropia para la variable Potencia 
data = ss.laplace.pdf(carros_pand['power'])
entrop = entropia(data)
print('Entropia potencia bajo la medida de laplace ', entrop)
data = ss.logistic.pdf(carros_pand['power'])
entrop = entropia(data)
print('Entropia potencia bajo la medida de la distribución logistica ', entrop)

#Entropia de la variables cambios 
data = ss.norm.pdf(carros_pand['shift'])
entrop = entropia(data)
print('Entropia cambios bajo la medida de  la distribución normal ', entrop)
data = ss.laplace.pdf(carros_pand['shift'])
entrop = entropia(data)
print('Entropia cambios bajo la medida de la distribución laplace ', entrop)

#Entropia Version 
data = ss.norm.pdf(carros_pand['version'])
entrop = entropia(data)
print('Entropia cambios bajo la medida de  la distribución normal ', entrop)

#Criterio de Laplace 
val1, val0 = carros_pand['year'].value_counts()
n = (val1+val0)
p1 =  val1/n 
p0 = val0/n
entrop = -(p1*np.log(p1) + p0*np.log(p0))
print('Entropia del año bajo el criteriio de Laplace', entrop)



# ------------ Analisis Estadistico de las variables de salida 

precio = file['price']
#Histograma 
plt.figure()
precio.diff().hist(color="r", alpha=0.5, bins=50,figsize=(10, 8), grid = False)
#Boxplot 
color = {"boxes": "Red",
"whiskers": "Orange",
"medians": "Blue",
"caps": "Gray"}
precio.plot.box(color = color)
#Etadisitca descriptiva 

estadistica = precio.describe()

#Pruebas de Bondad de ajuste confianza del 99%
distribucion = [ss.norm, ss.maxwell, ss.pareto, ss.t, ss.uniform, ss.beta,
               ss.logistic, ss.laplace]
distribucion_names =['norm', 'maxwell', 'pareto', 't', 'uniform', 'beta',
               'logistic', 'laplace']

for dist in range(len(distribucion)):
    print('----------------------- ' + distribucion_names[dist])
    param = distribucion[dist].fit(precio)
    print(param)
    d, pvalor = ss.kstest(precio,
                          distribucion_names[dist], param)
    print(nombre)
    print(pvalor)
    if pvalor < 0.01:
        print("No se ajusta a la distribucion")
    else:
        print("Se puede ajustar a la distribucion") 
    print('***********************+')

#Prueba Kruskal Wallies 
    
precio = pd.DataFrame(Price)
precio = precio.rename(columns = {0: 'price'})
for nombre in nombres:
    y = ss.kruskal(precio['price'], carros_pand[nombre])
    print(y)
    

    