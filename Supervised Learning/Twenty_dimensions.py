import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_squared_error, median_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import tree
import matplotlib.pyplot as plt 



data = np.array(pd.read_csv('AltasDimensiones.csv'))
Price = np.array(pd.read_csv('Price.csv'))
Price = Price[:,1]
Price = Price.reshape(-1,1)
n = len(data)
Price = Price[:n,:]

scaler = MinMaxScaler()
#Price = scaler.fit_transform(Price)
data = scaler.fit_transform(data)

#------------------- PCA ------------------------------------


x_train,x_rest, y_train, y_rest = train_test_split(data,Price, test_size=0.4)
x_val, x_test, y_val, y_test = train_test_split(x_rest,y_rest, test_size = 0.5)

#------------ Regression lineal -------------------------
#--------Entrenar ------------------------
regresor = LinearRegression()
regresor.fit(x_train, y_train)
#-------- Validar -------------------------
y_pred1 = regresor.predict(x_val)
#print('Error maximo : ',max_error(y_pred1,y_val))
#print('Procenta medio de error : ',mean_absolute_percentage_error(y_pred1,y_val))
#print('MSE: ',mean_squared_error(y_pred1,y_val))

#-------- Testear ------------------------------------------
print('---------------  Testeo Regresion Lineal ----------------')
y_pred1 = regresor.predict(x_test)
print('Error maximo : ',max_error(y_pred1,y_test))
print('Procenta medio de error : ',median_absolute_error(y_pred1,y_test))
print('MSE: ',mean_squared_error(y_pred1,y_test,squared = False))

plt.plot(y_pred1, 'or')
plt.show()
plt.plot(y_test, 'ob')
plt.show()

#------------ Redes neuronales -------------------------
regresor_n = MLPRegressor(hidden_layer_sizes = (500,3), activation = 'tanh')
regresor_n.fit(x_train, y_train)
#-------- Validar -------------------------
y_pred1n = regresor_n.predict(x_val)
#print('Error maximo : ',max_error(y_pred1n,y_val))
#print('Procenta medio de error : ',mean_absolute_percentage_error(y_pred1n,y_val))
#print('MSE: ',mean_squared_error(y_pred1n,y_val))

#-------- Testear ------------------------------------------
print('---------------  Testeo NN  ----------------')
y_pred1n = regresor_n.predict(x_test)
print('Error maximo : ',max_error(y_pred1n,y_test))
print('Procenta medio de error : ',median_absolute_error(y_pred1n,y_test))
print('MSE: ',mean_squared_error(y_pred1n,y_test, squared = False))

plt.plot(y_pred1, 'or')
plt.show()
plt.plot(y_test, 'ob')
plt.show()

##------------ Decesion tree  -------------------------
regressor = tree.DecisionTreeRegressor(max_depth = 20)
regressor.fit(x_train, y_train)
#-------- Validar -------------------------
#y_pred1 = regressor.predict(x_val)
#print('Error maximo : ',max_error(y_pred1,y_val))
#print('Procenta medio de error : ',mean_absolute_percentage_error(y_pred1,y_val))
#print('MSE: ',mean_squared_error(y_pred1,y_val))

#-------- Testear ------------------------------------------
print('---------------  Testeo   DT ----------------')
y_pred1 = regressor.predict(x_test)
print('Error maximo : ',max_error(y_pred1,y_test))
print('Procenta medio de error : ',median_absolute_error(y_pred1,y_test))
print('MSE: ',mean_squared_error(y_pred1,y_test, squared = False))

plt.plot(y_pred1, 'or')
plt.show()
plt.plot(y_test, 'ob')
plt.show()
