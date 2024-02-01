#!/usr/bin/env python
# coding: utf-8

# In[102]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
import numpy as np
from glob import glob
import plotly.graph_objects as go
import re
import random
import plotly.express as px

# Paquetes

from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from sktime.performance_metrics.forecasting import mean_absolute_percentage_error


from sklearn import metrics
import matplotlib.pyplot as plta

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from statsmodels.tsa.arima.model import ARIMA


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from joblib import dump, load

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.neighbors import KNeighborsRegressor

from sklearn.feature_selection import RFE, SequentialFeatureSelector, SelectKBest, f_regression, VarianceThreshold, RFECV

from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

import joblib


# # Lectura y funciones

# Recuerda que deberías tomar al menos las últimas 48 horas que tienes de información sobre los pluvios y los caudales

# In[103]:


# Esto es para obtener el path donde encontramos los datos de las ramblas
path = "../raw/SAIHdatasActualizados/Ramblas/*/*"  


# In[104]:


# Nombre de todos los archivos
files = glob(path, recursive=True)
files


# Función para cargar archivos de las diferentes variables en los subdirectorios asociando el nombre del subdirectorio en el diccionario

# In[105]:


def cargar_parquets(ruta_directorio,variable):
    archivos = {}
    for root, dirs, files in os.walk(ruta_directorio):
        for file in files:
            if re.search(fr'.*{variable}.parquet', file, re.IGNORECASE):  # Utilizando expresión regular
                filepath = os.path.join(root, file)
                df = pd.read_parquet(filepath)  # cargar el archivo csv
                # Obtener la última carpeta en la ruta como la clave
                _, ultima_carpeta = os.path.split(root)
                archivos[ultima_carpeta] = df  # guardar el dataframe en el diccionario

    return archivos


# La función de resampleado para obtener el agregado, la media, el max, el minimo y la desviación std de todos los puntos. (sum, mean, max, min, std)

# In[106]:


# Función agregado, media, max y min configurable
def resample_df(df, fecha, codigo_temporal):
    # Asegurémonos de que la fecha sea un datetime
    df[f'{fecha}'] = pd.to_datetime(df[f'{fecha}'])

    # Establecer fecha con hora como indice
    df = df.set_index(f'{fecha}')


    # Resampling a agregados, promedio, máximo y mínimo por código temporal
    df_sum = df.resample(f'{codigo_temporal}').sum().add_prefix(f'sum_{codigo_temporal}')
    df_mean = df.resample(f'{codigo_temporal}').mean().add_prefix(f'mean_{codigo_temporal}')
    df_max = df.resample(f'{codigo_temporal}').max().add_prefix(f'max_{codigo_temporal}')
    df_min = df.resample(f'{codigo_temporal}').min().add_prefix(f'min_{codigo_temporal}')
    df_std = df.resample(f'{codigo_temporal}').std().add_prefix(f'std{codigo_temporal}')


    df_resampled = pd.concat([df_sum, df_mean, df_max, df_min, df_std], axis=1)

    return df_resampled


# Función para obtener un número determinado de shift de las columnas de un dataframe 

# In[107]:


# Función para shiftear
def shift_df(df, num_shifts, name_variable):
    dfs = []
    for i in range(1, num_shifts+1):
        df_shifted = df.shift(i)
        df_shifted = df_shifted.add_prefix(f'shift_{i}_{name_variable}')
        dfs.append(df_shifted)
    df_shifted_all = pd.concat(dfs, axis=1)
    return df_shifted_all


# Función para hacer un rollado de un dataframe y calcular la media, el agregado, el max, el min y la desviación std

# In[108]:


# Función de rolling
def rolling_df(df, fecha, codigo_temporal, variable):
    # Asegurémonos de que la fecha sea un datetime
    df[f'{fecha}'] = pd.to_datetime(df[f'{fecha}'])

    # Establecer fecha con hora como indice
    df = df.set_index(f'{fecha}')

    df_rolling_mean = df.rolling(window=codigo_temporal).mean().add_prefix(f'mean_rolling_{codigo_temporal}_{variable}')
    df_rolling_sum = df.rolling(window=codigo_temporal).sum().add_prefix(f'sum_rolling_{codigo_temporal}_{variable}')
    df_rolling_max = df.rolling(window=codigo_temporal).max().add_prefix(f'max_rolling_{codigo_temporal}_{variable}')
    df_rolling_min = df.rolling(window=codigo_temporal).min().add_prefix(f'min_rolling_{codigo_temporal}_{variable}')
    df_rolling_std = df.rolling(window=codigo_temporal).std().add_prefix(f'std_rolling_{codigo_temporal}_{variable}')

    df_rolling = pd.concat([df_rolling_mean, df_rolling_sum, df_rolling_max, df_rolling_min, df_rolling_std], axis=1)

    return df_rolling


# # Caudales

# In[109]:


caudales_prueba = cargar_parquets('../raw/SAIHdatasActualizados/Ramblas/', "Caudal")

# Eliminar los espacios de los nombres
caudales = {clave.replace(' ', ''): valor for clave, valor in caudales_prueba.items()}


# Ponemos el formato de la fecha de cada DataFrame como *datetime*, remplazamos NAs y convertimos a float

# In[110]:


for key, dataframe in caudales.items():
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

for key, dataframe in caudales.items():
    caudales[key] = dataframe.replace('-', np.nan)

for key, dataframe in caudales.items():
    caudales[key] = dataframe.replace('', np.nan)    

for key, dataframe in caudales.items():
    caudales[key] = dataframe.replace('null', np.nan)    

for key, dataframe in caudales.items():
    dataframe[dataframe.columns[1]] = dataframe[dataframe.columns[1]].astype(float)

DateIndex = pd.DataFrame(pd.date_range(start=pd.to_datetime('2021-03-08 12:00:00'), # 2021-03-08 11:35:00  2021-01-12 12:00:00
                                       end = ('2023-11-18 23:55:00'), 
                                       freq = '5T'),
                        columns=['Date'])

resultado = DateIndex

for name, df in caudales.items():
        resultado = resultado.merge(df, on='Date', how='left')
        resultado = resultado.rename(columns={'Caudal':name})
        
caudales_ok = resultado.drop_duplicates(subset='Date')

#caudales_ok.columns = caudales_ok.columns.astype(float)




# Informe de columnas y de NAs de los caudales en la serie completa

# In[111]:


# Identificar NAs
nas = caudales_ok.isnull().sum()
print("Cantidad de NAs por columna:")
print(nas)


# In[112]:


# Cuenta la cantidad de valores diferentes de cero en cada columna
cantidad_no_cero = (caudales_ok != 0).sum()
print("Cantidad de datos diferentes de cero en cada columna:")
print(cantidad_no_cero)


# In[113]:


# Describir los datos
data_description = caudales_ok.describe()
print("\nDescripción de los datos:")
print(data_description)


# In[114]:


caudales_final = caudales_ok.drop('06A19-LaMarana', axis=1)
# La marana no 


# In[115]:


caudales_final_interpolado = caudales_final.interpolate(method="bfill")


# In[116]:


# Ahora hacemos el resampleado de los caudales

resampleado_caudal = resample_df(caudales_final_interpolado, "Date","H")


# In[117]:


# Describir los datos
data_description = resampleado_caudal.describe()
print("\nDescripción de los datos:")
print(data_description)


# # Piezometros

# In[118]:


# Piezometros
Piezometros_prueba = cargar_parquets('../raw/SAIHdatasActualizados/piezometros', "Piezometrico")

# Eliminar los espacios de los nombres
Piezometrico = {clave.replace(' ', ''): valor for clave, valor in Piezometros_prueba.items()}


# In[119]:


for key, dataframe in Piezometrico.items():
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

for key, dataframe in Piezometrico.items():
    Piezometrico[key] = dataframe.replace('-', np.nan)

for key, dataframe in Piezometrico.items():
    dataframe[dataframe.columns[1]] = dataframe[dataframe.columns[1]].astype(float)

DateIndex = pd.DataFrame(pd.date_range(start=pd.to_datetime('2021-03-08 12:00:00'), # 2021-03-08 11:35:00  2021-01-12 12:00:00
                                       end = ('2023-11-18 23:55:00'), 
                                       freq = '5T'),
                        columns=['Date'])

resultado = DateIndex

for name, df in Piezometrico.items():
        resultado = resultado.merge(df, on='Date', how='left')
        resultado = resultado.rename(columns={'Piezometrico':name})
        
Piezometrico_ok = resultado.drop_duplicates(subset='Date')


# In[120]:


# Identificar NAs
nas = Piezometrico_ok.isnull().sum()
print("Cantidad de NAs por columna:")
print(nas)


# In[121]:


piezos_antes = Piezometrico_ok.drop('06Z06-Sondeo06', axis=1)
piezos_final = piezos_antes.drop('06Z16-Sondeo16', axis=1)


# In[122]:


# Describir los datos
data_description = piezos_final.describe()
print("\nDescripción de los datos:")
print(data_description)


# In[123]:


piezos_final_interpolado = piezos_final.interpolate(method="ffill")


# In[124]:


# Identificar NAs
nas = piezos_final_interpolado.isnull().sum()
print("Cantidad de NAs por columna:")
print(nas)


# In[125]:


data_description = piezos_final_interpolado.describe()
print("\nDescripción de los datos:")
print(data_description)


# # Pluvios

# In[126]:


# Cargamos los pluviometros
pluvios = cargar_parquets('../raw/SAIHdatasActualizados/Ramblas/', "Pluviometro")


# Quitar espacios, NAs, float, fechas...
# 

# In[127]:


pluviometros = {clave.replace(' ', ''): valor for clave, valor in pluvios.items()}

for key, dataframe in pluviometros.items():
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

for key, dataframe in pluviometros.items():
    pluviometros[key] = dataframe.replace('-', np.nan)

for key, dataframe in pluviometros.items():
    dataframe[dataframe.columns[1]] = dataframe[dataframe.columns[1]].astype(float)

DateIndex = pd.DataFrame(pd.date_range(start=pd.to_datetime('2021-03-08 12:00:00'), # 2021-03-08 11:35:00  2021-01-12 12:00:00
                                       end = ('2023-11-18 23:55:00'), 
                                       freq = '5T'),
                        columns=['Date'])

resultado = DateIndex

for name, df in pluviometros.items():
        resultado = resultado.merge(df, on='Date', how='left')
        resultado = resultado.rename(columns={'Pluviometro':name})
        
pluviometros_ok = resultado.drop_duplicates(subset='Date')



# In[128]:


# Identificar NAs
nas = pluviometros_ok.isnull().sum()
print("Cantidad de NAs por columna:")
print(nas)


# In[129]:


# La marana y la desembocadura fuera
columnas_a_eliminar = ['06A19-LaMarana', '06A18-DesembocRblaAlbujon']

# Eliminar las columnas especificadas
pluviometros_final = pluviometros_ok.drop(columnas_a_eliminar, axis=1)


# In[130]:


pluviometros_final_interpolado = pluviometros_final.interpolate("ffill")


# In[132]:


# Identificar NAs
nas = pluviometros_final_interpolado.isnull().sum()
print("Cantidad de NAs por columna:")
print(nas)


# In[133]:


# Para obtener el de la hora
resampleado_pluviometros = resample_df(pluviometros_final_interpolado, "Date","H")


# In[134]:


resampleado_Piezometrico = resample_df(piezos_final_interpolado, "Date","H")


# In[135]:


# Quitar el guión
resampleado_pluviometros.columns = resampleado_pluviometros.columns.str.replace('-', '')
resampleado_caudal.columns = resampleado_caudal.columns.str.replace('-', '')
resampleado_Piezometrico.columns = resampleado_Piezometrico.columns.str.replace('-', '')


# In[136]:
# La hora que quiera
resampleado_caudal_24h = resampleado_caudal.copy()
resampleado_caudal_24h['mean_H06A01LaPuebla_24h'] = resampleado_caudal_24h['mean_H06A01LaPuebla'].shift(-24)

# Estos son dataframe que están indexados y los tengo que quitar para que sea una columna
Piezometrico_shift = shift_df(resampleado_Piezometrico, 1, 'Piezometrico')
caudal_shift = shift_df(resampleado_caudal, 1, 'caudal')
pluviometros_shift = shift_df(resampleado_pluviometros, 1, 'pluvios')

Piezometrico_shift['Date'] = Piezometrico_shift.index
caudal_shift['Date'] = caudal_shift.index
pluviometros_shift['Date'] = pluviometros_shift.index


# Hacer un rolado 6, 12, 18 de caudales y pluvios (Esto serían valores sobre el dataset cada 5 min)
rolado_caudal_6 =rolling_df(caudal_shift, "Date", 6, "caudal")
rolado_caudal_12 =rolling_df(caudal_shift, "Date", 12, "caudal")
rolado_caudal_18 =rolling_df(caudal_shift, "Date", 18, "caudal")
rolado_caudal_24 =rolling_df(caudal_shift, "Date", 24, "caudal")


rolado_pluvios_6 =rolling_df(pluviometros_shift, "Date", 6, "pluvios")
rolado_pluvios_12 =rolling_df(pluviometros_shift, "Date", 12, "pluvios")
rolado_pluvios_18 =rolling_df(pluviometros_shift, "Date", 18, "pluvios")
rolado_pluvios_24 =rolling_df(pluviometros_shift, "Date", 24, "pluvios")

rolado_Piezometrico_6 =rolling_df(Piezometrico_shift, "Date", 6, "Piezometrico")
rolado_Piezometrico_12 =rolling_df(Piezometrico_shift, "Date", 12, "Piezometrico")
rolado_Piezometrico_18 =rolling_df(Piezometrico_shift, "Date", 18, "Piezometrico")
rolado_Piezometrico_24 =rolling_df(Piezometrico_shift, "Date", 24, "Piezometrico")



# Mejor resultado con 6 (Valores resampleados horarios hasta 6 horas antes del momento a predecir)
caudal_shift_6 = shift_df(resampleado_caudal, 6, 'caudal')
pluviometros_shift_6 = shift_df(resampleado_pluviometros, 6, 'pluvios')
Piezometrico_shift_6 = shift_df(resampleado_Piezometrico, 6, 'Piezometrico')





# Mejor resultado antes de std desviation, con 6 shift y con rolados 6, 12 y 18. Sin selección de variables
result = pd.concat([resampleado_caudal_24h['mean_H06A01LaPuebla_24h'], caudal_shift_6, pluviometros_shift_6, Piezometrico_shift_6, rolado_caudal_6, 
                    rolado_caudal_12, rolado_caudal_18, rolado_caudal_24, rolado_pluvios_6,  rolado_pluvios_12, rolado_pluvios_18, rolado_pluvios_24,
                    rolado_Piezometrico_6, rolado_Piezometrico_12, rolado_Piezometrico_18, rolado_Piezometrico_24], axis=1)

result_sinna = result.dropna()

random.seed(123)
result_sinna.sort_index(inplace=True)

X = result_sinna.drop('mean_H06A01LaPuebla_24h',  axis=1)
y = result_sinna['mean_H06A01LaPuebla_24h']

# Verifica si tu DataFrame tiene columnas duplicadas
duplicate_columns = X.columns[X.columns.duplicated()]
print("Columnas duplicadas: ", duplicate_columns)


random.seed(123)
# Define el número de divisiones para la validación cruzada
tscv = TimeSeriesSplit(n_splits=3)
# Con 100
tscv.split(X)

for train_index, test_index in tscv.split(X):
    n5train = train_index
    n5test = test_index
    
# Selección de variables

column_names = X.columns

random.seed(123)
# Partir el dataset en train y test de forma temporal
X_train, X_test = X.iloc[n5train], X.iloc[n5test]
y_train, y_test = y.iloc[n5train], y.iloc[n5test]
    
# Escalar los datos
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=100)  # Ejemplo: selecciona las mejores 5 características

selector.fit(X_train, y_train)

selected_indices = selector.get_support()

# Get the selected column names
selected_column_names = [column_names[i] for i, selected in enumerate(selected_indices) if selected]

print("Selected columns:", selected_column_names)


result = pd.concat([resampleado_caudal_24h['mean_H06A01LaPuebla_24h'], caudal_shift_6, pluviometros_shift_6, Piezometrico_shift_6, rolado_caudal_6, 
                    rolado_caudal_12, rolado_caudal_18, rolado_caudal_24, rolado_pluvios_6,  rolado_pluvios_12, rolado_pluvios_18, rolado_pluvios_24,
                    rolado_Piezometrico_6, rolado_Piezometrico_12, rolado_Piezometrico_18, rolado_Piezometrico_24], axis=1)

result_sinna = result.dropna()

random.seed(123)
result_sinna.sort_index(inplace=True)

X = result_sinna.drop('mean_H06A01LaPuebla_24h',  axis=1)
y = result_sinna['mean_H06A01LaPuebla_24h']

X = X[selected_column_names]

# Verifica si tu DataFrame tiene columnas duplicadas
duplicate_columns = X.columns[X.columns.duplicated()]
print("Columnas duplicadas: ", duplicate_columns)


df_final_arurora = X.copy()

df_final_arurora["mean_H06A01LaPuebla_24h"] = result_sinna['mean_H06A01LaPuebla_24h']


datos_entrenamiento = df_final_arurora.iloc[n5train]
datos_prueba = df_final_arurora.iloc[n5test]

print(datos_entrenamiento.shape)
print(datos_prueba.shape)


datos_entrenamiento.to_csv('../acg_paper/24h_train_100variables.csv')
datos_prueba.to_csv('../acg_paper/24h_test_100variables.csv')



# # KNeighborsRegressor

# In[148]:


from cgi import test
from logging import BufferingFormatter
from tabnanny import verbose


random.seed(123)
# Define el número de divisiones para la validación cruzada
tscv = TimeSeriesSplit(n_splits=3)

# Elegir el modelo ordenados de mejor a peor
# model = GradientBoostingRegressor(random_state=42)
# model = RandomForestRegressor(random_state=42)
# model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)
# model = DecisionTreeRegressor(random_state=42) Bueno 
# model = KNeighborsRegressor()
# model = Lasso() ¿No converge?
# model = Ridge(alpha=1.0) Regular
# model = SVR() Malo
# model = LinearRegression() Regular

model = KNeighborsRegressor()
#model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)



# Lista para almacenar los valores deseados
predictions_list = []
real_values_list = []
index_list = []
train_values_list = []
split_num_list = []
split_num_list_train = []
index_list_train = []

split_num = 1

for train_index, test_index in tscv.split(X):
    random.seed(123)
    # Partir el dataset en train y test de forma temporal
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Escalar los datos
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar el modelo
    # Para XGBRegressor
    #model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)],verbose = 100)
    model.fit(X_train, y_train)

    
    # Predecir
    predictions = model.predict(X_test)
    
     # Almacena las predicciones, los valores reales y los índices en las listas
    predictions_list.extend(predictions)
    real_values_list.extend(y_test.values)
    index_list.extend(y_test.index)
    train_values_list.extend(y_train.values)
    split_num_list.extend([split_num]*len(y_test))
    split_num_list_train.extend([split_num]*len(y_train))
    index_list_train.extend(y_train.index)
    
    split_num += 1

# Convierte las listas en un DataFrame
results_df_predictions = pd.DataFrame({
    'Predictions': predictions_list,
    'Y_Real_Values': real_values_list,
    'Date': index_list,
    'Split_Num': split_num_list
})

results_df_train = pd.DataFrame({
    'Train_Values': train_values_list,
    'Split_Num':     split_num_list_train,
    'Date': index_list_train,
})
 


# In[149]:


# Asegúrate de que 'Date' es un objeto datetime
results_df_predictions['Date'] = pd.to_datetime(results_df_predictions['Date'])
results_df_train['Date'] = pd.to_datetime(results_df_train['Date'])

# Número de splits
n_splits = results_df_train['Split_Num'].nunique()

for split_num in range(1, n_splits + 1):
    plt.figure(figsize=(10, 6))
    
    # Filtra los datos por número de split
    train_data = results_df_train[results_df_train['Split_Num'] == split_num]
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Trazar los datos de entrenamiento, los datos reales y las predicciones
    plt.plot(train_data['Date'], train_data['Train_Values'], label='Train Values')
    plt.plot(pred_data['Date'], pred_data['Y_Real_Values'], label='Y_Real_Values')
    plt.plot(pred_data['Date'], pred_data['Predictions'], label='Predictions', alpha=0.6)
    
    plt.title(f'Split {split_num}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(f'../acg_paper/24H_k100_KNeighborsRegressor_Split_{split_num}.png')
    plt.show()


# In[150]:


def nash_sutcliffe_efficiency(observed, simulated):
    """
    Calcula el coeficiente de eficiencia de Nash-Sutcliffe (NSE).
    
    Args:
    observed: Array de numpy con los valores observados.
    simulated: Array de numpy con los valores simulados o predichos.
    
    Returns:
    nse: Coeficiente de eficiencia de Nash-Sutcliffe.
    """
    # Calcular el promedio de los valores observados
    mean_observed = np.mean(observed)
    
    # Calcular el error cuadrático
    numerator = np.sum((observed - simulated) ** 2)
    
    # Calcular la varianza de los valores observados
    denominator = np.sum((observed - mean_observed) ** 2)
    
    # Calcular el NSE
    nse = 1 - (numerator / denominator)
    
    return nse


def willmott_index(observed, simulated):
    """
    Calcula el índice de Willmott entre dos conjuntos de datos.
    
    Args:
    observed: Array de numpy con los valores observados.
    simulated: Array de numpy con los valores simulados o predichos.
    
    Returns:
    willmott: Índice de Willmott.
    """
    # Calcular la diferencia entre valores observados y simulados
    numerator = np.sum(np.abs(simulated - observed))
    
    # Calcular la diferencia entre los valores simulados y la media de los valores observados
    denominator = np.sum(np.abs(np.abs(simulated - np.mean(observed)) + np.abs(observed - np.mean(observed))))
    
    # Calcular el índice de Willmott
    willmott = 1 - (numerator / denominator)
    
    return willmott




# Calculo de errores por split de la serie
# Número de splits
n_splits = results_df_predictions['Split_Num'].nunique()

# Listas para almacenar los errores
mae_list = []
rmse_list = []
cvrmse_list = []
nash_list = []
willmott_list = []

for split_num in range(1, n_splits + 1):
    # Filtra los datos por número de split
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Calcular MAE y MSE
    mae = mean_absolute_error(pred_data['Y_Real_Values'], pred_data['Predictions'])
    rmse = np.sqrt(mean_squared_error(pred_data['Y_Real_Values'], pred_data['Predictions']))
    mean_obs = np.mean(pred_data['Y_Real_Values'])
    cvrmse = (rmse / mean_obs) * 100  # en porcentaje
    variance = np.var(pred_data['Y_Real_Values'])
    nash = nash_sutcliffe_efficiency(pred_data['Y_Real_Values'], pred_data['Predictions'])
    willmott = willmott_index(pred_data['Y_Real_Values'], pred_data['Predictions'])
    
    

    mae_list.append(mae)
    rmse_list.append(rmse)
    cvrmse_list.append(cvrmse)
    nash_list.append(nash)
    willmott_list.append(willmott)
    
    
  
# Crear un DataFrame con los errores
errors_df = pd.DataFrame({
    'Split_Num': range(1, n_splits + 1),
    'MAE': mae_list,
    'RMSE': rmse_list,
    'CVRMSE': cvrmse_list,
    'WI': willmott_list,
    'NSE': nash_list

})



errors_df.to_csv('../acg_paper/24h_puebla_k100_KNeighborsRegressor_errors_table.csv', index=False, float_format='%.4f') 


# #  Linear regression

# In[144]:


random.seed(123)
# Define el número de divisiones para la validación cruzada
tscv = TimeSeriesSplit(n_splits=3)

# Elegir el modelo ordenados de mejor a peor
# model = GradientBoostingRegressor(random_state=42)
# model = RandomForestRegressor(random_state=42)
# model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)
# model = DecisionTreeRegressor(random_state=42) Bueno 
# model = KNeighborsRegressor()
# model = Lasso() ¿No converge?
# model = Ridge(alpha=1.0) Regular
# model = SVR() Malo
# model = LinearRegression() Regular

model = LinearRegression()
#model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)



# Lista para almacenar los valores deseados
predictions_list = []
real_values_list = []
index_list = []
train_values_list = []
split_num_list = []
split_num_list_train = []
index_list_train = []

split_num = 1

for train_index, test_index in tscv.split(X):
    random.seed(123)
    # Partir el dataset en train y test de forma temporal
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Escalar los datos
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar el modelo
    # Para XGBRegressor
    #model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)],verbose = 100)
    model.fit(X_train, y_train)

    
    # Predecir
    predictions = model.predict(X_test)
    
     # Almacena las predicciones, los valores reales y los índices en las listas
    predictions_list.extend(predictions)
    real_values_list.extend(y_test.values)
    index_list.extend(y_test.index)
    train_values_list.extend(y_train.values)
    split_num_list.extend([split_num]*len(y_test))
    split_num_list_train.extend([split_num]*len(y_train))
    index_list_train.extend(y_train.index)
    
    split_num += 1

# Convierte las listas en un DataFrame
results_df_predictions = pd.DataFrame({
    'Predictions': predictions_list,
    'Y_Real_Values': real_values_list,
    'Date': index_list,
    'Split_Num': split_num_list
})

results_df_train = pd.DataFrame({
    'Train_Values': train_values_list,
    'Split_Num':     split_num_list_train,
    'Date': index_list_train,
})
 


# In[145]:


# Asegúrate de que 'Date' es un objeto datetime
results_df_predictions['Date'] = pd.to_datetime(results_df_predictions['Date'])
results_df_train['Date'] = pd.to_datetime(results_df_train['Date'])

# Número de splits
n_splits = results_df_train['Split_Num'].nunique()

for split_num in range(1, n_splits + 1):
    plt.figure(figsize=(10, 6))
    
    # Filtra los datos por número de split
    train_data = results_df_train[results_df_train['Split_Num'] == split_num]
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Trazar los datos de entrenamiento, los datos reales y las predicciones
    plt.plot(train_data['Date'], train_data['Train_Values'], label='Train Values')
    plt.plot(pred_data['Date'], pred_data['Y_Real_Values'], label='Y_Real_Values')
    plt.plot(pred_data['Date'], pred_data['Predictions'], label='Predictions', alpha=0.6)
    
    plt.title(f'Split {split_num}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(f'../acg_paper/24H_k100_LinearRegression_Split_{split_num}.png')
    plt.show()


# In[147]:



# Calculo de errores por split de la serie
# Número de splits
n_splits = results_df_predictions['Split_Num'].nunique()

# Listas para almacenar los errores
mae_list = []
rmse_list = []
cvrmse_list = []
nash_list = []
willmott_list = []

for split_num in range(1, n_splits + 1):
    # Filtra los datos por número de split
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Calcular MAE y MSE
    mae = mean_absolute_error(pred_data['Y_Real_Values'], pred_data['Predictions'])
    rmse = np.sqrt(mean_squared_error(pred_data['Y_Real_Values'], pred_data['Predictions']))
    mean_obs = np.mean(pred_data['Y_Real_Values'])
    cvrmse = (rmse / mean_obs) * 100  # en porcentaje
    variance = np.var(pred_data['Y_Real_Values'])
    nash = nash_sutcliffe_efficiency(pred_data['Y_Real_Values'], pred_data['Predictions'])
    willmott = willmott_index(pred_data['Y_Real_Values'], pred_data['Predictions'])
    
    

    mae_list.append(mae)
    rmse_list.append(rmse)
    cvrmse_list.append(cvrmse)
    nash_list.append(nash)
    willmott_list.append(willmott)
    
    
  
# Crear un DataFrame con los errores
errors_df = pd.DataFrame({
    'Split_Num': range(1, n_splits + 1),
    'MAE': mae_list,
    'RMSE': rmse_list,
    'CVRMSE': cvrmse_list,
    'WI': willmott_list,
    'NSE': nash_list

})



errors_df.to_csv('../acg_paper/24H_k100_LinearRegression_errors_table.csv', index=False, float_format='%.4f') 



# #  GradientBoostingRegressor

# In[144]:


random.seed(123)
# Define el número de divisiones para la validación cruzada
tscv = TimeSeriesSplit(n_splits=3)

# Elegir el modelo ordenados de mejor a peor
# model = GradientBoostingRegressor(random_state=42)
# model = RandomForestRegressor(random_state=42)
# model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)
# model = DecisionTreeRegressor(random_state=42) Bueno 
# model = KNeighborsRegressor()
# model = Lasso() ¿No converge?
# model = Ridge(alpha=1.0) Regular
# model = SVR() Malo
# model = LinearRegression() Regular

model = GradientBoostingRegressor(random_state=42)
#model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)



# Lista para almacenar los valores deseados
predictions_list = []
real_values_list = []
index_list = []
train_values_list = []
split_num_list = []
split_num_list_train = []
index_list_train = []

split_num = 1

for train_index, test_index in tscv.split(X):
    random.seed(123)
    # Partir el dataset en train y test de forma temporal
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Escalar los datos
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar el modelo
    # Para XGBRegressor
    #model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)],verbose = 100)
    model.fit(X_train, y_train)

    
    # Predecir
    predictions = model.predict(X_test)
    
     # Almacena las predicciones, los valores reales y los índices en las listas
    predictions_list.extend(predictions)
    real_values_list.extend(y_test.values)
    index_list.extend(y_test.index)
    train_values_list.extend(y_train.values)
    split_num_list.extend([split_num]*len(y_test))
    split_num_list_train.extend([split_num]*len(y_train))
    index_list_train.extend(y_train.index)
    
    split_num += 1

# Convierte las listas en un DataFrame
results_df_predictions = pd.DataFrame({
    'Predictions': predictions_list,
    'Y_Real_Values': real_values_list,
    'Date': index_list,
    'Split_Num': split_num_list
})

results_df_train = pd.DataFrame({
    'Train_Values': train_values_list,
    'Split_Num':     split_num_list_train,
    'Date': index_list_train,
})
 


# In[145]:


# Asegúrate de que 'Date' es un objeto datetime
results_df_predictions['Date'] = pd.to_datetime(results_df_predictions['Date'])
results_df_train['Date'] = pd.to_datetime(results_df_train['Date'])

# Número de splits
n_splits = results_df_train['Split_Num'].nunique()

for split_num in range(1, n_splits + 1):
    plt.figure(figsize=(10, 6))
    
    # Filtra los datos por número de split
    train_data = results_df_train[results_df_train['Split_Num'] == split_num]
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Trazar los datos de entrenamiento, los datos reales y las predicciones
    plt.plot(train_data['Date'], train_data['Train_Values'], label='Train Values')
    plt.plot(pred_data['Date'], pred_data['Y_Real_Values'], label='Y_Real_Values')
    plt.plot(pred_data['Date'], pred_data['Predictions'], label='Predictions', alpha=0.6)
    
    plt.title(f'Split {split_num}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(f'../acg_paper/24H_k100_GradientBoostingRegressor_Split_{split_num}.png')
    plt.show()


# In[147]:
# Calculo de errores por split de la serie
# Número de splits
n_splits = results_df_predictions['Split_Num'].nunique()

# Listas para almacenar los errores
mae_list = []
rmse_list = []
cvrmse_list = []
nash_list = []
willmott_list = []

for split_num in range(1, n_splits + 1):
    # Filtra los datos por número de split
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Calcular MAE y MSE
    mae = mean_absolute_error(pred_data['Y_Real_Values'], pred_data['Predictions'])
    rmse = np.sqrt(mean_squared_error(pred_data['Y_Real_Values'], pred_data['Predictions']))
    mean_obs = np.mean(pred_data['Y_Real_Values'])
    cvrmse = (rmse / mean_obs) * 100  # en porcentaje
    variance = np.var(pred_data['Y_Real_Values'])
    nash = nash_sutcliffe_efficiency(pred_data['Y_Real_Values'], pred_data['Predictions'])
    willmott = willmott_index(pred_data['Y_Real_Values'], pred_data['Predictions'])
    
    

    mae_list.append(mae)
    rmse_list.append(rmse)
    cvrmse_list.append(cvrmse)
    nash_list.append(nash)
    willmott_list.append(willmott)
    
    
  
# Crear un DataFrame con los errores
errors_df = pd.DataFrame({
    'Split_Num': range(1, n_splits + 1),
    'MAE': mae_list,
    'RMSE': rmse_list,
    'CVRMSE': cvrmse_list,
    'WI': willmott_list,
    'NSE': nash_list

})




errors_df.to_csv('../acg_paper/24H_k100_GradientBoostingRegressor_errors_table.csv', index=False, float_format='%.4f') 


# #  RandomForestRegressor

# In[144]:


random.seed(123)
# Define el número de divisiones para la validación cruzada
tscv = TimeSeriesSplit(n_splits=3)

# Elegir el modelo ordenados de mejor a peor
# model = GradientBoostingRegressor(random_state=42)
# model = RandomForestRegressor(random_state=42)
# model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)
# model = DecisionTreeRegressor(random_state=42) Bueno 
# model = KNeighborsRegressor()
# model = Lasso() ¿No converge?
# model = Ridge(alpha=1.0) Regular
# model = SVR() Malo
# model = LinearRegression() Regular

model = RandomForestRegressor(random_state=42)
#model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)



# Lista para almacenar los valores deseados
predictions_list = []
real_values_list = []
index_list = []
train_values_list = []
split_num_list = []
split_num_list_train = []
index_list_train = []

split_num = 1

for train_index, test_index in tscv.split(X):
    random.seed(123)
    # Partir el dataset en train y test de forma temporal
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Escalar los datos
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar el modelo
    # Para XGBRegressor
    #model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)],verbose = 100)
    model.fit(X_train, y_train)

    
    # Predecir
    predictions = model.predict(X_test)
    
     # Almacena las predicciones, los valores reales y los índices en las listas
    predictions_list.extend(predictions)
    real_values_list.extend(y_test.values)
    index_list.extend(y_test.index)
    train_values_list.extend(y_train.values)
    split_num_list.extend([split_num]*len(y_test))
    split_num_list_train.extend([split_num]*len(y_train))
    index_list_train.extend(y_train.index)
    
    split_num += 1

# Convierte las listas en un DataFrame
results_df_predictions = pd.DataFrame({
    'Predictions': predictions_list,
    'Y_Real_Values': real_values_list,
    'Date': index_list,
    'Split_Num': split_num_list
})

results_df_train = pd.DataFrame({
    'Train_Values': train_values_list,
    'Split_Num':     split_num_list_train,
    'Date': index_list_train,
})
 


# In[145]:


# Asegúrate de que 'Date' es un objeto datetime
results_df_predictions['Date'] = pd.to_datetime(results_df_predictions['Date'])
results_df_train['Date'] = pd.to_datetime(results_df_train['Date'])

# Número de splits
n_splits = results_df_train['Split_Num'].nunique()

for split_num in range(1, n_splits + 1):
    plt.figure(figsize=(10, 6))
    
    # Filtra los datos por número de split
    train_data = results_df_train[results_df_train['Split_Num'] == split_num]
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Trazar los datos de entrenamiento, los datos reales y las predicciones
    plt.plot(train_data['Date'], train_data['Train_Values'], label='Train Values')
    plt.plot(pred_data['Date'], pred_data['Y_Real_Values'], label='Y_Real_Values')
    plt.plot(pred_data['Date'], pred_data['Predictions'], label='Predictions', alpha=0.6)
    
    plt.title(f'Split {split_num}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(f'../acg_paper/24H_k100_RandomForestRegressor_Split_{split_num}.png')
    plt.show()


# In[147]:

# Número de splits
n_splits = results_df_predictions['Split_Num'].nunique()

# Listas para almacenar los errores
mae_list = []
rmse_list = []
cvrmse_list = []
nash_list = []
willmott_list = []

for split_num in range(1, n_splits + 1):
    # Filtra los datos por número de split
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Calcular MAE y MSE
    mae = mean_absolute_error(pred_data['Y_Real_Values'], pred_data['Predictions'])
    rmse = np.sqrt(mean_squared_error(pred_data['Y_Real_Values'], pred_data['Predictions']))
    mean_obs = np.mean(pred_data['Y_Real_Values'])
    cvrmse = (rmse / mean_obs) * 100  # en porcentaje
    variance = np.var(pred_data['Y_Real_Values'])
    nash = nash_sutcliffe_efficiency(pred_data['Y_Real_Values'], pred_data['Predictions'])
    willmott = willmott_index(pred_data['Y_Real_Values'], pred_data['Predictions'])
    
    

    mae_list.append(mae)
    rmse_list.append(rmse)
    cvrmse_list.append(cvrmse)
    nash_list.append(nash)
    willmott_list.append(willmott)
    
    
  
# Crear un DataFrame con los errores
errors_df = pd.DataFrame({
    'Split_Num': range(1, n_splits + 1),
    'MAE': mae_list,
    'RMSE': rmse_list,
    'CVRMSE': cvrmse_list,
    'WI': willmott_list,
    'NSE': nash_list

})




errors_df.to_csv('../acg_paper/24H_k100_RandomForestRegressor_errors_table.csv', index=False, float_format='%.4f') 

# #  DecisionTreeRegressor

# In[144]:


random.seed(123)
# Define el número de divisiones para la validación cruzada
tscv = TimeSeriesSplit(n_splits=3)

# Elegir el modelo ordenados de mejor a peor

# model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)
# model = DecisionTreeRegressor(random_state=42) Bueno 
# model = Ridge(alpha=1.0) Regular


model = DecisionTreeRegressor(random_state=42)
#model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)



# Lista para almacenar los valores deseados
predictions_list = []
real_values_list = []
index_list = []
train_values_list = []
split_num_list = []
split_num_list_train = []
index_list_train = []

split_num = 1

for train_index, test_index in tscv.split(X):
    random.seed(123)
    # Partir el dataset en train y test de forma temporal
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Escalar los datos
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar el modelo
    # Para XGBRegressor
    #model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)],verbose = 100)
    model.fit(X_train, y_train)

    
    # Predecir
    predictions = model.predict(X_test)
    
     # Almacena las predicciones, los valores reales y los índices en las listas
    predictions_list.extend(predictions)
    real_values_list.extend(y_test.values)
    index_list.extend(y_test.index)
    train_values_list.extend(y_train.values)
    split_num_list.extend([split_num]*len(y_test))
    split_num_list_train.extend([split_num]*len(y_train))
    index_list_train.extend(y_train.index)
    
    split_num += 1

# Convierte las listas en un DataFrame
results_df_predictions = pd.DataFrame({
    'Predictions': predictions_list,
    'Y_Real_Values': real_values_list,
    'Date': index_list,
    'Split_Num': split_num_list
})

results_df_train = pd.DataFrame({
    'Train_Values': train_values_list,
    'Split_Num':     split_num_list_train,
    'Date': index_list_train,
})
 


# In[145]:


# Asegúrate de que 'Date' es un objeto datetime
results_df_predictions['Date'] = pd.to_datetime(results_df_predictions['Date'])
results_df_train['Date'] = pd.to_datetime(results_df_train['Date'])

# Número de splits
n_splits = results_df_train['Split_Num'].nunique()

for split_num in range(1, n_splits + 1):
    plt.figure(figsize=(10, 6))
    
    # Filtra los datos por número de split
    train_data = results_df_train[results_df_train['Split_Num'] == split_num]
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Trazar los datos de entrenamiento, los datos reales y las predicciones
    plt.plot(train_data['Date'], train_data['Train_Values'], label='Train Values')
    plt.plot(pred_data['Date'], pred_data['Y_Real_Values'], label='Y_Real_Values')
    plt.plot(pred_data['Date'], pred_data['Predictions'], label='Predictions', alpha=0.6)
    
    plt.title(f'Split {split_num}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(f'../acg_paper/24H_k100_DecisionTreeRegressor_Split_{split_num}.png')
    plt.show()


# In[147]:
# Número de splits
n_splits = results_df_predictions['Split_Num'].nunique()

# Listas para almacenar los errores
mae_list = []
rmse_list = []
cvrmse_list = []
nash_list = []
willmott_list = []

for split_num in range(1, n_splits + 1):
    # Filtra los datos por número de split
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Calcular MAE y MSE
    mae = mean_absolute_error(pred_data['Y_Real_Values'], pred_data['Predictions'])
    rmse = np.sqrt(mean_squared_error(pred_data['Y_Real_Values'], pred_data['Predictions']))
    mean_obs = np.mean(pred_data['Y_Real_Values'])
    cvrmse = (rmse / mean_obs) * 100  # en porcentaje
    variance = np.var(pred_data['Y_Real_Values'])
    nash = nash_sutcliffe_efficiency(pred_data['Y_Real_Values'], pred_data['Predictions'])
    willmott = willmott_index(pred_data['Y_Real_Values'], pred_data['Predictions'])
    
    

    mae_list.append(mae)
    rmse_list.append(rmse)
    cvrmse_list.append(cvrmse)
    nash_list.append(nash)
    willmott_list.append(willmott)
    
    
  
# Crear un DataFrame con los errores
errors_df = pd.DataFrame({
    'Split_Num': range(1, n_splits + 1),
    'MAE': mae_list,
    'RMSE': rmse_list,
    'CVRMSE': cvrmse_list,
    'WI': willmott_list,
    'NSE': nash_list

})



errors_df.to_csv('../acg_paper/24H_k100_DecisionTreeRegressor_errors_table.csv', index=False, float_format='%.4f') 


# #  Ridge(alpha=1.0)

# In[144]:


random.seed(123)
# Define el número de divisiones para la validación cruzada
tscv = TimeSeriesSplit(n_splits=3)

# Elegir el modelo ordenados de mejor a peor

# model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)
# model = DecisionTreeRegressor(random_state=42) Bueno 
# model = Ridge(alpha=1.0) Regular


model = Ridge(alpha=1.0)
#model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)



# Lista para almacenar los valores deseados
predictions_list = []
real_values_list = []
index_list = []
train_values_list = []
split_num_list = []
split_num_list_train = []
index_list_train = []

split_num = 1

for train_index, test_index in tscv.split(X):
    random.seed(123)
    # Partir el dataset en train y test de forma temporal
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Escalar los datos
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar el modelo
    # Para XGBRegressor
    #model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)],verbose = 100)
    model.fit(X_train, y_train)

    
    # Predecir
    predictions = model.predict(X_test)
    
     # Almacena las predicciones, los valores reales y los índices en las listas
    predictions_list.extend(predictions)
    real_values_list.extend(y_test.values)
    index_list.extend(y_test.index)
    train_values_list.extend(y_train.values)
    split_num_list.extend([split_num]*len(y_test))
    split_num_list_train.extend([split_num]*len(y_train))
    index_list_train.extend(y_train.index)
    
    split_num += 1

# Convierte las listas en un DataFrame
results_df_predictions = pd.DataFrame({
    'Predictions': predictions_list,
    'Y_Real_Values': real_values_list,
    'Date': index_list,
    'Split_Num': split_num_list
})

results_df_train = pd.DataFrame({
    'Train_Values': train_values_list,
    'Split_Num':     split_num_list_train,
    'Date': index_list_train,
})
 


# In[145]:


# Asegúrate de que 'Date' es un objeto datetime
results_df_predictions['Date'] = pd.to_datetime(results_df_predictions['Date'])
results_df_train['Date'] = pd.to_datetime(results_df_train['Date'])

# Número de splits
n_splits = results_df_train['Split_Num'].nunique()

for split_num in range(1, n_splits + 1):
    plt.figure(figsize=(10, 6))
    
    # Filtra los datos por número de split
    train_data = results_df_train[results_df_train['Split_Num'] == split_num]
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Trazar los datos de entrenamiento, los datos reales y las predicciones
    plt.plot(train_data['Date'], train_data['Train_Values'], label='Train Values')
    plt.plot(pred_data['Date'], pred_data['Y_Real_Values'], label='Y_Real_Values')
    plt.plot(pred_data['Date'], pred_data['Predictions'], label='Predictions', alpha=0.6)
    
    plt.title(f'Split {split_num}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(f'../acg_paper/24H_k100_Ridge_Split_{split_num}.png')
    plt.show()


# In[147]:


# In[147]:
# Número de splits
n_splits = results_df_predictions['Split_Num'].nunique()

# Listas para almacenar los errores
mae_list = []
rmse_list = []
cvrmse_list = []
nash_list = []
willmott_list = []

for split_num in range(1, n_splits + 1):
    # Filtra los datos por número de split
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Calcular MAE y MSE
    mae = mean_absolute_error(pred_data['Y_Real_Values'], pred_data['Predictions'])
    rmse = np.sqrt(mean_squared_error(pred_data['Y_Real_Values'], pred_data['Predictions']))
    mean_obs = np.mean(pred_data['Y_Real_Values'])
    cvrmse = (rmse / mean_obs) * 100  # en porcentaje
    variance = np.var(pred_data['Y_Real_Values'])
    nash = nash_sutcliffe_efficiency(pred_data['Y_Real_Values'], pred_data['Predictions'])
    willmott = willmott_index(pred_data['Y_Real_Values'], pred_data['Predictions'])
    
    

    mae_list.append(mae)
    rmse_list.append(rmse)
    cvrmse_list.append(cvrmse)
    nash_list.append(nash)
    willmott_list.append(willmott)
    
    
  
# Crear un DataFrame con los errores
errors_df = pd.DataFrame({
    'Split_Num': range(1, n_splits + 1),
    'MAE': mae_list,
    'RMSE': rmse_list,
    'CVRMSE': cvrmse_list,
    'WI': willmott_list,
    'NSE': nash_list

})



errors_df.to_csv('../acg_paper/24H_k100_Ridge_errors_table.csv', index=False, float_format='%.4f') 


# #  XGBRegressor

# In[144]:


random.seed(123)
# Define el número de divisiones para la validación cruzada
tscv = TimeSeriesSplit(n_splits=3)

# Elegir el modelo ordenados de mejor a peor

# model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)
# model = DecisionTreeRegressor(random_state=42) Bueno 
# model = Ridge(alpha=1.0) Regular


#model = Ridge(alpha=1.0)
model = XGBRegressor(n_estimators=1000, early_stopping_rounds= 50, random_state=42, learning_rate = 0.01)



# Lista para almacenar los valores deseados
predictions_list = []
real_values_list = []
index_list = []
train_values_list = []
split_num_list = []
split_num_list_train = []
index_list_train = []

split_num = 1

for train_index, test_index in tscv.split(X):
    random.seed(123)
    # Partir el dataset en train y test de forma temporal
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Escalar los datos
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar el modelo
    # Para XGBRegressor
    model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)],verbose = 100)
    #model.fit(X_train, y_train)

    
    # Predecir
    predictions = model.predict(X_test)
    
     # Almacena las predicciones, los valores reales y los índices en las listas
    predictions_list.extend(predictions)
    real_values_list.extend(y_test.values)
    index_list.extend(y_test.index)
    train_values_list.extend(y_train.values)
    split_num_list.extend([split_num]*len(y_test))
    split_num_list_train.extend([split_num]*len(y_train))
    index_list_train.extend(y_train.index)
    
    split_num += 1

# Convierte las listas en un DataFrame
results_df_predictions = pd.DataFrame({
    'Predictions': predictions_list,
    'Y_Real_Values': real_values_list,
    'Date': index_list,
    'Split_Num': split_num_list
})

results_df_train = pd.DataFrame({
    'Train_Values': train_values_list,
    'Split_Num':     split_num_list_train,
    'Date': index_list_train,
})
 


# In[145]:


# Asegúrate de que 'Date' es un objeto datetime
results_df_predictions['Date'] = pd.to_datetime(results_df_predictions['Date'])
results_df_train['Date'] = pd.to_datetime(results_df_train['Date'])

# Número de splits
n_splits = results_df_train['Split_Num'].nunique()

for split_num in range(1, n_splits + 1):
    plt.figure(figsize=(10, 6))
    
    # Filtra los datos por número de split
    train_data = results_df_train[results_df_train['Split_Num'] == split_num]
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Trazar los datos de entrenamiento, los datos reales y las predicciones
    plt.plot(train_data['Date'], train_data['Train_Values'], label='Train Values')
    plt.plot(pred_data['Date'], pred_data['Y_Real_Values'], label='Y_Real_Values')
    plt.plot(pred_data['Date'], pred_data['Predictions'], label='Predictions', alpha=0.6)
    
    plt.title(f'Split {split_num}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(f'../acg_paper/24H_k100_XGBRegressor_Split_{split_num}.png')
    plt.show()


# In[147]:

# In[147]:
# Número de splits
n_splits = results_df_predictions['Split_Num'].nunique()

# Listas para almacenar los errores
mae_list = []
rmse_list = []
cvrmse_list = []
nash_list = []
willmott_list = []

for split_num in range(1, n_splits + 1):
    # Filtra los datos por número de split
    pred_data = results_df_predictions[results_df_predictions['Split_Num'] == split_num]
    
    # Calcular MAE y MSE
    mae = mean_absolute_error(pred_data['Y_Real_Values'], pred_data['Predictions'])
    rmse = np.sqrt(mean_squared_error(pred_data['Y_Real_Values'], pred_data['Predictions']))
    mean_obs = np.mean(pred_data['Y_Real_Values'])
    cvrmse = (rmse / mean_obs) * 100  # en porcentaje
    variance = np.var(pred_data['Y_Real_Values'])
    nash = nash_sutcliffe_efficiency(pred_data['Y_Real_Values'], pred_data['Predictions'])
    willmott = willmott_index(pred_data['Y_Real_Values'], pred_data['Predictions'])
    
    

    mae_list.append(mae)
    rmse_list.append(rmse)
    cvrmse_list.append(cvrmse)
    nash_list.append(nash)
    willmott_list.append(willmott)
    
    
  
# Crear un DataFrame con los errores
errors_df = pd.DataFrame({
    'Split_Num': range(1, n_splits + 1),
    'MAE': mae_list,
    'RMSE': rmse_list,
    'CVRMSE': cvrmse_list,
    'WI': willmott_list,
    'NSE': nash_list

})




errors_df.to_csv('../acg_paper/24H_k100_XGBRegressor_errors_table.csv', index=False, float_format='%.4f') 