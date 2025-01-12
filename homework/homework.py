# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".

from sklearn import  decomposition
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import gzip
import os
from sklearn.preprocessing import MinMaxScaler
# Funcion para limpiar los datos
def clean_data(path):
    
    df=pd.read_csv(
        path,
        index_col=False,
        compression='zip')
    
    df=df.rename(columns={'default payment next month': 'default'})
    df=df.drop(columns=['ID'])
    df = df.loc[df["MARRIAGE"] != 0]
    df = df.loc[df["EDUCATION"] != 0]
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)

    return df

df_train = clean_data('files/input/train_data.csv.zip')
df_test = clean_data('files/input/test_data.csv.zip')

# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
x_train, y_train = df_train.drop(columns='default'), df_train['default']
x_test, y_test = df_test.drop(columns='default'), df_test['default']

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.

#Columnas categoricas
categorical_features=["SEX","EDUCATION","MARRIAGE"]

#preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder=StandardScaler()
)

# Creación del Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selection", SelectKBest(score_func=f_classif)),
    ("pca", PCA()),
    ("mlp", MLPClassifier(max_iter=15000,random_state=17))
])

#validation_fraction=0.3
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
param_grid = {

    'pca__n_components': [None],
    'feature_selection__k':[20],
    "mlp__hidden_layer_sizes": [(50,30,40,60)],
    'mlp__alpha': [0.26],
    "mlp__learning_rate_init": [0.001],
}

model=GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True,
    verbose=1
    )

model.fit(x_train, y_train)
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
file_path = "files/models/model.pkl.gz"
os.makedirs(os.path.dirname(file_path), exist_ok=True)

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Archivo existente eliminado: {file_path}")

# Guardar el modelo
with gzip.open(file_path, "wb") as file:
    pickle.dump(model, file)
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

with gzip.open('files/models/model.pkl.gz', 'rb') as f:
    model = pickle.load(f)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)


def metrics (y_true, y_pred, dataset):
    return {
    'type': 'metrics',
    'dataset': dataset,
    'precision': precision_score(y_true, y_pred),
    'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'f1_score': f1_score(y_true, y_pred)
    }

metrics_train = metrics(y_train, y_train_pred, 'train')
metrics_test = metrics(y_test, y_test_pred, 'test')
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
output_dir = "files/output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "metrics.json")

# Eliminar el archivo si ya existe
if os.path.exists(output_path):
    os.remove(output_path)
    print(f"Archivo existente eliminado: {output_path}")

# Crear las métricas de la matriz de confusión
def cm_matrix(cm, dataset):
    return {
        'type': 'cm_matrix',
        'dataset': dataset,
        'true_0': {"predicted_0": cm[0, 0], "predicted_1": cm[0, 1]},
        'true_1': {"predicted_0": cm[1, 0], "predicted_1": cm[1, 1]}
    }

# Calcular las matrices de confusión
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

cm_matrix_train = cm_matrix(cm_train, 'train')
cm_matrix_test = cm_matrix(cm_test, 'test')

# Guardar las métricas
metrics = [metrics_train, metrics_test, cm_matrix_train, cm_matrix_test]
print("SCRORE",model.score(x_train, y_train) )
print("SCRORE TEST",model.score(x_test, y_test) )
print(metrics)
pd.DataFrame(metrics).to_json(output_path, orient='records', lines=True)


