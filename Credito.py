# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# %%
# file_path = "/content/drive/MyDrive/application_record.csv"
# df = pd.read_csv(file_path)

df = pd.read_csv("application_record.csv")

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
df.info()


# %%
df[(df.CODE_GENDER == "M") & (df['FLAG_OWN_CAR'] == "Y") & (df['NAME_INCOME_TYPE'] == "Working") & (df['NAME_FAMILY_STATUS'] == "Married")]


# %%
df = df.astype({'CNT_FAM_MEMBERS': 'int', 'AMT_INCOME_TOTAL': 'int'})

# %%
df.dtypes

# %%
#Realizamos una Limpieza de datos, para poder manipularlos
df['CODE_GENDER'].replace('M',0,inplace=True)
df['CODE_GENDER'].replace('F',1,inplace=True)
df['FLAG_OWN_CAR'].replace('Y',0,inplace=True)
df['FLAG_OWN_CAR'].replace('N',1,inplace=True)
df['FLAG_OWN_REALTY'].replace('Y',0,inplace=True)
df['FLAG_OWN_REALTY'].replace('N',1,inplace=True)
df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].astype('category')
df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].astype('category')
df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].astype('category')
df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].astype('category')
df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].astype('category')

# %%
df.head()

# %%
df.isnull().sum()

# %%
df = df.dropna(how='any', axis=0)
df.isnull().sum()

# %%
# Filtrar solo "Commercial associate" si lo deseas
subset = df[df.NAME_INCOME_TYPE == 'Commercial associate']

# Agrupar por FLAG_EMAIL y FLAG_WORK_PHONE y calcular ingreso promedio y conteo
income_stats = subset.groupby(['FLAG_EMAIL', 'FLAG_WORK_PHONE'])['AMT_INCOME_TOTAL'].agg(['mean', 'count']).reset_index()

# Crear una columna combinada para graficar
income_stats['combo'] = income_stats['FLAG_EMAIL'].astype(str) + ' / ' + income_stats['FLAG_WORK_PHONE'].astype(str)

# Graficar ingreso promedio por combinación
plt.figure(figsize=(8, 5))
plt.bar(income_stats['combo'], income_stats['mean'], color='skyblue')

# Opcional: mostrar cantidad sobre cada barra
for i, val in enumerate(income_stats['mean']):
    plt.text(i, val + 1000, f"${val:,.0f}", ha='center', fontsize=9)

plt.title('Ingreso promedio según Email y Teléfono de trabajo')
plt.xlabel('FLAG_EMAIL / FLAG_WORK_PHONE')
plt.ylabel('Ingreso promedio (AMT_INCOME_TOTAL)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# Este grafico nos añade informacion acerca de las personas que tenemos en la base de datos, nos suministra la informacion para poder saber que tan dificies o faciles son de contactar para poder darles un credito, resultando 1/1 menos favorable

# %%
df['NAME_INCOME_TYPE'].hist()

# %%
df['NAME_FAMILY_STATUS'].hist()

# %%
object_cols = df.select_dtypes(include=['object']).columns

# Recorrer cada columna de tipo object
for col in object_cols:
    plt.figure(figsize=(8, 4))  # Opcional: tamaño del gráfico
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribución de valores en {col}')
    plt.xlabel(col)
    plt.ylabel('Conteo')
    plt.xticks(rotation=45)  # Opcional: rotar etiquetas si son largas
    plt.tight_layout()
    plt.show()

# %%
df_num = df.select_dtypes(include=["number"])
df_num

# %%
df["CNT_CHILDREN"].describe()

# %%
df["CNT_CHILDREN"].value_counts()

# %% [markdown]
# si la columna tiene 0 no hijos
# si toma valor de 1 y 2 = almenos 2 hijos
# si tiene mayor a 2  = tiene mas de 3 hijos

# %%
df = df[df['CNT_CHILDREN'].notnull()]

# Dividir la columna 'CNT_CHILDREN' en 2 segmentos: 'Tiene hijos' y 'No tiene hijos'
df['segmento_hijos'] = df['CNT_CHILDREN'].apply(lambda x: 0 if x == 0 else 1)

# Contar cuántos registros hay en cada grupo
segmentos = df['segmento_hijos'].value_counts().sort_index()

#elimina la columna hijos, y deja dos columnas simplificadas
df = df.drop(columns=['CNT_CHILDREN'])


# Graficar
segmentos.plot(kind='bar', color='lightcoral')
plt.title('Distribución por cantidad de hijos')
plt.xlabel('Segmento de hijos')
plt.ylabel('Cantidad de registros')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
df.head()

# %% [markdown]
# # Pendientes por hacer en la base de datos
# 
# 
# *   corregir datos NaN
# *   Corregir fechas de cumpleaños, y dias trabajados
# 
# 

# %%
#cantidad de personas por estado civil/familiar
df["NAME_FAMILY_STATUS"].value_counts()

# %%
df = df[df['NAME_FAMILY_STATUS'].notnull()]

# Simplificar los valores de estado civil en 2 categorías principales
df['segmento_estado_civil'] = df['NAME_FAMILY_STATUS'].apply(
    lambda x: 1 if x in ['Married', 'Civil marriage'] else 0
)

# Contar cuántos registros hay en cada grupo
segmentos = df['segmento_estado_civil'].value_counts().sort_index()

# Eliminar la columna original (opcional)
df = df.drop(columns=['NAME_FAMILY_STATUS'])

# Graficar

segmentos.plot(kind='bar', color='skyblue')
plt.title('Distribución por estado civil')
plt.xlabel('Estado civil simplificado')
plt.ylabel('Cantidad de registros')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
#Comprobar datos y no validos - ocupacion
df["OCCUPATION_TYPE"].value_counts(dropna=False)

# %%
#Mostrar todos los nulos que aparecen en los datos de las columnas
df.isnull().sum()

# %%
df['DAYS_BIRTH'] = df['DAYS_BIRTH'].abs()

# Mostrar el resultado
print(df['DAYS_BIRTH'].head())

# %%
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].abs()

# Mostrar el resultado
print(df['DAYS_EMPLOYED'].head())

# %%
df

# %% [markdown]
# 

# %%
# Usando 365.2425 días por año (más preciso)
df['AGE_YEARS'] = round(df['DAYS_BIRTH'].abs() / 365.2425, 1)


# %%
# Usando 365.2425 días por año (más preciso)
df['AGE_EMPLOYED'] = round(df['DAYS_EMPLOYED'].abs() / 365.2425, 1)

# %%
df

# %%
df

# %%
df_filtrado = df.select_dtypes(include=['int64', 'float64']).drop(columns=['ID', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'segmento_hijos', 'segmento_estado_civil', 'FLAG_EMAIL', 'FLAG_WORK_PHONE', 'FLAG_MOBIL', 'CODE_GENDER', 'DAYS_BIRTH', 'FLAG_PHONE'])
plt.figure(figsize=(25,5))
sns.heatmap(df_filtrado.select_dtypes(include='number').corr(), annot=True, cmap="viridis")

# %% [markdown]
# aqui tenemos una respuesta de las variables int, y float, para poder darles un analisis correcto, excluyendo las tipo banderas, para poder darle una mejor interpretacion al grafico, para poder, dar una mejor conclusion, es bueno incluir otras variables economicas como: el tipo de trabajo y nivel educativo, para obtener el modelado del credito con una mayor presicion

# %%
df.info() #en este caso vamos a convertir las variables categoricas a numericas, entonces procedemos con la identificacion de las variables categoricas

# %%
from sklearn.preprocessing import LabelEncoder
# Crear una instancia del codificador
le = LabelEncoder()
# Aplicar el codificador a las columnas categóricas
categorical_cols = df.select_dtypes(include=['category']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
# Verificar el resultado
df[categorical_cols].head()
df.info()

# %%
df

# %%
plt.figure(figsize=(25,5))
# Graficar la matriz de correlación
sns.heatmap(df.drop(columns=['ID', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'segmento_hijos', 'segmento_estado_civil', 'FLAG_EMAIL', 'FLAG_WORK_PHONE', 'FLAG_MOBIL', 'CODE_GENDER', 'DAYS_BIRTH', 'FLAG_PHONE', 'NAME_HOUSING_TYPE']).corr(), annot=True, cmap="viridis")

# %%
# Criterios lógicos
df['criterio_ingreso'] = df['AMT_INCOME_TOTAL'] > 120000
df['criterio_edad'] = df['AGE_YEARS'].between(25, 60)
df['criterio_empleo'] = df['DAYS_EMPLOYED'] > 1000
df['criterio_educacion'] = df['NAME_EDUCATION_TYPE'] <= 1
df['criterio_familia'] = df['CNT_FAM_MEMBERS'] <= 5

# %%
df['estado_credito'] = (
    df['criterio_ingreso'].astype(int) +
    df['criterio_edad'].astype(int) +
    df['criterio_empleo'].astype(int) +
    df['criterio_educacion'].astype(int) +
    df['criterio_familia'].astype(int)
)

# %%
df['estado_credito'] = df['estado_credito'].apply(lambda x: 1 if x >= 4 else 0)

# %%
X = df[["AMT_INCOME_TOTAL", "AGE_YEARS", "DAYS_EMPLOYED", "NAME_EDUCATION_TYPE", "CNT_FAM_MEMBERS"]]
y = df['estado_credito']

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.tree import DecisionTreeClassifier

modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
modelo.fit(x_train, y_train)

# %%
from sklearn.metrics import accuracy_score

y_pred = modelo.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# %%
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))
plot_tree(modelo, feature_names=X.columns, class_names=["No Aprobado", "Aprobado"], filled=True)
plt.title("Árbol de decisión para estado_credito")
plt.show()

# %%
import numpy as np

df['estado_credito_rnd'] = df['estado_credito']
indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
df.loc[indices, 'estado_credito_rnd'] = 1 - df.loc[indices, 'estado_credito_rnd']

# %%
x_train.shape, x_test.shape

# %%
escala = StandardScaler()
x_train = escala.fit_transform(x_train)
x_test = escala.transform(x_test)

# %%
import sklearn


RGL = sklearn.linear_model.LogisticRegression(class_weight = 'balanced')
RGL.fit(x_train, y_train)

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_pred = RGL.predict(x_test)

# %%
matrix = confusion_matrix(y_test, y_pred)
matrix

# %%
plt.figure(figsize=(10,5))
sns.heatmap(matrix, annot=True, cmap="viridis", fmt="d")
plt.title("Matriz de confusion")
plt.xlabel("Prediccion")
plt.ylabel("Real")

# %%
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# %%
y_proba = RGL.predict_proba(x_test)
y_proba

# %%
plt.hist(y_proba[:,1], bins=20)
plt.title("Distribución de probabilidades para clase 1 (credito aprovado)")
plt.xlabel("Probabilidad")
plt.ylabel("Frecuencia")
plt.show()

# %%
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(RGL, method='sigmoid')  # o 'isotonic'
calibrated_model.fit(x_train, y_train)


# %%
y_proba_calibrated = calibrated_model.predict_proba(x_test)
y_proba_calibrated

# %%
y_proba_original = RGL.predict_proba(x_test)
y_proba_original

# %%
#  probabilidades de clase 1
y_proba_original = RGL.predict_proba(x_test)[:, 1]
y_proba_calibrated = calibrated_model.predict_proba(x_test)[:, 1]

# %%
from sklearn.calibration import calibration_curve
# Curvas de calibración
prob_true_orig, prob_pred_orig = calibration_curve(y_test, y_proba_original, n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_proba_calibrated, n_bins=10)

# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(prob_pred_orig, prob_true_orig, marker='o', label='Modelo original')
plt.plot(prob_pred_cal, prob_true_cal, marker='s', label='Modelo calibrado')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfecta calibración')

plt.xlabel('Probabilidad predicha')
plt.ylabel('Frecuencia observada')
plt.title('Curva de calibración (modelo original vs calibrado)')
plt.legend()
plt.grid()

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Calcular las curvas
train_sizes, train_scores, val_scores = learning_curve(
    estimator=RGL,
    X=x_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10 valores de tamaño
    cv=cv,
    scoring='accuracy',  # puedes cambiar a 'f1', 'roc_auc', etc.
    n_jobs=-1
)

# %%
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# %%
train_scores_mean

# %%
# Gráfico
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', label="Entrenamiento", color="blue")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="blue")

plt.plot(train_sizes, val_scores_mean, 'o-', label="Validación", color="orange")
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color="orange")

plt.title("Curva de aprendizaje")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Precisión (accuracy)")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# %%
df['estado_credito'].value_counts()

# %%
df


