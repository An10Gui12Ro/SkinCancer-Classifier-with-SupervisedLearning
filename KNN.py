from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

# Suponiendo que tienes X (matriz de características) y y (vector de etiquetas) listos
df = pd.read_csv('OneDrive/Documentos/Verano2024/IntelligentAgents/archive/HAM10000_metadata.csv')
image_id = df.iloc[:, 1]
histograms = np.load('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/histograma_data.npy')
dic = {}

for i, id in enumerate(image_id):
    dic[id] = histograms[i]

train_images_id = np.load('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/train_images_id.npy', allow_pickle=True)
train_dx = np.load('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/train_dx.npy', allow_pickle=True)
validation_images_id = np.load('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/validation_images_id.npy', allow_pickle=True)
validation_dx = np.load('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/validation_dx.npy', allow_pickle=True)
test_images_id = np.load('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/test_images_id.npy', allow_pickle=True)
test_dx = np.load('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/test_dx.npy', allow_pickle=True)
"""print(train_images_id)
print()
print(train_dx)
print()
print(validation_images_id)
print()
print(validation_dx)
print()
print(train_images_id[0])
print(test_images_id)
print(test_dx)"""

max_accuracy = 0

#Se crea un 'for' para los 5 splits
for i in range(5):
    X_train = []
    y_train = []
    for img in train_images_id[i]:
        if img != -1:
            X_train.append(dic[img])
    X_train = np.array(X_train)
    """print(X_train, len(X_train))"""
    for dx in train_dx[i]:
        if dx != -1:
            y_train.append(dx)
    y_train = np.array(y_train)
    """print(y_train, len(y_train))"""

    X_test = []
    y_test = []
    for img in validation_images_id[i]:
        if img != -1:
            X_test.append(dic[img])
    X_test = np.array(X_test)
    """print(X_test, len(X_test))"""
    for dx in validation_dx[i]:
        if dx != -1:
            y_test.append(dx)
    y_test = np.array(y_test)
    """print(y_test, len(y_test))"""

    # Paso 2: Crear una instancia del modelo (por ejemplo, con k=3)
    knn = KNeighborsClassifier(n_neighbors=3)

    # Paso 3: Entrenar el modelo
    knn.fit(X_train, y_train)

    # Paso 4: Hacer predicciones
    y_pred = knn.predict(X_test)

    """for i in range(len(y_pred)):
        print(y_pred[i], "-->", y_test[i])"""

    # Paso 5: Evaluar el rendimiento del modelo
    print("Validation:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'F1 score: {f1}')

    # Predicción y rendimiento del modelo con los test

    X_test = []
    for img in test_images_id:
        X_test.append(dic[img])
    X_test = np.array(X_test)

    y_pred = knn.predict(X_test)
    print("Test:")
    accuracy = accuracy_score(test_dx, y_pred)
    print(f'Accuracy: {accuracy}')
    f1 = f1_score(test_dx, y_pred, average='weighted')
    print(f'F1 score: {f1}')


    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_pred = y_pred

# Matriz de confusión para el conjunto de prueba
# La matriz sirve para saber cómo fueron clasificados cada uno de los elementos de ese tipo de cancer, en este caso
cm = confusion_matrix(test_dx, best_pred)
# Normalizar la matriz de confusión
# Cada elemento de la matriz se cambia a 'float' y se divide entre la suma de los elementos de esa fila (para saber el porcentaje)
# 'axis=1' se utiliza para referirse a las filas y 'axis=0' a las columnas
# [:, np.newaxis] Convierte el array de sumas de filas en una columna permitiendo que cada elemento en la matriz de confusión sea 
# dividido por el total de su fila correspondiente.
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# Se le pasa como parámetro la matriz de confusión normalizada para que la pueda graficar, 'knn.classes_' contiene las etiquetas de 
# clase aprendidas por el modelo knn durante el entrenamiento y esas son las que se ponene en los ejes
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=knn.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Normalized Confusion Matrix")
plt.show()

"""suma = 0
for type in test_dx:
    if type == 'bkl':
        suma += 1
print(suma)"""