import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd

df = pd.read_csv('OneDrive/Documentos/Verano2024/IntelligentAgents/archive/HAM10000_metadata.csv')
image_id = df.iloc[:, 1]
dx = df.iloc[:, 2]

# Ejemplo de datos de imágenes y etiquetas (sustituye con tus propios datos)
X = np.array(image_id)  # Matriz de características, por ejemplo, las imágenes
y = np.array(dx)  # Vector de etiquetas, por ejemplo, tipos de cáncer de piel

#Primero separo test y train para después hacer lo de 'StratifiedKFold' para train
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_index, test_index = next(iter(kf.split(X, y)))
#Ahora 'X' y 'y' serán sobre las nuevas imagenes de train
test_idx_vec = np.array(X[test_index])
test_dx_idx_vec = np.array(y[test_index])
np.save('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/test_dx.npy', test_dx_idx_vec)
np.save('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/test_images_id.npy', test_idx_vec)
print(test_idx_vec)
X = np.array(X[train_index])
y = np.array(y[train_index])

# Inicialización de Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_indices = []
train_dx_indices = []
test_indices = []
test_dx_indices = []

# Iteración sobre los pliegues
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]  # Datos de entrenamiento y prueba
    y_train, y_test = y[train_index], y[test_index]  # Etiquetas de entrenamiento y prueba
    print(f"  Train {i+1}: index={train_index}")
    print(f"  Test {i+1}:  index={test_index}")
    print(f"  Train {i+1}: image_id={X[train_index]}")
    print(f"  Test {i+1}:  image_id={X[test_index]}")
    print(f"  Train {i+1}: dx={y[train_index]}")
    print(f"  Test {i+1}:  dx={y[test_index]}")
    print()
    print(len(train_index), " y ", len(test_index))
    train_indices.append(X[train_index])
    train_dx_indices.append(y[train_index])
    test_indices.append(X[test_index])
    test_dx_indices.append(y[test_index])
    # Aquí puedes usar X_train, X_test, y_train, y_test para entrenar y evaluar tu modelo
    # Por ejemplo:
    # model.fit(X_train, y_train)
    # score = model.score(X_test, y_test)
    
    # Puedes realizar cualquier operación de modelado aquí dentro del bucle de validación cruzada
max_train_len = max(len(indices) for indices in train_indices)
max_test_len = max(len(indices) for indices in test_indices)

max_train_dx_len = max(len(indices) for indices in train_dx_indices)
max_test_dx_len = max(len(indices) for indices in test_dx_indices)

def pad_indices(indices, max_len):
    return np.pad(indices, (0, max_len - len(indices)), 'constant', constant_values=-1)

# Modificar cada vector dentro de las listas
train_indices = [pad_indices(indices, max_train_len) for indices in train_indices]
test_indices = [pad_indices(indices, max_test_len) for indices in test_indices]

train_dx_indices = [pad_indices(indices, max_train_dx_len) for indices in train_dx_indices]
test_dx_indices = [pad_indices(indices, max_test_dx_len) for indices in test_dx_indices]

train_idx_mat = np.array(train_indices)
test_idx_mat = np.array(test_indices)
train_dx_idx_mat = np.array(train_dx_indices)
test_dx_idx_mat = np.array(test_dx_indices)

print(train_idx_mat)
print()
print(test_idx_mat)

np.save('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/train_images_id.npy', train_idx_mat)
np.save('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/validation_images_id.npy', test_idx_mat)
np.save('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/train_dx.npy', train_dx_idx_mat)
np.save('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/validation_dx.npy', test_dx_idx_mat)