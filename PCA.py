import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

histograms_array = np.load('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/histograma_data.npy')

# Estandarizar los datos
scaler = StandardScaler()
data_std = scaler.fit_transform(histograms_array)
"""data_std = histograms_array"""

# Aplicar PCA para reducir la dimensionalidad
# Puedes elegir el número de componentes principales (e.g., 2 o 3 para visualización)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_std)
"""print(data_pca)"""
"""# Imprimir la varianza explicada por cada componente principal
print(f'Varianza explicada por cada componente principal: {pca.explained_variance_ratio_}')"""

# Ahora puedes usar data_pca para análisis adicionales o visualización
# Aquí hay un ejemplo de cómo graficar los datos transformados en 2D

df = pd.read_csv('OneDrive/Documentos/Verano2024/IntelligentAgents/archive/HAM10000_metadata.csv')
labels = df.iloc[:, 2]
dic = {'nv': 'red', 'mel': 'green', 'bkl': 'blue', 'bcc': 'yellow', 'akiec': 'orange', 'vasc': 'black', 'df':'grey'}
colors = [dic[label] for label in labels]

plt.scatter(data_pca[:, 0], data_pca[:, 1], c=colors, marker='.')
#Personaliza el 'label' para que no se genere automaticamente en base a la gráfica
lab = [
    plt.Line2D([0], [0], marker='.', color='w', markerfacecolor=color, markersize=10, label=name)
    for name, color in dic.items()
]

plt.title('Datos transformados mediante PCA')
plt.legend(handles=lab)
plt.show()