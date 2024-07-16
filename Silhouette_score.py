"""from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
import pandas as pd

histograms = np.load('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/histograma_data.npy')
X = histograms

df = pd.read_csv('OneDrive/Documentos/Verano2024/IntelligentAgents/archive/HAM10000_metadata.csv')
labels = df.iloc[:, 2]

""# Supongamos que tenemos un conjunto de datos X y queremos agruparlo en 7 clusters
kmeans = KMeans(n_clusters=7)
labels = kmeans.fit_predict(X)""

# Calculamos la puntuación de silueta
score = silhouette_score(X, labels)
print(f'Silhouette Score: {score}')

# Reducir las dimensiones a 2D usando PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Crear un gráfico de dispersión para cada cluster
plt.figure(figsize=(10, 6))

# Lista de colores para los clusters
colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink']

# Graficar cada cluster
for cluster in range(7):
    cluster_points = X_reduced[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=20, c=colors[cluster], label=cluster)

# Graficar los centros de los clusters
centers = kmeans.cluster_centers_
centers_reduced = pca.transform(centers)
plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], s=200, c='yellow', marker='X', edgecolor='black', linewidth=2, label='Centros')

# Añadir etiquetas y título
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clusters y sus Centros')
plt.legend()
plt.show()


# Crear el visualizador de silhouette
visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
# Ajustar el visualizador
visualizer.fit(X)
# Visualizar el silhouette plot
visualizer.show()"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

histograms = np.load('OneDrive/Documentos/Verano2024/IntelligentAgents/Proyect/histograma_data.npy')
X = histograms

df = pd.read_csv('OneDrive/Documentos/Verano2024/IntelligentAgents/archive/HAM10000_metadata.csv')
labels = df.iloc[:, 2]


# Convert string labels to numerical labels
le = LabelEncoder()
numeric_labels = le.fit_transform(labels)


# Get the number of unique labels
n_clusters = len(np.unique(labels))

# Create a subplot with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)

# The 1st subplot is the silhouette plot
ax1.set_xlim([-0.5, .5])
ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

# The silhouette_score gives the average value for all the samples.
silhouette_avg = silhouette_score(X, numeric_labels)
print(
    "For n_clusters =",
    n_clusters,
    "The average silhouette_score is :",
    silhouette_avg,
)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X, numeric_labels)

y_lower = 10
for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[numeric_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = plt.cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(le.inverse_transform([i])[0]))
    y_lower = y_upper + 10

ax1.set_title("The silhouette plot for the various classes.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Class label")
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
ax1.set_yticks([])
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# 2nd Plot showing the actual data points
colors = plt.cm.nipy_spectral(numeric_labels.astype(float) / n_clusters)
scatter = ax2.scatter(
    X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
)

ax2.set_title("The visualization of the labeled data.")
ax2.set_xlabel("First PCA component")
ax2.set_ylabel("Second PCA component")

# Add a legend
legend_labels = le.classes_
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.nipy_spectral(i / n_clusters), 
                      markersize=10, label=label) for i, label in enumerate(legend_labels)]
ax2.legend(handles=handles, title="Classes", loc="center left", bbox_to_anchor=(1, 0.5))

plt.suptitle(
    "Silhouette analysis for labeled data with n_classes = %d" % n_clusters,
    fontsize=14,
    fontweight="bold",
)

plt.tight_layout()
plt.show()