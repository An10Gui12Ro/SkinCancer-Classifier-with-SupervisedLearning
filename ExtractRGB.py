import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('OneDrive/Documentos/Verano2024/IntelligentAgents/archive/HAM10000_metadata.csv')

contenido_primera_columna = df.iloc[:, 1]
i = 0
for elemento in contenido_primera_columna:
    direction = 'OneDrive/Documentos/Verano2024/IntelligentAgents/archive/all_images/{0}.jpg'.format(elemento)
    img = plt.imread(direction)

    # Definir los colores para cada canal
    colors = ("red", "green", "blue")

    # Crear la figura y los ejes para el histograma
    fig, ax = plt.subplots()
    ax.set_xlim([0, 256])

    # Calcular y trazar el histograma para cada canal de color
    # La función 'enumerate' enumera los colores, por lo tanto, 'channel_id' es el idx del color y 'color' es el color
    print(i)
    for channel_id, color in enumerate(colors):
        # 'histogram' contiene el conteo de píxeles para cada bin(es decir, cuántos píxeles hay en cada rango en los que se dividió)
        # 'bins' son intervalos de 8 para agrupar datos
        # 'img[:, :, channel_id]' accede a todas las filas y columnas y se enfoca en los píxeles que se encuentren en ese canal
        # 'range' es para indicar los límites de los colores RGB y no se salgan
        # 'bin_edges' contiene estos datos: [0, 32, 64, 96, 128, 160, 192, 224, 256]
        histogram, bin_edges = np.histogram(img[:, :, channel_id], bins=8, range=(0, 256))
        # 'ax.plot' sirve para gráficar, en este caso grafica 'histogram' en base a 'bin_edges' (se le pone '[:-1]' creo porque son 
        # 9 elementos en 'bin_edges' [0, 32, 64, 96, 128, 160, 192, 224, 256] pero 8 rangos [0-31, 32-63, 64-95, 96-127, 128-159, 160-191, 192-223, 224-255])
        # entonces no se toma el último para que no ocurra error
        # 'color' es para específicar de qué color se hará la gráfica y 'label' es para agregar una etiqueta
        ax.plot(bin_edges[:-1], histogram, color=color, label=f'{color.capitalize()} channel')

        print(color, ":", histogram)

    # Configurar el título y las etiquetas del histograma
    ax.set_title("Color Histogram")
    ax.set_xlabel("Color value")
    ax.set_ylabel("Pixel count")
    ax.legend()

    # Mostrar el histograma
    if i <= 5:
        plt.show()
    i += 1

    plt.close(fig)