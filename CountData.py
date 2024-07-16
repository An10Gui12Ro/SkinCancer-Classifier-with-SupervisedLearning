import pandas as pd
import matplotlib.pyplot as plt

# Leer los datos del archivo CSV
df = pd.read_csv('OneDrive/Documentos/Verano2024/IntelligentAgents/archive/HAM10000_metadata.csv')

# Contar el número de valores únicos en cada columna
dx_counts = df["dx"].value_counts()
dx_type_counts = df["dx_type"].value_counts()
sex_counts = df["sex"].value_counts()

# Categorizar las edades en rangos
bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

# Contar la frecuencia de cada rango de edad
age_group_counts = df['age_group'].value_counts().sort_index()

# Contar el número de valores únicos en la columna de 'localization'
localization_counts = df["localization"].value_counts()

# Imprimir los resultados
print("Unique 'dx' values:")
print(dx_counts)

print("\nUnique 'dx_type' values:")
print(dx_type_counts)

print("\nAge group counts:")
print(age_group_counts)

print("\nSex counts:")
print(sex_counts)

print("\nLocalization")
print(localization_counts)

# Crear un gráfico de pastel para 'dx'
plt.figure(figsize=(8, 6))
plt.pie(dx_counts.values, labels=dx_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of 'dx'")
plt.show()

# Crear un gráfico de pastel para 'dx_type'
plt.figure(figsize=(8, 6))
plt.pie(dx_type_counts.values, labels=dx_type_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of 'dx_type'")
plt.show()

# Crear un gráfico de pastel para los 'age'
plt.figure(figsize=(8, 6))
plt.pie(age_group_counts.values, labels=age_group_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of Age Groups")
plt.show()

# Crear un gráfico de pastel para 'sex'
plt.figure(figsize=(8, 6))
plt.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of 'sex'")
plt.show()

# Crear un gráfico de pastel para 'localization'
plt.figure(figsize=(8, 6))
plt.pie(localization_counts.values, labels=localization_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of 'localization'")
plt.show()

#Excel / vs / canva / chat