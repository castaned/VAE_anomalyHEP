#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tables

# === CMS Style ===
mpl.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'figure.figsize': (8, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
})

# Define la carpeta con los archivos .h5
folder_path = '/eos/user/c/castaned/NanoAOD_mixed/forEstephania4/'
h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

# Variables a usar como features y etiquetas
features = [
    'D_Dr_Z', 'D_Lep1Z_pt', 'D_Lep2Z_pt', 'D_Lep1Z_eta', 'D_Lep2Z_eta',
    'D_Lep1Z_phi', 'D_Lep2Z_phi', 'D_Lep3W_pt', 'D_Lep3W_eta', 'D_Lep3W_phi',
    'D_Sum_mass', 'D_Sum_pt', 'D_Wmass', 'D_Zmass', 'D_nlep', 'MET_pt','MET_phi',
    'D_ptZ','D_mu1ip3d','D_mu2ip3d','D_mu3ip3d'
]
labels = ['Dataset_ID']
nfeatures = len(features)

# === Ranges and bins for selected variables ===
binning_settings = {
    'D_Zmass': {'range': (0, 200), 'bins': 60},
    'D_Sum_mass': {'range': (0, 1000), 'bins': 60},
    'D_Sum_pt': {'range': (0, 1000), 'bins': 60},
    'D_Lep1Z_pt': {'range': (0, 1000), 'bins': 60},
    'D_ptZ': {'range': (0, 600), 'bins': 60},
    'D_mu1ip3d': {'range': (0, 0.1), 'bins': 60},
    'D_mu2ip3d': {'range': (0, 0.1), 'bins': 60},
    'D_mu3ip3d': {'range': (0, 0.1), 'bins': 60},
}

# === Cargar datos desde archivos .h5 ===
def get_all_features_labels(folder_path):
    feature_list = []
    dataset_id_list = []

    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

    for file_name in h5_files:
        file_path = os.path.join(folder_path, file_name)
        with tables.open_file(file_path, 'r') as h5file:
            nevents = getattr(h5file.root, features[0]).shape[0]
            feature_array = np.zeros((nevents, nfeatures))

            for i, feat in enumerate(features):
                feature_array[:, i] = getattr(h5file.root, feat)[:]

            dataset_id_array = getattr(h5file.root, 'Dataset_ID')[:]

        feature_list.append(feature_array)
        dataset_id_list.append(dataset_id_array)

    all_features = np.vstack(feature_list)
    all_dataset_ids = np.concatenate(dataset_id_list)
    return all_features, all_dataset_ids

# === Ejecutar carga ===
feature_array, dataset_id_array = get_all_features_labels(folder_path)

print("Datos cargados:")
print("Features:", feature_array.shape)
print("Dataset_IDs:", dataset_id_array.shape)

# === Crear carpeta para guardar los plots ===
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# === Limpiar plots anteriores ===
for f in os.listdir(plot_dir):
    if f.endswith(".png"):
        os.remove(os.path.join(plot_dir, f))

# === Crear DataFrame y separar por clase ===
df = pd.DataFrame(feature_array, columns=features)
df['Dataset_ID'] = dataset_id_array
df_1 = df[df['Dataset_ID'] == 1]
df_3 = df[df['Dataset_ID'] == 3]

print(f"Eventos con Dataset_ID == 1: {len(df_1)}")
print(f"Eventos con Dataset_ID == 3: {len(df_3)}")

# === Generar histogramas estilo CMS tipo TH1F ===
for var in features:
    print(f"Plotting variable: {var}")
    settings = binning_settings.get(var, {})
    bins = settings.get('bins', 60)
    range_ = settings.get('range', (min(df[var]), max(df[var])))

    fig, ax = plt.subplots()
    histtype = 'stepfilled'  # similar a TH1F relleno
    ax.hist(df_1[var], bins=bins, range=range_, alpha=0.6, label='Dataset_ID = 1', color='navy', density=True, histtype=histtype)
    ax.hist(df_3[var], bins=bins, range=range_, alpha=0.6, label='Dataset_ID = 3', color='crimson', density=True, histtype=histtype)

    ax.set_xlabel(var)
    ax.set_ylabel("Events (norm.)")
    ax.legend(loc='best')

    # Etiqueta estilo CMS (ligeramente más baja)
    fig.text(0.11, 0.89, 'CMS Simulation', fontsize=16, fontweight='bold', ha='left', va='bottom')

    # Guardar PNG y PDF, asegurando sobrescritura
    output_base = os.path.join(plot_dir, f"{var}_distribution")
    for ext in [".png", ".pdf"]:
        if os.path.exists(output_base + ext):
            os.remove(output_base + ext)
        plt.savefig(output_base + ext)
        print(f"Saved {output_base + ext}")

    plt.close()

print(f"Se guardaron las distribuciones en la carpeta '{plot_dir}'")


# Calculate the correlation matrix
corr = df[features].corr()



plt.figure(figsize=(14, 12))  # Bigger canvas

sns.heatmap(
    df[features].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    annot_kws={"size": 8},       # smaller font for correlation numbers
    cbar_kws={"shrink": 0.8},
    linewidths=0.5,
    linecolor='gray',
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title("Feature Correlation Matrix", fontsize=18)
plt.tight_layout()
plt.savefig("plots/correlation_matrix_full.png")
plt.close()

