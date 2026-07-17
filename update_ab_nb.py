import json

# Actualizar HAT-PHT-AB.ipynb
nb_ab_path = '/mnt/AI-DATA-SSD/alara/CiberIA_O1_A1/CONCEPT_DRIFT/HAT-PHT-AB.ipynb'
with open(nb_ab_path, 'r', encoding='utf-8') as f:
    nb_ab = json.load(f)

save_cell_ab = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ════════════════════════════════════════════════════════════════\n",
        "# 💾  Guardar Resultados para Comparativa Posterior\n",
        "# ════════════════════════════════════════════════════════════════\n",
        "import numpy as np\n",
        "np.savez('/mnt/AI-DATA-SSD/alara/CiberIA_O1_A1/CONCEPT_DRIFT/results_ab.npz',\n",
        "         snap_x_a=snap_x_a, snap_f1_a=snap_f1_a, snap_acc_a=snap_acc_a, snap_time_a=snap_time_a,\n",
        "         snap_x_b=snap_x_b, snap_f1_b=snap_f1_b, snap_acc_b=snap_acc_b, snap_time_b=snap_time_b,\n",
        "         drift_points=drift_points,\n",
        "         N_PRETRAIN=N_PRETRAIN)\n",
        "print(\"✅ Datos de Fase A y B guardados en results_ab.npz\")\n"
    ]
}

# Insert before the last cell (which is the plot cell)
nb_ab['cells'].insert(-1, save_cell_ab)

with open(nb_ab_path, 'w', encoding='utf-8') as f:
    json.dump(nb_ab, f, indent=1, ensure_ascii=False)

print("Updated HAT-PHT-AB.ipynb")
