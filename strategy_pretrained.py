import os
import pandas as pd
from spectral_wassertein_distance import SpectralWassersteinDistance
import numpy as np


def pretrained_strategy():
    """
    Implémentation de la stratégie Pretrained multilingual
    Compare les embeddings Ngomalah obtenus après fine-tuning sur différents modèles
    """

    # Dossier contenant les embeddings pretrained
    pretrained_dir = "embeddings_pretrained"

    # Codes des modèles
    models = ['xlmr', 'mbert', 'rembert', 'camembert', 'flaubert', 'afriberta']

    # Initialiser les calculateurs de distance
    swd_r1 = SpectralWassersteinDistance(k=100, method='r1')
    swd_r_multi = SpectralWassersteinDistance(k=100, method='r_multi')

    # Créer les matrices de distances
    distance_matrix_r1 = np.zeros((len(models), len(models)))
    distance_matrix_r_multi = np.zeros((len(models), len(models)))

    print("Calcul des distances pour la stratégie Pretrained...")

    # Calculer les distances entre tous les modèles
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i <= j:  # Matrice symétrique, calculer seulement la moitié supérieure
                file1 = os.path.join(pretrained_dir, f"ngh_{model1}.vec")
                file2 = os.path.join(pretrained_dir, f"ngh_{model2}.vec")

                if os.path.exists(file1) and os.path.exists(file2):
                    print(f"Calcul distance: {model1} vs {model2}")

                    # Distance r=1
                    dist_r1 = swd_r1.compute_distance_from_files(file1, file2)
                    distance_matrix_r1[i, j] = round(dist_r1, 5)
                    distance_matrix_r1[j, i] = round(dist_r1, 5) # Symétrie

                    # Distance r>1
                    dist_r_multi = swd_r_multi.compute_distance_from_files(file1, file2)
                    distance_matrix_r_multi[i, j] = round(dist_r_multi, 5)
                    distance_matrix_r_multi[j, i] = round(dist_r_multi, 5)   # Symétrie

                    print(f"  r=1: {dist_r1:.6f}, r>1: {dist_r_multi:.6f}")
                else:
                    print(f"Fichier manquant: {file1} ou {file2}")

    # Sauvegarder les résultats en CSV
    df_r1 = pd.DataFrame(distance_matrix_r1, index=models, columns=models)
    df_r_multi = pd.DataFrame(distance_matrix_r_multi, index=models, columns=models)

    df_r1.to_csv("pretrained_distances_r1.csv")
    df_r_multi.to_csv("pretrained_distances_r_multi.csv")

    print("\nRésultats sauvegardés:")
    print("- pretrained_distances_r1.csv")
    print("- pretrained_distances_r_multi.csv")

    return df_r1, df_r_multi


if __name__ == "__main__":
    pretrained_strategy()