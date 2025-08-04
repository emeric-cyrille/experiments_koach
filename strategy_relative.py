import os
import pandas as pd
from spectral_wassertein_distance import SpectralWassersteinDistance
import numpy as np


def relative_crosslingual_strategy():
    """
    Implémentation de la stratégie Relative cross-lingual
    Compare embeddings Ngomalah enrichis par différentes langues sources
    Génère un fichier par modèle (6 modèles x 2 méthodes = 12 fichiers)
    """

    # Dossier
    crosslingual_dir = "embeddings_crosslingual"

    # Codes des langues sources et modèles
    lang_codes = ['fr', 'en', 'yem', 'ful', 'ewo', 'wol', 'ban', 'baf', 'lam']
    models = ['xlmr', 'mbert', 'rembert', 'camembert', 'flaubert', 'afriberta']

    # Initialiser les calculateurs de distance
    swd_r1 = SpectralWassersteinDistance(k=100, method='r1')
    swd_r_multi = SpectralWassersteinDistance(k=100, method='r_multi')

    print("Calcul des distances pour la stratégie Relative cross-lingual...")

    # Pour chaque modèle, créer une matrice de distances entre langues sources
    for model in models:
        print(f"\nTraitement du modèle: {model}")

        # Créer les matrices de distances pour ce modèle
        distance_matrix_r1 = np.zeros((len(lang_codes), len(lang_codes)))
        distance_matrix_r_multi = np.zeros((len(lang_codes), len(lang_codes)))

        # Calculer les distances entre toutes les paires de langues sources
        for i, lang1 in enumerate(lang_codes):
            for j, lang2 in enumerate(lang_codes):
                if i <= j:  # Matrice symétrique
                    file1 = os.path.join(crosslingual_dir, f"ngh_{model}_{lang1}.vec")
                    file2 = os.path.join(crosslingual_dir, f"ngh_{model}_{lang2}.vec")

                    if os.path.exists(file1) and os.path.exists(file2):
                        print(f"  Calcul distance: {lang1} vs {lang2}")

                        # Distance r=1
                        dist_r1 = swd_r1.compute_distance_from_files(file1, file2)
                        distance_matrix_r1[i, j] = round(dist_r1, 5)
                        distance_matrix_r1[j, i] = round(dist_r1, 5)  # Symétrie

                        # Distance r>1
                        dist_r_multi = swd_r_multi.compute_distance_from_files(file1, file2)
                        distance_matrix_r_multi[i, j] = round(dist_r_multi, 5)
                        distance_matrix_r_multi[j, i] = round(dist_r_multi, 5)  # Symétrie

                        print(f"    r=1: {dist_r1:.6f}, r>1: {dist_r_multi:.6f}")
                    else:
                        print(f"    Fichier manquant: {file1} ou {file2}")

        # Sauvegarder les résultats pour ce modèle
        df_r1 = pd.DataFrame(distance_matrix_r1, index=lang_codes, columns=lang_codes)
        df_r_multi = pd.DataFrame(distance_matrix_r_multi, index=lang_codes, columns=lang_codes)

        df_r1.to_csv(f"relative_distances_{model}_r1.csv")
        df_r_multi.to_csv(f"relative_distances_{model}_r_multi.csv")

        print(f"  Sauvegardé: relative_distances_{model}_r1.csv")
        print(f"  Sauvegardé: relative_distances_{model}_r_multi.csv")

    print(f"\nStratégie Relative terminée. {len(models) * 2} fichiers générés.")


if __name__ == "__main__":
    relative_crosslingual_strategy()