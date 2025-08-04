import os
import pandas as pd
from spectral_wassertein_distance import SpectralWassersteinDistance
import numpy as np


def absolute_crosslingual_strategy():
    """
    Implémentation de la stratégie Absolute cross-lingual
    Compare embeddings monolingue Ngomalah avec embeddings crosslingual enrichis
    """

    # Dossiers
    monolingual_dir = "embeddings_monolingual"
    crosslingual_dir = "embeddings_crosslingual"

    # Codes des langues sources et modèles
    lang_codes = ['fr', 'en', 'yem', 'ful', 'ewo', 'wol', 'ban', 'baf', 'lam']
    models = ['xlmr', 'mbert', 'rembert', 'camembert', 'flaubert', 'afriberta']

    # Initialiser les calculateurs de distance
    swd_r1 = SpectralWassersteinDistance(k=100, method='r1')
    swd_r_multi = SpectralWassersteinDistance(k=100, method='r_multi')

    # Créer les matrices de distances
    distance_matrix_r1 = np.zeros((len(models), len(lang_codes)))
    distance_matrix_r_multi = np.zeros((len(models), len(lang_codes)))

    print("Calcul des distances pour la stratégie Absolute cross-lingual...")

    # Calculer les distances
    for i, model in enumerate(models):
        # Fichier monolingue Ngomalah pour ce modèle
        mono_file = os.path.join(monolingual_dir, f"ngh_{model}.vec")

        if not os.path.exists(mono_file):
            print(f"Fichier monolingue manquant: {mono_file}")
            continue

        for j, lang_source in enumerate(lang_codes):
            # Fichier crosslingual Ngomalah enrichi par lang_source
            cross_file = os.path.join(crosslingual_dir, f"ngh_{model}_{lang_source}.vec")

            if os.path.exists(cross_file):
                print(f"Calcul distance: {model} mono vs {model}+{lang_source}")

                # Distance r=1
                dist_r1 = swd_r1.compute_distance_from_files(mono_file, cross_file)
                distance_matrix_r1[i, j] = round(dist_r1, 5)

                # Distance r>1
                dist_r_multi = swd_r_multi.compute_distance_from_files(mono_file, cross_file)
                distance_matrix_r_multi[i, j] = round(dist_r_multi, 5)

                print(f"  r=1: {dist_r1:.6f}, r>1: {dist_r_multi:.6f}")
            else:
                print(f"Fichier crosslingual manquant: {cross_file}")

    # Sauvegarder les résultats en CSV
    df_r1 = pd.DataFrame(distance_matrix_r1, index=models, columns=lang_codes)
    df_r_multi = pd.DataFrame(distance_matrix_r_multi, index=models, columns=lang_codes)

    df_r1.to_csv("absolute_distances_r1.csv")
    df_r_multi.to_csv("absolute_distances_r_multi.csv")

    print("\nRésultats sauvegardés:")
    print("- absolute_distances_r1.csv")
    print("- absolute_distances_r_multi.csv")

    return df_r1, df_r_multi


if __name__ == "__main__":
    absolute_crosslingual_strategy()