# BeeCounter - outil de labellisation

Interface graphique pour estimer le nombre d'abeilles sur les 4 images (une par
caméra) d'une ruche à un instant donné, par échantillonnage de densité.

## Installation

L'environnement conda `imaging` (déjà présent sur cette machine, `opencv-python`,
`pandas`, `numpy`, `matplotlib`, `tqdm`) contient tout ce qu'il faut :

```bash
/opt/miniconda3/envs/imaging/bin/python label_app.py
```

Ou, une fois l'environnement activé :

```bash
conda activate imaging
python label_app.py
```

## Utilisation

1. Vérifiez le "Dossier images" (par défaut le dossier de la campagne
   `24.11-25.01_metabolism_OH`), choisissez la ruche et un timestamp
   (heure UTC), puis cliquez sur **Charger**. L'outil va
   chercher l'image la plus proche pour chacune des 4 caméras (recherche
   exacte à la minute, puis recherche de l'image la plus proche si besoin).
2. Pour chaque caméra (onglets en haut) :
   - Mode **Dessiner la zone** (optionnel) : tracez au clic-glissé un
     rectangle sur une zone représentative de la densité d'abeilles. Sert
     uniquement à calculer la suggestion de facteur ; "Supprimer la zone"
     l'efface si besoin, et vous pouvez valider une caméra sans en avoir
     dessiné une.
   - Mode **Compter les abeilles** : cliquez sur chaque abeille pour la
     compter (une croix rouge est ajoutée), à n'importe quel niveau de zoom.
     "Annuler dernier point" et "Effacer tous les points" permettent de
     corriger. Clic droit maintenu pour déplacer la vue, molette pour zoomer.
   - Mode **Naviguer** : utilise la barre d'outils matplotlib (zoom/pan) pour
     s'approcher et compter plus précisément.
   - Une suggestion de facteur (aire totale de l'image / aire de la zone)
     est affichée à titre indicatif — c'est vous qui décidez du facteur de
     multiplication à utiliser (champ éditable ; "Copier la suggestion" le
     pré-remplit si vous le souhaitez).
   - Cliquez sur **Valider cette caméra** une fois la zone, le comptage et le
     facteur définis.
3. Une fois les 4 caméras validées (ou moins, avec confirmation), cliquez sur
   **Enregistrer la session (CSV)**.

## Résultats

Les résultats sont ajoutés (append) dans `results/bee_counts.csv`, une ligne
par caméra labellisée. Pour obtenir le total par ruche/instant :

```python
import pandas as pd
df = pd.read_csv("results/bee_counts.csv")
totals = df.groupby("session_id").agg(
    hive=("hive", "first"),
    timestamp=("requested_timestamp", "first"),
    n_cameras=("camera", "count"),
    total_bees=("estimated_total_bees", "sum"),
)
```

## Fichiers

- `label_app.py` : application Tkinter + Matplotlib (point d'entrée).
- `image_source.py` : localisation des 4 images pour une ruche/timestamp
  donnés (réutilise `fetchImagesPaths` de `../libimage.py`).
- `results_store.py` : écriture des résultats dans le CSV.
