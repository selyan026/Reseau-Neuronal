# Dataset

Ce projet utilise le dataset **Digit Recognizer** (MNIST) de Kaggle.

Le fichier `train.csv` n'est pas versionné dans ce dépôt (trop volumineux).
Pour l'obtenir :

## Option 1 — via Kaggle
1. Créer un compte sur [kaggle.com](https://www.kaggle.com)
2. Récupérer une clé API : `Compte > Create New API Token` (télécharge `kaggle.json`)
3. Placer `kaggle.json` dans `~/.kaggle/`
4. Lancer :
```bash
pip install kaggle
kaggle competitions download -c digit-recognizer
unzip digit-recognizer.zip -d data/
```

## Option 2 — téléchargement manuel
Télécharger `train.csv` directement depuis la page de la compétition
[Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data) et le
placer dans ce dossier `data/`.

Le fichier attendu par le code est : `data/train.csv`.
