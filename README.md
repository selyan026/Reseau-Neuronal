# Réseau neuronal — Reconnaissance de chiffres manuscrits

Implémentation **from scratch** (NumPy uniquement, aucun framework de deep
learning type PyTorch/TensorFlow) d'un réseau de neurones pour classifier
les chiffres manuscrits du dataset MNIST.

## Pourquoi ce projet

Je travaille dans les systèmes/réseaux, en évoluant vers la cybersécurité
(alternance en cybersécurité à partir de septembre 2026). Or une bonne
partie des outils que je manipule ou dont j'entends parler au quotidien —
EDR, SIEM, détection d'anomalies — s'appuient de plus en plus sur du machine
learning pour repérer des comportements suspects. Je me servais de ces
outils comme des boîtes noires, sans vraiment comprendre ce qu'il y avait
derrière le mot "IA" qu'on met un peu partout.

Ce projet est parti de cette frustration : plutôt que d'utiliser une
librairie haut niveau (scikit-learn, Keras...) qui masque tout le
fonctionnement interne, j'ai voulu **réimplémenter un réseau de neurones à
la main**, sans aucun framework de deep learning, pour comprendre
concrètement ce qui se passe entre l'entrée (une image) et la sortie (une
prédiction) : d'où viennent les poids, comment ils sont corrigés, pourquoi
on utilise telle fonction plutôt qu'une autre.

> Pour une explication détaillée et pédagogique du fonctionnement de
> l'entraînement (forward propagation, rétropropagation, descente de
> gradient...), voir [`ENTRAINEMENT.md`](ENTRAINEMENT.md).

## Ce que ce projet m'a apporté

- **NumPy en profondeur** : manipuler des matrices, vectoriser des calculs
  (éviter les boucles Python explicites sur des milliers d'images),
  comprendre le broadcasting — des réflexes utiles bien au-delà du ML,
  y compris pour du traitement de données ou des scripts d'automatisation.
- **Pandas** : chargement, nettoyage et mise en forme d'un dataset réel
  (lecture CSV, conversion en tableaux NumPy, séparation
  entraînement/test).
- **Les bases mathématiques du machine learning** : algèbre linéaire
  (produits matriciels, transposées), calcul différentiel (dérivées
  partielles, règle de la chaîne appliquée à la rétropropagation), et leur
  traduction directe en code.
- **Une meilleure lecture des outils de sécurité basés sur l'IA** :
  comprendre comment un modèle "apprend" à partir de données rend beaucoup
  plus lisibles les discussions autour des faux positifs, du besoin de
  données d'entraînement représentatives, ou des limites d'un modèle de
  détection.
- **De bonnes pratiques de code**, au-delà du script qui marche une fois :
  architecture orientée objet, principes SOLID, tests unitaires, gestion de
  version avec Git — pour repartir de ce projet plus tard sans devoir tout
  relire.

## Résultats

| Configuration | Précision (test) |
|---|---|
| 1 couche cachée, 500 itérations, alpha fixe | ~83–86% |
| 1 couche cachée, 1000 itérations, alpha fixe | ~86–88% |
| 1 couche cachée, 500 itérations, alpha adaptatif (1 → 0.1) | **90.7%** |

## Architecture du code

Le code a été structuré autour de quelques principes SOLID et design
patterns, pour rester lisible et facilement extensible :

```
src/
├── activations.py    # ReLU, Softmax (Strategy Pattern)
├── layers.py          # DenseLayer : une couche entièrement connectée
├── losses.py          # CategoricalCrossEntropy
├── optimizers.py       # SGD, StepDecayOptimizer (Strategy + Decorator Pattern)
├── model.py            # NeuralNetwork : orchestre couches/loss/optimizer
├── data_loader.py      # Chargement et préparation du dataset MNIST
└── visualization.py     # Courbes de précision, affichage de prédictions
```

**Pourquoi cette architecture ?**

- **Single Responsibility** : chaque classe a un seul rôle. `DenseLayer` ne
  connaît que ses poids, `MnistDataLoader` ne fait que charger les données,
  `NeuralNetwork` orchestre sans jamais manipuler directement des poids.
- **Open/Closed** : ajouter une nouvelle fonction d'activation (Sigmoid,
  Tanh...) ou un nouvel optimiseur ne demande de modifier aucune classe
  existante — seulement d'en créer une nouvelle qui respecte l'interface
  `Activation` ou `Optimizer`.
- **Dependency Inversion** : `NeuralNetwork` dépend d'abstractions
  (`Activation`, `Optimizer`, la fonction de coût), jamais d'une
  implémentation concrète codée en dur.
- **Strategy Pattern** : `ReLU`/`Softmax` d'un côté, `SGD` de l'autre, sont
  interchangeables sans changer le reste du code.
- **Decorator Pattern** : `StepDecayOptimizer` enrichit un `SGD` existant
  (réduction du taux d'apprentissage après N itérations) sans le modifier
  ni le dupliquer. C'est la stratégie "alpha=1 puis alpha=0.1" qui a donné
  le meilleur résultat expérimental (90.7%).

## Installation

```bash
git clone https://github.com/selyan026/Reseau-Neuronal.git
cd Reseau-Neuronal
python3 -m venv venv
source venv/bin/activate   # sous Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Récupérer ensuite le dataset MNIST — voir [`data/README.md`](data/README.md).

## Utilisation

```bash
python main.py
```

Cela entraîne le modèle sur 500 itérations, affiche la courbe de précision,
la précision finale sur les données de test, et un exemple de prédiction.

Les hyperparamètres (nombre d'itérations, alpha, moment du "decay"...) sont
configurables en haut de `main.py`.

## Tests

```bash
pytest tests/ -v
```

## Structure complète du dépôt

```
reseau-neuronal-mnist/
├── README.md
├── ENTRAINEMENT.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
│   └── README.md
├── src/
│   ├── activations.py
│   ├── layers.py
│   ├── losses.py
│   ├── optimizers.py
│   ├── model.py
│   ├── data_loader.py
│   └── visualization.py
└── tests/
    ├── test_activations.py
    └── test_losses.py
```

## Licence

MIT — voir [`LICENSE`](LICENSE).
