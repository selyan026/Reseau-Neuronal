# Réseau neuronal : reconnaissance de chiffres manuscrits

Réseau de neurones implémenté entièrement en NumPy, sans framework de deep
learning (pas de PyTorch, TensorFlow ou Keras). Le modèle apprend à
reconnaître des chiffres manuscrits (0 à 9) à partir d'images 28x28 pixels
issues du dataset MNIST, avec une précision finale de 90.7% sur les données
de test.

Toute la mécanique d'apprentissage (propagation avant, rétropropagation,
descente de gradient) est codée à la main, sans bibliothèque de machine
learning haut niveau. L'objectif était de comprendre en détail le
fonctionnement interne d'un réseau de neurones plutôt que d'utiliser une
implémentation existante.

Pour une explication pas à pas du fonctionnement de l'entraînement (calcul
des gradients, rôle de chaque fonction, intuition mathématique), voir
[ENTRAINEMENT.md](ENTRAINEMENT.md).

## Résultats

| Configuration | Précision (test) |
|---|---|
| 1 couche cachée, 500 itérations, taux d'apprentissage fixe | 83 à 86% |
| 1 couche cachée, 1000 itérations, taux d'apprentissage fixe | 86 à 88% |
| 1 couche cachée, 500 itérations, taux d'apprentissage adaptatif (1 puis 0.1) | 90.7% |

## Ce que ce projet a permis d'approfondir

* **NumPy** : manipulation matricielle, vectorisation des calculs pour
  éviter les boucles Python sur des milliers d'images, broadcasting.
* **Pandas** : chargement et préparation d'un dataset réel (lecture CSV,
  conversion en tableaux NumPy, séparation entraînement/test).
* **Les fondements mathématiques du machine learning** : algèbre linéaire
  (produits matriciels, transposées) et calcul différentiel (dérivées
  partielles, règle de la chaîne appliquée à la rétropropagation),
  traduits directement en code.
* **Des pratiques de développement structurées** : architecture orientée
  objet, principes SOLID, tests unitaires, gestion de version avec Git.

## Architecture du code

Le code est organisé autour de quelques principes SOLID et design patterns
pour rester lisible et facilement extensible.

```
src/
├── activations.py    # ReLU, Softmax (Strategy Pattern)
├── layers.py          # DenseLayer : une couche entièrement connectée
├── losses.py          # CategoricalCrossEntropy
├── optimizers.py       # SGD, StepDecayOptimizer (Strategy + Decorator Pattern)
├── model.py            # NeuralNetwork : orchestre couches, coût et optimiseur
├── data_loader.py      # Chargement et préparation du dataset MNIST
└── visualization.py     # Courbes de précision, affichage de prédictions
```

**Principes appliqués**

* **Responsabilité unique** : chaque classe a un seul rôle. `DenseLayer` ne
  connaît que ses propres poids, `MnistDataLoader` ne fait que charger les
  données, `NeuralNetwork` orchestre l'ensemble sans manipuler directement
  les poids.
* **Ouvert/fermé** : ajouter une nouvelle fonction d'activation (Sigmoid,
  Tanh) ou un nouvel optimiseur ne nécessite de modifier aucune classe
  existante, seulement d'en créer une nouvelle respectant l'interface
  `Activation` ou `Optimizer`.
* **Inversion des dépendances** : `NeuralNetwork` dépend d'abstractions
  (`Activation`, `Optimizer`, fonction de coût), jamais d'une implémentation
  concrète codée en dur.
* **Strategy Pattern** : `ReLU`/`Softmax` d'un côté, `SGD` de l'autre, sont
  interchangeables sans modifier le reste du code.
* **Decorator Pattern** : `StepDecayOptimizer` enrichit un `SGD` existant
  (réduction du taux d'apprentissage après un certain nombre d'itérations)
  sans le modifier ni le dupliquer. C'est cette stratégie qui a donné le
  meilleur résultat expérimental (90.7%).

## Installation

```bash
git clone https://github.com/selyan026/Reseau-Neuronal.git
cd Reseau-Neuronal
python3 -m venv venv
source venv/bin/activate   # sous Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Le dataset MNIST doit être récupéré séparément, voir
[data/README.md](data/README.md).

## Utilisation

```bash
python main.py
```

Cette commande entraîne le modèle sur 500 itérations, affiche la courbe de
précision, la précision finale sur les données de test, et un exemple de
prédiction.

Les hyperparamètres (nombre d'itérations, taux d'apprentissage, moment du
changement de taux) sont configurables en haut de `main.py`.

## Tests

```bash
pytest tests/ -v
```

## Structure du dépôt

```
Reseau-Neuronal/
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

MIT, voir [LICENSE](LICENSE).
