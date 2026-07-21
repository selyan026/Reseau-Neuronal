# Comment fonctionne l'entraînement du réseau — explication de A à Z

Ce document explique en détail comment le réseau de neurones apprend à reconnaître les chiffres manuscrits, étape par étape, à partir des données brutes jusqu'à la mise à jour des poids.

## 1. Le point de départ : les données

Chaque image du dataset MNIST est un chiffre manuscrit de **28x28 pixels**, donc **784 pixels** au total. Chaque pixel a une valeur de gris entre 0 (noir) et 255 (blanc).

Dans le code :
```python
X_train = X_train / 255.
```
On divise par 255 pour que chaque pixel soit une valeur entre 0 et 1 plutôt qu'entre 0 et 255. C'est la **normalisation**. Sans ça, le réseau apprendrait beaucoup plus lentement (les gradients seraient mal calibrés).

Chaque image est donc transformée en un **vecteur de 784 nombres**. Si on a `m` images d'entraînement, la matrice `X` a la forme `(784, m)` : 784 lignes (une par pixel), m colonnes (une par image).

`Y` contient les labels : le vrai chiffre (0 à 9) associé à chaque image.

## 2. L'architecture du réseau

Le réseau a 3 couches :
- **Couche d'entrée** : 784 neurones (un par pixel)
- **Couche cachée** : 10 neurones
- **Couche de sortie** : 10 neurones (un par chiffre possible, 0 à 9)

Chaque connexion entre deux neurones a un **poids** (`W`), et chaque neurone a un **biais** (`b`). Ce sont ces poids et biais que le réseau va apprendre à ajuster.

```python
W1 = np.random.rand(10, 784) - 0.5   # poids entrée -> couche cachée
b1 = np.random.rand(10, 1) - 0.5     # biais de la couche cachée
W2 = np.random.rand(10, 10) - 0.5    # poids couche cachée -> sortie
b2 = np.random.rand(10, 1) - 0.5     # biais de la couche de sortie
```
Au départ, tout est **aléatoire** (`- 0.5` centre les valeurs autour de 0). Le réseau ne sait rien faire au début : il doit tout apprendre.

## 3. La propagation en avant (forward propagation)

C'est le calcul qui transforme une image en prédiction. On fait passer les données à travers le réseau, couche par couche.

**Étape 1 — couche cachée**
```
Z1 = W1.X + b1
A1 = ReLU(Z1)
```
- `Z1` est une **combinaison linéaire** : chaque neurone caché fait une somme pondérée des 784 pixels, plus un biais.
- `ReLU` (Rectified Linear Unit) est la **fonction d'activation** : `ReLU(x) = max(0, x)`. Elle transforme toute valeur négative en 0, et laisse les valeurs positives inchangées. Sans fonction d'activation, empiler des couches reviendrait juste à faire une seule grosse opération linéaire — le réseau ne pourrait pas apprendre de relations complexes (non linéaires) dans les données. ReLU introduit cette non-linéarité tout en restant simple et rapide à calculer.

**Étape 2 — couche de sortie**
```
Z2 = W2.A1 + b2
A2 = softmax(Z2)
```
- Même principe : combinaison linéaire des sorties de la couche cachée.
- `softmax` transforme les 10 valeurs de sortie en **probabilités** qui somment à 1 :
```python
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
```
Le résultat `A2` est donc un vecteur de 10 nombres entre 0 et 1, par exemple `[0.01, 0.02, 0.85, 0.03, ...]` — ici le réseau serait sûr à 85% que l'image représente un "2" (3ᵉ position = index 2).

## 4. Mesurer l'erreur : encodage one-hot et fonction de coût

Pour dire au réseau "tu t'es trompé de tant", il faut comparer sa prédiction (`A2`, des probabilités) au vrai label (`Y`, un simple chiffre comme `2`).

On transforme donc chaque label en vecteur **one-hot** : un vecteur de 10 valeurs, toutes à 0 sauf un 1 à la position du bon chiffre.
```python
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
```
Exemple : le label `2` devient `[0,0,1,0,0,0,0,0,0,0]`.

La **fonction de coût** quantifie l'écart entre la prédiction et la réalité sur l'ensemble des exemples :
```
C(w,b) = (1/n) Σ ||y(x) - a||²
```
Plus le coût est élevé, plus le réseau se trompe. **L'objectif de l'entraînement est de minimiser ce coût.**

## 5. La rétropropagation (backpropagation) : comprendre d'où vient l'erreur

C'est le cœur mathématique de l'apprentissage. Le principe : une fois qu'on connaît l'erreur en sortie, on la fait **remonter** couche par couche vers l'entrée, pour savoir quelle part de responsabilité a chaque poids et chaque biais dans cette erreur. C'est une application directe de la **règle de dérivation en chaîne** (chain rule) de l'analyse.

```python
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)
    return dW1, db1, dW2, db2
```

Décomposons :

- **`dZ2 = A2 - one_hot_Y`** : c'est l'erreur brute en sortie. Simplement la différence entre ce que le réseau a prédit et ce qu'il aurait dû prédire. Si le réseau était sûr à 100% du bon chiffre, `dZ2` serait un vecteur de zéros — pas d'erreur, pas de correction nécessaire.

- **`dW2 = 1/m * dZ2.A1ᵀ`** : indique de combien il faut ajuster chaque poids de la couche de sortie. L'idée : un poids qui reliait un neurone caché très actif (`A1` élevé) à une erreur importante (`dZ2` élevé) porte une plus grande responsabilité dans l'erreur, donc doit être corrigé davantage.

- **`db2`** : même logique pour les biais, en moyenne sur tous les exemples (`1/m`).

- **`dZ1 = W2ᵀ.dZ2 * ReLU_deriv(Z1)`** : c'est l'étape clé de la "rétro"-propagation. On fait remonter l'erreur de la couche de sortie vers la couche cachée, en passant à nouveau par les poids `W2` (d'où `W2ᵀ`). On multiplie ensuite par la **dérivée de ReLU** (`ReLU_deriv(Z1) = (Z1 > 0)`) : un neurone qui était "éteint" (sortie 0 pendant la propagation en avant, car son entrée était négative) n'a pas contribué à la sortie, donc il ne reçoit **aucune correction** — sa dérivée est nulle.

- **`dW1`, `db1`** : mêmes calculs que pour la couche 2, mais appliqués à la couche 1, avec `X` (l'entrée) à la place de `A1`.

En résumé : on calcule l'erreur en sortie, puis on la "distribue" en arrière à travers chaque couche, proportionnellement à la contribution de chaque poids et chaque neurone à cette erreur.

## 6. La descente de gradient : corriger les poids

Une fois qu'on sait dans quelle direction et de combien chaque poids a contribué à l'erreur (les `dW1, db1, dW2, db2`), on les ajuste dans le sens **opposé** au gradient — c'est-à-dire dans le sens qui fait diminuer l'erreur :

```python
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2
```

L'intuition géométrique (le graphe "Cost(a)" de la présentation) : on imagine la fonction de coût comme une vallée. Le gradient indique la direction de la pente la plus raide **vers le haut**. On avance donc dans la direction opposée, pas à pas, pour descendre vers le minimum (l'erreur la plus faible possible).

**`alpha`** est le **taux d'apprentissage (learning rate)** : il contrôle la taille du pas à chaque itération.
- Trop petit → l'apprentissage est très lent.
- Trop grand → le réseau peut "sauter" par-dessus le minimum et osciller, voire diverger (c'est le phénomène de "saut" observé sur les slides 12-13 avec alpha = 10).

## 7. La boucle complète

L'entraînement répète ce cycle des centaines de fois :

```python
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    return W1, b1, W2, b2
```

À chaque itération :
1. **Propagation en avant** : le réseau fait une prédiction avec ses poids actuels.
2. **Rétropropagation** : on calcule l'erreur et sa contribution à chaque poids.
3. **Mise à jour** : on ajuste les poids pour réduire l'erreur.

Répété suffisamment de fois, sur suffisamment d'exemples, le réseau converge progressivement vers des poids qui lui permettent de reconnaître les chiffres avec une bonne précision.

## 8. L'amélioration : alpha adaptatif

Ton expérimentation a montré qu'un `alpha` fixe pose un compromis : grand alpha = apprentissage rapide au début mais instable en fin de course (les "sauts" observés) ; petit alpha = stable mais lent.

Solution testée : commencer avec un grand `alpha` (= 1) pour progresser vite, puis le réduire (= 0.1) une fois qu'on approche du minimum, pour affiner sans "sauter" par-dessus. C'est ce qui a donné le meilleur résultat : **90.7% de précision sur les données de test**, contre ~83-88% avec un alpha fixe.

C'est un principe assez classique en machine learning, souvent appelé *learning rate decay* ou *learning rate scheduling*.

## 9. Pourquoi ça marche : le résumé en une phrase

Le réseau part de poids aléatoires (donc de prédictions n'importe quoi), et à chaque itération : il mesure son erreur, calcule comment chaque poids y a contribué (rétropropagation), puis ajuste légèrement chaque poids dans la direction qui réduit cette erreur (descente de gradient). Répété des centaines de fois sur des milliers d'exemples, ce processus purement mécanique — sans aucune règle écrite à la main sur "à quoi ressemble un 7" — suffit à faire émerger un système capable de reconnaître des chiffres manuscrits avec une précision proche de 91%.
