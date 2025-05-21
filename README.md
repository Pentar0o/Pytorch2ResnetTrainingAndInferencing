# Détecteur d'Objets avec Faster R-CNN

Ce dépôt contient deux scripts Python pour l'entraînement et l'inférence d'un modèle de détection d'objets basé sur Faster R-CNN avec PyTorch.

## Description

Ce projet permet de:
1. **Entraîner** un modèle Faster R-CNN sur un jeu de données au format Pascal VOC
2. **Effectuer des prédictions** avec le modèle entraîné sur de nouvelles images

Le modèle utilise une architecture Faster R-CNN avec un backbone ResNet50 FPN v2 de torchvision.

## Prérequis

- Python 3.6+
- PyTorch 1.8+
- torchvision
- Pillow
- tqdm
- numpy

## Installation

```bash
git clone https://github.com/votre-username/votre-repo.git
cd votre-repo
pip3 install torch torchvision tqdm pillow --break-system-packages
```

## Structure des données

Pour l'entraînement, votre jeu de données doit être au format Pascal VOC avec la structure suivante:

```
dataset_directory/
├── JPEGImages/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Annotations/
    ├── image1.xml
    ├── image2.xml
    └── ...
```

## Utilisation

### Entraînement

```bash
python train.py -f CHEMIN_DATASET -d DEVICE -b BATCH_SIZE -e NUM_EPOCHS -l LEARNING_RATE -w NUM_WORKERS
```

Arguments:
- `-f, --repertoire`: Chemin vers le répertoire du jeu de données
- `-d, --device`: Device à utiliser pour l'entraînement ('cpu' ou 'cuda')
- `-b, --batch`: Taille du batch
- `-e, --epoch`: Nombre d'époques d'entraînement
- `-l, --lr`: Taux d'apprentissage (learning rate)
- `-w, --workers`: Nombre de workers pour le chargement des données

Exemple:
```bash
python train.py -f ./dataset -d cuda -b 4 -e 50 -l 0.005 -w 4
```

Le modèle avec les meilleures performances sur l'ensemble de validation sera sauvegardé dans `best_model.pth`.

### Inférence

```bash
python inference.py CHEMIN_MODELE REPERTOIRE_IMAGES [options]
```

Arguments obligatoires:
- `model_path`: Chemin vers le fichier du modèle entraîné (.pth)
- `images_dir`: Répertoire contenant les images à traiter

Options:
- `--confidence_threshold`: Seuil de confiance (défaut: 0.5)
- `--num_classes`: Nombre de classes (défaut: 2)
- `--device`: Device pour l'inférence ('cpu', 'cuda', 'mps') (défaut: 'cpu')
- `--results_dir`: Répertoire pour sauvegarder les résultats (défaut: 'results')

Exemple:
```bash
python inference.py best_model.pth ./test_images --confidence_threshold 0.7 --device cuda
```

## Fonctionnalités

### Train.py

- Chargement des données au format Pascal VOC
- Redimensionnement intelligent des images et des boîtes englobantes
- Entraînement avec programmation du taux d'apprentissage (Cosine Annealing)
- Évaluation avec la métrique mAP (mean Average Precision)
- Sauvegarde automatique du meilleur modèle
- Arrêt anticipé (early stopping) si les performances stagnent

### Inference.py

- Chargement d'un modèle préentraîné
- Traitement par lots d'images
- Visualisation des détections avec boîtes englobantes
- Filtrage des détections par seuil de confiance
- Mesure du temps d'inférence
- Support multi-plateformes (CPU, CUDA, MPS pour Mac M1/M2)

## Remarques

- Le modèle est configuré par défaut pour 2 classes (une classe d'objet + background)
- Pour l'entraînement sur plusieurs classes, le modèle détectera automatiquement les classes à partir des fichiers d'annotation
- Pour les Mac avec puces Apple Silicon, utilisez l'option `--device mps` lors de l'inférence

## Licence

MIT
