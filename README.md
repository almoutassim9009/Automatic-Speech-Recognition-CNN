# -Speech_CNN.py

# Reconnaissance Automatique de la Parole avec CNN

Ce projet présente une implémentation de la reconnaissance automatique de la parole (ASR) en utilisant un réseau de neurones convolutionnel (CNN). Le modèle CNN est conçu pour traiter des spectrogrammes audio en entrée et prédire la classe correspondante, représentant le mot ou le phonème prononcé.

## Fonctionnalités

- **Modèle CNN pour l'ASR**: Le modèle CNN est conçu pour extraire des caractéristiques pertinentes à partir des spectrogrammes audio et effectuer une classification précise des données vocales.
  
- **Prétraitement des Données**: Les données audio sont prétraitées pour être converties en spectrogrammes, qui sont ensuite utilisés comme entrée pour le modèle CNN.

- **Entraînement et Évaluation**: Le modèle est entraîné sur un ensemble de données d'entraînement et évalué sur un ensemble de données de test pour évaluer sa performance en termes de précision de classification.

## Contenu du Projet

- `speech_cnn.py`: Fichier principal contenant l'implémentation du modèle CNN pour l'ASR, ainsi que le code d'entraînement et d'évaluation du modèle.

- `data_preprocessing.py`: Script de prétraitement des données audio, convertissant les fichiers audio en spectrogrammes pour l'entraînement et l'évaluation du modèle.

- `requirements.txt`: Fichier contenant les dépendances Python nécessaires pour exécuter le projet.

## Utilisation

1. **Installation des Dépendances**: Installez les dépendances nécessaires en utilisant `pip install -r requirements.txt`.

2. **Prétraitement des Données**: Utilisez `data_preprocessing.py` pour prétraiter les données audio et générer les spectrogrammes nécessaires pour l'entraînement du modèle.

3. **Entraînement du Modèle**: Exécutez `speech_cnn.py` pour entraîner le modèle CNN sur les données prétraitées.

4. **Évaluation du Modèle**: Une fois l'entraînement terminé, évaluez les performances du modèle en termes de précision de classification.


