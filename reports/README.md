# Rapport d'Evaluation - SocialMetrics AI

Ce dossier contient le rapport d'evaluation du modele d'analyse de sentiments
ainsi que le script qui le genere automatiquement.

## Contenu

| Fichier | Description |
| --- | --- |
| `generate_report.py` | Script de generation : entrainement, matrices de confusion, mesures et export PDF. |
| `rapport_evaluation.pdf` | Rapport final en francais (livrable a rendre). |
| `confusion_matrix_positive.png` | Matrice de confusion des predictions positives. |
| `confusion_matrix_negative.png` | Matrice de confusion des predictions negatives. |

## Generer le rapport

Depuis la racine du projet :

```bash
python reports/generate_report.py
```

Le script charge les tweets annotes depuis MySQL si la base est disponible,
sinon il se rabat automatiquement sur `database/seed.sql`. Il produit ensuite
les deux matrices de confusion (PNG) et le rapport complet `rapport_evaluation.pdf`.

## Plan du rapport

1. Description du jeu de donnees (source, volumes, categories).
2. Explication du modele TF-IDF + LogisticRegression.
3. Protocole d'evaluation (separation train / validation).
4. Matrice de confusion pour les predictions positives, avec interpretation.
5. Matrice de confusion pour les predictions negatives, avec interpretation.
6. Precision, rappel et F1-score pour chaque classe.
7. Analyse des performances : forces, faiblesses, erreurs frequentes.
8. Biais possibles dans les predictions.
9. Recommandations pour ameliorer le modele.
