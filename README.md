# SocialMetrics AI - API d'Analyse de Sentiments

Ce projet est une implementation du TP final : developper une API Flask capable d'analyser le sentiment de tweets, avec un modele de machine learning entraine a partir de donnees annotees stockees dans MySQL.

Le sujet ne fournit pas de dataset et ne demande pas de se connecter directement a X/Twitter. Les tweets utilises pour l'entrainement doivent etre ajoutes dans la table MySQL `tweets`, avec leurs labels `positive` et `negative`.

## 1. Objectif du Projet

L'objectif est de fournir un service pour l'entreprise fictive SocialMetrics AI. Le service recoit une liste de tweets et retourne un score de sentiment pour chaque texte :

- `-1` : sentiment tres negatif ;
- `0` : sentiment neutre ou ambigu ;
- `1` : sentiment tres positif.

Le projet couvre les points demandes dans l'enonce :

- une API avec Flask ;
- une base MySQL contenant les tweets annotes ;
- un modele `LogisticRegression` avec scikit-learn ;
- un script de reentrainement du modele ;
- un rapport d'evaluation avec matrices de confusion, precision, rappel et F1-score.

## 2. Structure du Projet

```text
.
├── app.py                     # Point d'entree pour lancer l'API Flask
├── app/
│   ├── config.py              # Chargement de la configuration et du fichier .env
│   ├── db.py                  # Connexion MySQL et lecture des tweets annotes
│   ├── model.py               # Entrainement, sauvegarde, chargement et prediction
│   └── server.py              # Routes Flask
├── database/
│   ├── schema.sql             # Creation de la base et de la table tweets
│   └── seed.sql               # Donnees annotees d'exemple
├── Dockerfile                 # Image Docker de l'API Flask
├── docker-compose.yml         # API Flask + MySQL avec Docker
├── scripts/
│   └── train.py               # Script de lancement de l'entrainement du modele
├── reports/
│   └── README.md              # Plan du rapport d'evaluation
└── requirements.txt
```

`scripts/train.py` permet de lancer l'entrainement du modele depuis la base MySQL.  
La configuration du reentrainement automatique via cronjob ou tache planifiee reste a completer.

## 3. Installation

### Option recommandee : Docker

Lancer l'API Flask et MySQL :

```bash
docker compose up --build
```

L'API est ensuite disponible sur :

```text
http://127.0.0.1:5001
```

Tester l'API :

```bash
curl http://127.0.0.1:5001/health
```

Arreter les conteneurs :

```bash
docker compose down
```

### Option alternative : environnement Python local

Creer un environnement virtuel :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Installer les dependances :

```bash
pip install -r requirements.txt
```

Creer le fichier de configuration local :

```bash
cp .env.example .env
```

Adapter ensuite `.env` selon la configuration MySQL locale :

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=socialmetrics
MODEL_DIR=models
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
```

Avec Docker, le mot de passe par defaut est `root`. Si vous utilisez une installation MySQL locale, adaptez `MYSQL_PASSWORD` a votre configuration.

Le fichier `.env.example` peut etre partage sur GitHub. Le fichier `.env` ne doit pas etre partage, car il peut contenir un mot de passe.

## 4. Base de Donnees MySQL

### Avec Docker

Demarrer uniquement MySQL avec Docker Compose :

```bash
docker compose up -d mysql
```

Le conteneur utilise :

```text
host depuis la machine locale: localhost
host depuis le conteneur API: mysql
port: 3306
user: root
password: root
database: socialmetrics
```

Au premier demarrage, Docker execute automatiquement les fichiers du dossier `database/` :

- `schema.sql` cree la base et la table `tweets` ;
- `seed.sql` insere des donnees annotees d'exemple.

Verifier que le conteneur fonctionne :

```bash
docker compose ps
```

Se connecter a MySQL depuis le conteneur :

```bash
docker compose exec mysql mysql -u root -proot socialmetrics
```

Exemple de verification dans MySQL :

```sql
SELECT COUNT(*) FROM tweets;
SELECT * FROM tweets LIMIT 5;
```

### Option alternative : MySQL local

Si MySQL est installe localement, creer la base et la table :

```bash
mysql -u root -p < database/schema.sql
```

Puis ajouter les donnees d'exemple :

```bash
mysql -u root -p < database/seed.sql
```

Structure principale demandee par l'enonce :

```sql
tweets(
  id INT PRIMARY KEY,
  text TEXT,
  positive TINYINT,
  negative TINYINT
)
```

Les donnees dans `database/seed.sql` sont des exemples pour demarrer le projet. Pour le rendu final, il faut idealement enrichir cette table avec davantage de tweets annotes manuellement ou avec un dataset public.

## 5. Entrainement du Modele

Avec Docker :

```bash
docker compose run --rm api python scripts/train.py
```

En local :

```bash
python scripts/train.py
```

Etapes realisees :

1. Lecture des colonnes `text`, `positive` et `negative` dans MySQL.
2. Transformation des textes avec `TfidfVectorizer`.
3. Entrainement de deux modeles `LogisticRegression` :
   - un modele pour predire `positive` ;
   - un modele pour predire `negative`.
4. Evaluation sur un jeu de validation.
5. Generation des matrices de confusion et des mesures precision, rappel et F1-score.
6. Sauvegarde du modele dans `models/sentiment_model.joblib`.

Le modele est entraine a partir des donnees MySQL. Si MySQL n'est pas disponible ou si la table `tweets` est vide, l'entrainement echoue avec une erreur explicite.

## 6. Lancement de l'API

Avec Docker :

```bash
docker compose up --build
```

L'API Docker est exposee sur le port `5001` :

```bash
curl http://127.0.0.1:5001/health
```

En local :

```bash
python app.py
```

Verifier que le serveur local fonctionne :

```bash
curl http://127.0.0.1:5000/health
```

Analyser des tweets :

```bash
curl -X POST http://127.0.0.1:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"tweets":["I love this feature","This service is terrible"]}'
```

Exemple de reponse :

```json
{
  "I love this feature": 0.5321,
  "This service is terrible": -0.4876
}
```

L'API accepte aussi directement une liste JSON :

```bash
curl -X POST http://127.0.0.1:5000/analyze \
  -H "Content-Type: application/json" \
  -d '["Great product","Bad experience"]'
```

## 7. Reentrainement

Lancement manuel de l'entrainement :

Avec Docker :

```bash
docker compose run --rm api python scripts/train.py
```

En local :

```bash
python scripts/train.py
```

Lancement via l'API :

```bash
curl -X POST http://127.0.0.1:5001/retrain
```

Le reentrainement automatique regulier reste a faire. Il devra etre ajoute avec un cronjob ou une tache planifiee, par exemple pour relancer `scripts/train.py` chaque semaine.

## 8. Reste a Faire pour le Membre B

Le projet est deja executable avec Docker, Flask et MySQL. Le membre B doit finaliser la partie donnees, reentrainement automatique et rapport d'evaluation.

Taches a realiser :

1. Enrichir les donnees d'entrainement

   Modifier `database/seed.sql` pour ajouter davantage de tweets annotes. Les labels doivent suivre la structure :

   ```text
   positive = 1 si le tweet est positif, 0 sinon
   negative = 1 si le tweet est negatif, 0 sinon
   ```

   Il faut ajouter des exemples positifs, negatifs, neutres et mixtes afin d'ameliorer l'entrainement du modele.

2. Verifier les donnees dans MySQL

   Apres modification de `seed.sql`, relancer la base si necessaire puis verifier que les tweets sont bien presents :

   ```bash
   docker compose exec mysql mysql -u root -proot socialmetrics
   ```

   Exemple de requete :

   ```sql
   SELECT COUNT(*) FROM tweets;
   SELECT positive, negative, COUNT(*) FROM tweets GROUP BY positive, negative;
   ```

3. Relancer l'entrainement

   Executer le script :

   ```bash
   docker compose run --rm api python scripts/train.py
   ```

   Recuperer les matrices de confusion et les scores affiches dans le terminal.

4. Ajouter le reentrainement automatique

   Ajouter un fichier d'exemple de cronjob ou de tache planifiee, par exemple :

   ```text
   scripts/reentrainement_cron.example
   ```

   Ce fichier doit montrer comment relancer `scripts/train.py` automatiquement chaque semaine.

5. Rediger le rapport d'evaluation

   Creer le rapport final en francais dans `reports/`, puis l'exporter en PDF. Le rapport doit contenir :

   - la description du dataset utilise ;
   - les deux matrices de confusion ;
   - precision, rappel et F1-score ;
   - une analyse des erreurs frequentes ;
   - les biais possibles ;
   - des recommandations pour ameliorer le modele.

Livrables attendus pour le membre B :

- `database/seed.sql` enrichi ;
- un exemple de reentrainement automatique ;
- le rapport final d'evaluation en PDF dans `reports/`.

## 9. Rapport d'Evaluation

Le rapport final doit etre exporte en PDF et ajoute dans le depot GitHub, par exemple dans le dossier `reports/`.

Contenu attendu :

1. Description du dataset utilise.
2. Explication du modele TF-IDF + LogisticRegression.
3. Matrice de confusion pour les predictions positives.
4. Matrice de confusion pour les predictions negatives.
5. Precision, rappel et F1-score pour chaque classe.
6. Analyse des erreurs frequentes et des biais possibles.
7. Recommandations pour ameliorer le modele.

Un plan detaille est disponible dans `reports/README.md`.

## 10. Repartition du Travail

Membre A : API, base de donnees et infrastructure

- Developper l'API Flask et les endpoints `/analyze`, `/sentiment` et `/retrain`.
- Mettre en place la connexion entre l'API et MySQL.
- Configurer Docker, Docker Compose et la structure de la base de donnees.
- Documenter l'installation et l'utilisation de l'API.

Membre B : donnees, reentrainement et rapport

- Enrichir `database/seed.sql` avec davantage de tweets annotes.
- Mettre en place le reentrainement automatise via cronjob ou tache planifiee.
- Evaluer le modele `LogisticRegression` avec les matrices de confusion.
- Analyser les scores precision, rappel et F1-score.
- Rediger le rapport d'evaluation final en francais.

Validation commune :

- verifier que l'API retourne un JSON correct ;
- verifier que MySQL contient des donnees annotees ;
- verifier que le modele se reentraine depuis la base ;
- verifier que le rapport contient les deux matrices de confusion et leur interpretation.
