"""Generation du rapport d'evaluation du modele SocialMetrics AI.

Ce script :
  1. charge les tweets annotes (MySQL si disponible, sinon depuis
     database/seed.sql en repli, ce qui permet de generer le rapport
     meme sans base demarree) ;
  2. entraine deux regressions logistiques (positive et negative) avec
     exactement le meme pipeline TF-IDF + LogisticRegression que l'API ;
  3. calcule les matrices de confusion et les mesures precision, rappel
     et F1-score sur un jeu de validation ;
  4. exporte les deux matrices de confusion en PNG ;
  5. genere un rapport PDF complet en francais dans reports/.

Utilisation :
    python reports/generate_report.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Permet d'importer le package `app` quel que soit le repertoire courant.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.model import _build_pipeline  # noqa: E402  (import apres sys.path)

REPORTS_DIR = PROJECT_ROOT / "reports"
SEED_FILE = PROJECT_ROOT / "database" / "seed.sql"
RANDOM_STATE = 42
N_SPLITS = 5


# ---------------------------------------------------------------------------
# Chargement des donnees
# ---------------------------------------------------------------------------
def load_tweets() -> tuple[list[str], list[int], list[int], str]:
    """Retourne (textes, labels_positifs, labels_negatifs, source).

    Essaie d'abord MySQL ; en cas d'echec, se rabat sur seed.sql.
    """
    try:
        from app.db import fetch_annotated_tweets

        rows = fetch_annotated_tweets()
        if rows:
            texts = [r.text for r in rows]
            positive = [r.positive for r in rows]
            negative = [r.negative for r in rows]
            return texts, positive, negative, "MySQL (table tweets)"
    except Exception as exc:  # noqa: BLE001 - repli volontaire
        print(f"[info] MySQL indisponible ({exc}). Lecture depuis seed.sql.")

    return (*_parse_seed(SEED_FILE), "Fichier database/seed.sql")


def _parse_seed(path: Path) -> tuple[list[str], list[int], list[int]]:
    """Extrait les tuples ('texte', positive, negative) du fichier seed.sql."""
    content = path.read_text(encoding="utf-8")
    # Capture: ('... texte ...', 0/1, 0/1)
    pattern = re.compile(r"\('((?:[^']|'')*)'\s*,\s*([01])\s*,\s*([01])\)")
    texts: list[str] = []
    positive: list[int] = []
    negative: list[int] = []
    for match in pattern.finditer(content):
        texts.append(match.group(1).replace("''", "'"))
        positive.append(int(match.group(2)))
        negative.append(int(match.group(3)))
    if not texts:
        raise RuntimeError(f"Aucun tweet trouve dans {path}")
    return texts, positive, negative


# ---------------------------------------------------------------------------
# Entrainement et evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(
    texts: list[str],
    positive: list[int],
    negative: list[int],
) -> dict:
    """Evalue chaque modele par validation croisee stratifiee.

    Avec un petit jeu de donnees, une simple separation train/validation
    est trop bruitee. On utilise donc cross_val_predict : chaque tweet
    recoit une prediction lorsqu'il se trouve dans le pli de test, ce qui
    fournit une matrice de confusion agregee sur l'ensemble des donnees.
    """
    x = np.array(texts, dtype=object)
    n_splits = min(N_SPLITS, _min_class_count(positive), _min_class_count(negative))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    results = {"n_samples": len(texts), "n_splits": n_splits}
    for name, labels in (("positive", positive), ("negative", negative)):
        y = np.array(labels)
        predictions = cross_val_predict(_build_pipeline(), x, y, cv=cv)
        results[name] = {
            "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]),
            "report": classification_report(
                y,
                predictions,
                labels=[0, 1],
                target_names=["classe 0", "classe 1"],
                output_dict=True,
                zero_division=0,
            ),
            "accuracy": accuracy_score(y, predictions),
        }
    return results


def _min_class_count(labels: list[int]) -> int:
    counts = {label: labels.count(label) for label in set(labels)}
    return min(counts.values())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plot_confusion_matrix(matrix: np.ndarray, title: str, positive_label: str):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = [f"Non {positive_label}", positive_label]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Verite terrain")
    ax.set_title(title)

    threshold = matrix.max() / 2.0 if matrix.max() > 0 else 0.5
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                int(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=14,
                fontweight="bold",
            )
    fig.tight_layout()
    return fig


def _text_page(title: str, lines: list[str]):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(0.08, 0.94, title, fontsize=16, fontweight="bold")
    y = 0.89
    for line in lines:
        if line.startswith("## "):
            y -= 0.012
            fig.text(0.08, y, line[3:], fontsize=12, fontweight="bold")
            y -= 0.028
        else:
            fig.text(0.08, y, line, fontsize=10, family="DejaVu Sans")
            y -= 0.022
    return fig


def _metrics_lines(name: str, block: dict) -> list[str]:
    report = block["report"]
    cm = block["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    lines = [
        f"## Modele {name}",
        f"Exactitude (accuracy) : {block['accuracy']:.3f}",
        f"Matrice de confusion : VN={tn}  FP={fp}  FN={fn}  VP={tp}",
        "",
        "Mesures par classe :",
        f"  Classe 0 (absence)  -> precision {report['classe 0']['precision']:.3f} | "
        f"rappel {report['classe 0']['recall']:.3f} | F1 {report['classe 0']['f1-score']:.3f}",
        f"  Classe 1 (presence) -> precision {report['classe 1']['precision']:.3f} | "
        f"rappel {report['classe 1']['recall']:.3f} | F1 {report['classe 1']['f1-score']:.3f}",
        f"  Moyenne ponderee    -> precision {report['weighted avg']['precision']:.3f} | "
        f"rappel {report['weighted avg']['recall']:.3f} | F1 {report['weighted avg']['f1-score']:.3f}",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Rapport PDF
# ---------------------------------------------------------------------------
def build_report(results: dict, source: str, n_total: int) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig_pos = _plot_confusion_matrix(
        results["positive"]["confusion_matrix"],
        "Matrice de confusion - Sentiment positif",
        "Positif",
    )
    fig_pos.savefig(REPORTS_DIR / "confusion_matrix_positive.png", dpi=150)

    fig_neg = _plot_confusion_matrix(
        results["negative"]["confusion_matrix"],
        "Matrice de confusion - Sentiment negatif",
        "Negatif",
    )
    fig_neg.savefig(REPORTS_DIR / "confusion_matrix_negative.png", dpi=150)

    pdf_path = REPORTS_DIR / "rapport_evaluation.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1 - Couverture + dataset + modele
        cover = _text_page(
            "Rapport d'evaluation - SocialMetrics AI",
            [
                "Analyse de sentiments de tweets - TF-IDF + Regression logistique",
                "",
                "## 1. Jeu de donnees",
                f"Source des donnees : {source}",
                f"Nombre total de tweets annotes : {n_total}",
                "Annotation : positive=1 si positif, negative=1 si negatif.",
                "Categories : positifs (1,0), negatifs (0,1), neutres (0,0),",
                "mixtes (1,1). Le jeu melange des tweets en anglais et francais.",
                "",
                "## 2. Modele",
                "Deux modeles independants de regression logistique sont",
                "entraines : l'un predit le label 'positive', l'autre 'negative'.",
                "Chaque texte est vectorise avec TfidfVectorizer (unigrammes et",
                "bigrammes). La regression logistique utilise class_weight=",
                "'balanced' pour compenser le desequilibre des classes.",
                "Le score final renvoye par l'API vaut P(positif) - P(negatif),",
                "borne entre -1 (tres negatif) et +1 (tres positif).",
                "",
                "## 3. Protocole d'evaluation",
                f"Validation croisee stratifiee a {results['n_splits']} plis",
                "(cross_val_predict) : chaque tweet est predit lorsqu'il appartient",
                "au pli de test. La matrice de confusion est donc agregee sur les",
                f"{results['n_samples']} tweets, ce qui reduit la variance liee a la",
                "petite taille du jeu de donnees.",
            ],
        )
        pdf.savefig(cover)
        plt.close(cover)

        # Page 2 - Matrices de confusion
        pdf.savefig(fig_pos)
        pdf.savefig(fig_neg)
        plt.close(fig_pos)
        plt.close(fig_neg)

        # Page 3 - Mesures detaillees
        metrics_page = _text_page(
            "4. Mesures detaillees (precision, rappel, F1-score)",
            _metrics_lines("positif", results["positive"])
            + _metrics_lines("negatif", results["negative"])
            + [
                "## Lecture des matrices",
                "VN = vrais negatifs, FP = faux positifs,",
                "FN = faux negatifs, VP = vrais positifs.",
                "precision = VP / (VP + FP) : fiabilite des predictions positives.",
                "rappel = VP / (VP + FN) : capacite a retrouver les cas positifs.",
                "F1 = moyenne harmonique de la precision et du rappel.",
            ],
        )
        pdf.savefig(metrics_page)
        plt.close(metrics_page)

        # Page 4 - Analyse, biais et recommandations
        analysis_page = _text_page(
            "5. Analyse, biais et recommandations",
            [
                "## Forces",
                "- Les tweets clairement positifs ou negatifs sont bien separes",
                "  grace a un vocabulaire polarise (love, excellent / hate, awful).",
                "- L'usage de bigrammes capture des expressions comme 'not work'.",
                "- class_weight='balanced' limite le biais vers la classe majoritaire.",
                "",
                "## Faiblesses et erreurs frequentes",
                "- Les tweets MIXTES (1,1) sont les plus difficiles : un meme texte",
                "  contient des indices positifs et negatifs (ex. 'great design but",
                "  poor performance'), ce qui provoque des faux positifs/negatifs.",
                "- Les tweets NEUTRES (0,0) factuels peuvent etre mal classes si un",
                "  mot ambigu apparait.",
                "- La negation et l'ironie ne sont que partiellement captees par TF-IDF.",
                "",
                "## Biais possibles",
                "- Biais de vocabulaire : le modele s'appuie sur des mots-cles ; un",
                "  tweet positif sans terme fortement polarise peut etre manque.",
                "- Biais linguistique : melange anglais/francais avec un corpus reduit,",
                "  les mots rares d'une langue sont sous-representes.",
                "- Biais de taille : jeu de donnees encore reduit, les mesures",
                "  conservent une variance non negligeable malgre la validation",
                "  croisee ; il faut confirmer les resultats sur un corpus plus grand.",
                "",
                "## Recommandations",
                "- Augmenter fortement le volume de tweets annotes (plusieurs milliers)",
                "  et equilibrer les quatre categories.",
                "- Ajouter un pretraitement : normalisation, gestion des emojis, des",
                "  hashtags, des mentions et de la negation.",
                "- Tester des n-grammes plus larges, un lemmatiseur et un reglage",
                "  d'hyperparametres (C, min_df) par validation croisee.",
                "- Envisager un modele unique multi-label ou un classifieur a 3 classes",
                "  (negatif / neutre / positif) pour mieux gerer les cas ambigus.",
                "- Mettre en place un suivi des mesures a chaque reentrainement",
                "  hebdomadaire pour detecter toute derive du modele.",
            ],
        )
        pdf.savefig(analysis_page)
        plt.close(analysis_page)

    print(f"[ok] Matrices PNG : {REPORTS_DIR / 'confusion_matrix_positive.png'}")
    print(f"[ok]              {REPORTS_DIR / 'confusion_matrix_negative.png'}")
    print(f"[ok] Rapport PDF  : {pdf_path}")


def main() -> None:
    texts, positive, negative, source = load_tweets()
    print(f"[info] {len(texts)} tweets charges depuis : {source}")
    results = train_and_evaluate(texts, positive, negative)
    build_report(results, source, n_total=len(texts))


if __name__ == "__main__":
    main()
