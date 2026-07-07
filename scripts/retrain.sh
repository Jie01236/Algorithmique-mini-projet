#!/usr/bin/env bash
#
# Reentrainement hebdomadaire du modele SocialMetrics AI.
# Ce script relance l'entrainement depuis la base MySQL et enregistre
# la sortie (metriques, matrices de confusion) dans le dossier logs/.
#
# Utilisation manuelle :
#   ./scripts/retrain.sh
#
# Utilisation via cron : voir scripts/reentrainement_cron.example

set -euo pipefail

# Racine du projet (le dossier parent de scripts/)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG_FILE="$LOG_DIR/retrain_${TIMESTAMP}.log"

# Choix de l'interpreteur Python (venv si disponible)
if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
  PYTHON="$PROJECT_DIR/.venv/bin/python"
else
  PYTHON="python3"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Debut du reentrainement" | tee -a "$LOG_FILE"

if "$PYTHON" scripts/train.py >>"$LOG_FILE" 2>&1; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reentrainement termine avec succes" | tee -a "$LOG_FILE"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ECHEC du reentrainement (voir $LOG_FILE)" | tee -a "$LOG_FILE"
  exit 1
fi
