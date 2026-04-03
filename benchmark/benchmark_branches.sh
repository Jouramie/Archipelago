#!/usr/bin/env bash

# =============================================================================
# CONFIGURATION — modifier ces variables selon vos besoins
# =============================================================================

# Les branches git à comparer (la première sert de référence)
BRANCHES=(
    "main"
    "core/improve-region-cache"
)

# Script Python à benchmarker (relatif à la racine du repo)
PYTHON_SCRIPT="import Generate; from Main import main as ERmain; erargs, seed = Generate.main(); ERmain(erargs, seed)"

SCRIPT_ARGS=(--seed 1 --skip_output)

# Nombre de répétitions du benchmark par branche (pour la moyenne)
RUNS=10

# Dossier de sortie des résultats (relatif à la racine du repo)
OUTPUT_DIR="benchmark"

# Nom de base des fichiers de résultats (suffixe _A / _B ajouté automatiquement)
RESULT_BASENAME="bench_compare"

# =============================================================================
# FIN DE CONFIGURATION — ne pas modifier en dessous sauf si vous savez ce que
# vous faites
# =============================================================================

set -euo pipefail

RESUME=0
for arg in "$@"; do
    case "$arg" in
        --resume) RESUME=1 ;;
        *) error "Argument inconnu : $arg"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR_ABS="$REPO_ROOT/$OUTPUT_DIR"
RESULTS_FILE="$OUTPUT_DIR_ABS/${RESULT_BASENAME}_results.txt"

WORKTREE_BASE="$REPO_ROOT/../Archipelago_bench_worktree"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()    { echo -e "${CYAN}[bench]${NC} $*" >&2; }
success(){ echo -e "${GREEN}[bench]${NC} $*" >&2; }
warn()   { echo -e "${YELLOW}[bench]${NC} $*" >&2; }
error()  { echo -e "${RED}[bench]${NC} $*" >&2; }
header() { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${NC}" >&2; \
           echo -e "${BOLD}${CYAN}  $*${NC}" >&2; \
           echo -e "${BOLD}${CYAN}══════════════════════════════════════════${NC}" >&2; }

# Vérifie que les outils nécessaires sont disponibles
check_dependencies() {
    local missing=0
    for cmd in git python pyperf; do
        if ! command -v "$cmd" &>/dev/null; then
            error "Commande manquante : $cmd"
            missing=1
        fi
    done
    if [[ $missing -eq 1 ]]; then
        error "Installez les dépendances manquantes (ex: pip install pyperf) et relancez."
        exit 1
    fi
}

# Vérifie que la branche existe (locale ou distante)
branch_exists() {
    local branch="$1"
    git -C "$REPO_ROOT" rev-parse --verify "$branch" &>/dev/null || \
    git -C "$REPO_ROOT" rev-parse --verify "origin/$branch" &>/dev/null
}

# Crée (ou réutilise) un worktree et y copie le dossier Players/
setup_worktree() {
    local branch="$1"
    local worktree="$2"
    local commit

    if git -C "$REPO_ROOT" rev-parse --verify "$branch" &>/dev/null; then
        commit="$(git -C "$REPO_ROOT" rev-parse "$branch")"
    else
        log "Branche locale '$branch' introuvable, utilisation de origin/$branch"
        commit="$(git -C "$REPO_ROOT" rev-parse "origin/$branch")"
    fi

    if [[ -d "$worktree" ]]; then
        warn "Worktree existant détecté, réutilisation : $worktree"
        warn "Positionnement en detached HEAD sur $commit"
        git -C "$worktree" checkout --detach "$commit"
    else
        log "Création du worktree pour $branch → $worktree"
        git -C "$REPO_ROOT" worktree add --detach "$worktree" "$commit"
    fi

    log "Copie de Players/ dans le worktree $branch..."
    rm -rf "$worktree/Players/"*
    cp -r "$REPO_ROOT/Players/." "$worktree/Players/"
    success "Players/ copié dans $worktree"
}

# Lance le benchmark pyperf dans un worktree donné et enregistre le JSON
run_benchmark() {
    local label="$1"
    local worktree="$2"
    local safe_label="${label//\//_}"
    local out_json="$OUTPUT_DIR_ABS/${RESULT_BASENAME}_${safe_label}.json"

    if [[ $RESUME -eq 1 && -f "$out_json" ]]; then
        warn "Résultat existant trouvé pour '$label', réutilisation : $out_json"
        LAST_BENCH_JSON="$out_json"
        return
    fi

    log "Benchmark branche ${BOLD}$label${NC} — $RUNS répétitions..."
    rm -f "$out_json"

    # IMPORTANT: pyperf attend un programme + args séparés après "--".
    # Et on évite Generate.py en __main__ (atexit(input) provoque un échec en worker).
    cd "$worktree"
    python -m pyperf command \
        --stats \
        --processes 1 \
        --loops 1 \
        --values "$RUNS" \
        --output "$out_json" \
        -- python -O -c "$PYTHON_SCRIPT" "${SCRIPT_ARGS[@]}"

    success "Résultats enregistrés → $out_json"
    LAST_BENCH_JSON="$out_json"
}

# Affiche un résumé des métriques depuis un JSON pyperf
summarize() {
    local label="$1"
    local json="$2"
    echo -e "\n${BOLD}── Branche : $label ──${NC}"
    python -m pyperf stats "$json"
}

# ── Main ──────────────────────────────────────────────────────────────────────

header "Benchmark comparatif de branches git"

log "Dépôt  : $REPO_ROOT"
for branch in "${BRANCHES[@]}"; do
    log "Branche : $branch"
done
log "Runs par branche : $RUNS"
echo ""

check_dependencies

# Vérification des branches
for branch in "${BRANCHES[@]}"; do
    if ! branch_exists "$branch"; then
        error "La branche '$branch' est introuvable (locale ou distante)."
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR_ABS"
: > "$RESULTS_FILE"   # réinitialise le fichier de résumé

declare -a JSONS=()
LAST_BENCH_JSON=""

# ── Benchmark de chaque branche ───────────────────────────────────────────────
for i in "${!BRANCHES[@]}"; do
    branch="${BRANCHES[$i]}"
    worktree="${WORKTREE_BASE}_${i}"
    safe_branch="${branch//\//_}"
    existing_json="$OUTPUT_DIR_ABS/${RESULT_BASENAME}_${safe_branch}.json"
    header "Branche $i : $branch"
    if [[ $RESUME -eq 1 && -f "$existing_json" ]]; then
        warn "Mode --resume : worktree ignoré pour '$branch' (résultat existant)"
        JSONS+=("$existing_json")
    else
        setup_worktree "$branch" "$worktree"
        run_benchmark "$branch" "$worktree"
        JSONS+=("$LAST_BENCH_JSON")
    fi
done

# ── Comparaison ───────────────────────────────────────────────────────────────
header "Résultats"

{
    echo "===== Benchmark comparatif — $(date) ====="
    echo "Script  : $PYTHON_SCRIPT"
    printf 'Args    :'; printf ' %q' "${SCRIPT_ARGS[@]}"; echo
    echo "Runs    : $RUNS"
    echo ""
} | tee -a "$RESULTS_FILE"

for i in "${!BRANCHES[@]}"; do
    summarize "${BRANCHES[$i]}" "${JSONS[$i]}" | tee -a "$RESULTS_FILE"
done

echo "" | tee -a "$RESULTS_FILE"
echo -e "${BOLD}── Comparaison directe (référence = ${BRANCHES[0]}) ──${NC}" | tee -a "$RESULTS_FILE"
python -m pyperf compare_to "${JSONS[0]}" "${JSONS[@]:1}" --table 2>&1 | tee -a "$RESULTS_FILE"

success "Résumé complet sauvegardé → $RESULTS_FILE"
