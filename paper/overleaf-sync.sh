#!/usr/bin/env bash
#
# Overleaf <-> repo sync for the ictc paper, the SIMPLE way.
# --------------------------------------------------------------------------
# Instead of fighting Overleaf's git-bridge (no force-push, unrelated history)
# with git subtree, we keep a PLAIN CLONE of the Overleaf project inside
# paper/.overleaf/ (gitignored) and mirror source files in/out with rsync.
# Every git op in that clone is an ordinary fast-forward push/pull.
#
# The Overleaf project mirrors this layout:
#     <overleaf root>/ictc/     <- from paper/ictc/
#     <overleaf root>/shared/   <- from paper/shared/
# so ictc/main.tex's  \graphicspath{{../shared/}}  and
# \bibliography{../shared/references}  resolve untouched.
# In Overleaf: Menu -> Main document -> ictc/main.tex
#
# One-time setup (in a browser): create a BLANK Overleaf project, then
# Menu -> Git to get its URL (https://git.overleaf.com/<ID>) and a git token.
# Cache the token so you aren't re-prompted:  git config --global credential.helper store
#
# Usage (run from anywhere in the repo):
#   paper/overleaf-sync.sh clone https://git.overleaf.com/<ID>
#       Clone the Overleaf project into paper/.overleaf/ (do this once).
#       username = git , password = your Overleaf git token.
#
#   paper/overleaf-sync.sh push        # paper/ictc + paper/shared -> Overleaf
#   paper/overleaf-sync.sh pull        # Overleaf -> paper/ictc + paper/shared
#
# Rule of thumb: if you edited in the Overleaf web editor, `pull` before `push`.
# --------------------------------------------------------------------------
set -euo pipefail

MONO_ROOT="$(git rev-parse --show-toplevel)"
PAPER="$MONO_ROOT/paper"
CLONE_DIR="${OVERLEAF_DIR:-$PAPER/.overleaf}"

# LaTeX build artefacts (regenerable) — never mirror these. Figure PDFs are
# NOT excluded: they are build inputs and must sync.
EXCLUDES=(
  --exclude '.git'
  --exclude '*.aux' --exclude '*.log' --exclude '*.out'
  --exclude '*.bbl' --exclude '*.blg' --exclude '*.toc'
  --exclude '*.lof' --exclude '*.lot'
  --exclude '*.fls' --exclude '*.fdb_latexmk'
  --exclude '*.synctex.gz' --exclude '*.synctex(busy)'
)

usage() { sed -n '2,40p' "$0"; }

require_clone() {
  if [ ! -d "$CLONE_DIR/.git" ]; then
    echo "error: no Overleaf clone at $CLONE_DIR" >&2
    echo "  run:  paper/overleaf-sync.sh clone https://git.overleaf.com/<ID>" >&2
    exit 1
  fi
}

cmd="${1:-}"
case "$cmd" in
  clone)
    url="${2:-}"
    if [ -z "$url" ]; then
      echo "error: clone needs the Overleaf git URL" >&2
      echo "  paper/overleaf-sync.sh clone https://git.overleaf.com/<ID>" >&2
      exit 1
    fi
    if [ -d "$CLONE_DIR" ]; then
      echo "error: $CLONE_DIR already exists (remove it or set OVERLEAF_DIR)" >&2
      exit 1
    fi
    git clone "$url" "$CLONE_DIR"
    echo ">> cloned into $CLONE_DIR"
    echo ">> next:  paper/overleaf-sync.sh push"
    ;;

  push)
    require_clone
    echo ">> mirroring paper/ictc + paper/shared into the Overleaf clone ..."
    rsync -a --delete "${EXCLUDES[@]}" "$PAPER/ictc/"   "$CLONE_DIR/ictc/"
    rsync -a --delete "${EXCLUDES[@]}" "$PAPER/shared/" "$CLONE_DIR/shared/"
    # Blank Overleaf projects ship a placeholder main.tex at the root — remove it.
    rm -f "$CLONE_DIR/main.tex"
    ( cd "$CLONE_DIR"
      git add -A
      if git diff --cached --quiet; then
        echo ">> nothing changed; Overleaf already up to date."
      else
        git commit -m "Sync from monorepo paper/ ($(git -C "$MONO_ROOT" rev-parse --short HEAD))"
        git push
        echo ">> pushed to Overleaf."
      fi
    )
    echo ">> reminder: set Overleaf main document to ictc/main.tex"
    ;;

  pull)
    require_clone
    echo ">> pulling latest from Overleaf ..."
    ( cd "$CLONE_DIR" && git pull --ff-only )
    echo ">> mirroring Overleaf clone back into paper/ictc + paper/shared ..."
    rsync -a --delete "${EXCLUDES[@]}" "$CLONE_DIR/ictc/"   "$PAPER/ictc/"
    rsync -a --delete "${EXCLUDES[@]}" "$CLONE_DIR/shared/" "$PAPER/shared/"
    echo ">> done. Review with: git -C \"$MONO_ROOT\" status"
    ;;

  *)
    usage
    exit 1
    ;;
esac
