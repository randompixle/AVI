#!/usr/bin/env bash
set -euo pipefail

REMOTE_URL="https://github.com/randompixle/AVI.git"
BRANCH="main"

# init repo if needed
if [ ! -d .git ]; then
  git init
fi

git lfs install

git lfs track "datasets/**"

git add .gitattributes

# ensure datasets are not ignored
if [ -f .gitignore ]; then
  sed -i '/^datasets\/$/d' .gitignore
  sed -i '/^datasets\*\*$/d' .gitignore
  git add .gitignore || true
fi

# add everything
 git add -A

# commit if needed
if ! git diff --cached --quiet; then
  git commit -m "Sync datasets and code"
fi

# set branch + remote
if ! git remote | grep -q '^origin$'; then
  git remote add origin "$REMOTE_URL"
fi

git branch -M "$BRANCH"

# push
 git push -u origin "$BRANCH"
