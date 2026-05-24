#!/bin/bash
REPO="/Users/gaetan/Documents/IA/ia-trading-bot"
cd "$REPO" || exit 1

git diff --quiet && git diff --cached --quiet && exit 0

DATE=$(date '+%Y-%m-%d %H:%M')
git add -u
git commit -m "auto: sauvegarde automatique $DATE"
git push
