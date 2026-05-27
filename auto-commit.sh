#!/bin/bash
REPO="/Users/gaetan/Documents/IA/ia-trading-bot"
cd "$REPO" || exit 1

git add -A
git diff --quiet HEAD && exit 0

DATE=$(date '+%Y-%m-%d %H:%M')
git commit -m "auto: sauvegarde automatique $DATE"
git push
