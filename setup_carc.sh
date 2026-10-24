#!/bin/bash
set -euo pipefail

# CSCI 566 - CARC Shared Storage Setup
SHARED_DIR="/project2/yzhao010_1531/csci566-genimage"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Setting up CSCI 566 CARC directories..."

for dir in data features checkpoints; do
    TARGET="$SHARED_DIR/$dir"
    LINK="$REPO_DIR/$dir"

    if [ ! -d "$TARGET" ]; then
        echo "ERROR: Shared directory not found: $TARGET"
        exit 1
    fi

    if [ -L "$LINK" ]; then
        if [ "$(readlink "$LINK")" = "$TARGET" ]; then
            echo "Already configured: $dir"
        else
            echo "ERROR: $dir points to a different location."
            exit 1
        fi
    elif [ -e "$LINK" ]; then
        echo "ERROR: $dir already exists. Not overwriting."
        exit 1
    else
        ln -s "$TARGET" "$LINK"
        echo "Linked: $dir"
    fi
done

echo "CARC setup complete!"
