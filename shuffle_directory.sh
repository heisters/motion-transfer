#!/bin/sh -e

shufdir=../train_B_shuffled  # directory at this path will be deleted and recreated

rm -rf "$shufdir"
mkdir -p "$shufdir"

printf '%s\n' "$@" | shuf |
while read -r fname; do
    ln -s "$( readlink -f "$1" )" "$shufdir/$fname"
    shift
done

