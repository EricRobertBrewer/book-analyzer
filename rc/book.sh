#!/bin/bash

[[ -e rc/gen ]] || mkdir rc/gen

# Rough training times were collected:
# https://docs.google.com/spreadsheets/d/1m7OqgYRY67Thj3HvJ-wWv84Juw4R6oCseppLhLySMJg/edit#gid=0
plateau_patience=(57 69 14 17 32 11 24 27)
early_stopping_patience=(86 103 21 26 48 16 36 41)
epochs=(1153 1382 289 349 640 220 490 557)

for j in {0..7}
do
    # Use `book_master.txt` as a base file.
    # Slurm doesn't require that scripts be executable.
    f=rc/gen/book_${j}.sh
    [[ ! -e $f ]] || rm $f
    cp rc/res/book_master.txt $f

    # Append python statement.
    py="python3 -m python.classifiers.book_net ${j}"
    py+=" --epochs ${epochs[j]}"
    py+=" --plateau_patience ${plateau_patience[j]}"
    py+=" --early_stopping_patience ${early_stopping_patience[j]}"
    echo $py >> $f

    # Send to slurm.
    sbatch -J "book_${j}" $f
done
