#!/bin/bash

[[ -e rc/gen ]] || mkdir rc/gen

# Rough training times were collected:
# https://docs.google.com/spreadsheets/d/1m7OqgYRY67Thj3HvJ-wWv84Juw4R6oCseppLhLySMJg/edit#gid=0
# Try sacrificing the 'Adult' predictions in the Crude Humor, Drug, and Nudity categories.
remove_classes=(" --remove_classes 3" " --remove_classes 3" "" "" " --remove_classes 3" "" "" "")
plateau_patience=(16 22 14 17 15 11 24 27)
early_stopping_patience=(25 34 21 26 23 16 36 41)
epochs=(335 457 289 349 314 220 490 557)

for j in {0..7}
do
    # Use `book_master.txt` as a base file.
    # Slurm doesn't require that scripts be executable.
    f=rc/gen/book_${j}.sh
    [[ ! -e $f ]] || rm $f
    cp rc/res/book_master.txt $f

    # Append python statement.
    py="python3 -m python.classifiers.book_net ${j}"
    py+=" --remove_stopwords"
    py+="${remove_classes[j]}"
    py+=" --embedding_trainable"
    py+=" --plateau_patience ${plateau_patience[j]}"
    py+=" --early_stopping_patience ${early_stopping_patience[j]}"
    py+=" --epochs ${epochs[j]}"
    py+=" --save_model"
    py+=" --note nostop-trainemb"
    echo $py >> $f

    # Send to slurm.
    sbatch -J "book_${j}" $f
done
