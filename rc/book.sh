#!/bin/bash

[[ -e rc/gen ]] || mkdir rc/gen

for net in 'rnn' 'cnn' 'rnncnn'
do
    # Rough training times were collected:
    # https://docs.google.com/spreadsheets/d/1m7OqgYRY67Thj3HvJ-wWv84Juw4R6oCseppLhLySMJg/edit#gid=0
    # Try sacrificing the 'Adult' predictions in the Crude Humor, Drug, and Nudity categories.
    remove_classes=(" --remove_classes 3" " --remove_classes 3" "" "" " --remove_classes 3" "" "" "")
    if [[ $net == 'rnn' ]]
    then
        plateau_patience=(16 20 4 4 9 6 6 7)
        early_stopping_patience=(22 26 5 6 12 8 8 10)
        epochs=(449 533 109 130 248 176 176 205)
    elif [[ $net == 'cnn' ]]
    then
        plateau_patience=(16 22 14 17 15 11 24 27)
        early_stopping_patience=(25 34 21 26 23 16 36 41)
        epochs=(335 457 289 349 314 220 490 557)
    else  # [[ $net == 'rnncnn' ]]
        plateau_patience=(12 11 2 3 6 6 6 5)
        early_stopping_patience=(16 15 3 4 8 8 8 7)
        epochs=(337 302 78 96 173 176 176 153)
    fi
    for j in {0..8}
    do
        # Use `book_master` as a base file.
        # Slurm doesn't require that scripts be executable.
        f=rc/gen/book_${net}_${j}.sh
        [[ ! -e $f ]] || rm $f
        cp rc/res/book_master.txt $f

        # Append python statement.
        py="python3 -m python.classifiers.book_net ${j}"
        py+=" --net_mode ${net}"
        py+="${remove_classes[j]}"
        py+=" --embedding_trainable"
        py+=" --plateau_patience ${plateau_patience[j]}"
        py+=" --early_stopping_patience ${early_stopping_patience[j]}"
        py+=" --epochs ${epochs[j]}"
        py+=" --save_model"
        echo $py >> $f

        # Send to slurm.
        sbatch -J $net $f
    done
done
