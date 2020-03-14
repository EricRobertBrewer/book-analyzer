#!/bin/bash

[[ -e rc/gen ]] || mkdir rc/gen

for net in 'rnn' 'cnn' 'rnncnn'
do
    # Rough training times were collected:
    # https://docs.google.com/spreadsheets/d/1m7OqgYRY67Thj3HvJ-wWv84Juw4R6oCseppLhLySMJg/edit#gid=0
    if [[ $net == 'rnn' ]]
    then
        plateau_patience=(16 20 4 4 9 6 6 7)
        early_stopping_patience=(22 26 5 6 12 8 8 10)
        epochs=(449 533 109 130 248 176 176 205)
    elif [[ $net == 'cnn' ]]
    then
        plateau_patience=(42 27 11 12 22 8 18 20)
        early_stopping_patience=(56 36 14 17 30 11 24 27)
        epochs=(1134 736 296 341 611 225 480 544)
    else  # [ $net == 'rnncnn' ]
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
        py+=" --epochs ${epochs[j]}"
        py+=" --plateau_patience ${plateau_patience[j]}"
        py+=" --early_stopping_patience ${early_stopping_patience[j]}"
        echo $py >> $f

        # Send to slurm.
        sbatch -J $net $f
    done
done
