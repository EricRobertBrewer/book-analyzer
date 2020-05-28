#!/bin/bash

[[ -e rc/gen ]] || mkdir rc/gen

model_file_names=("35082769_0.h5" "35082760_trainemb_1.h5" "35082771_2.h5" "35082762_trainemb_3.h5" "35082763_trainemb_4.h5" "35082764_trainemb_5.h5" "35082765_trainemb_6.h5" "35082776_7.h5")

for j in {0..7}
do
    f=rc/gen/book_window_${j}.sh
    [[ ! -e $f ]] || rm $f
    cp rc/res/book_window_master.txt $f

    py="python3 -m python.classifiers.book_net_window ${model_file_names[j]} 1"
    echo $py >> $f

    sbatch -J "bookw_${j}" $f
done
