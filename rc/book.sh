#!/bin/bash

[[ -e rc/gen ]] || mkdir rc/gen

for net in 'rnn' 'cnn' 'rnncnn'
do
    for i in {0..8}
    do
        f=rc/gen/book_${net}_${i}.sh
        [[ ! -e $f ]] || rm $f
        cp rc/res/book_master.txt $f
        echo "python3 -m python.classifiers.book_net ${i} --net_mode ${net}" >> $f
        sbatch -J $net $f
    done
done
