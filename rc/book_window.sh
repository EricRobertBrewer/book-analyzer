#!/bin/bash

[[ -e rc/gen ]] || mkdir rc/gen

classifier_names = ("paragraph_max_ordinal" "paragraph_rnn_max_ordinal" "paragraph_rnncnn_max_ordinal")
model_file_names=(
  ("35082769_0.h5" "35082760_trainemb_1.h5" "35082771_2.h5" "35082762_trainemb_3.h5" "35082763_trainemb_4.h5" "35082764_trainemb_5.h5" "35082765_trainemb_6.h5" "35082776_7.h5")
  ("36263810_0.h5" "36263811_1.h5" "36263812_2.h5" "36263813_3.h5" "36263814_4.h5" "36263815_5.h5" "36263816_6.h5" "36263817_7.h5")
  ("36263819_0.h5" "36263820_1.h5" "36263821_2.h5" "36263822_3.h5" "36263823_4.h5" "36263824_5.h5" "36263825_6.h5" "36263826_7.h5")
)

for i in {0..2}
do
    for j in {0..7}
    do
        for size in (1 3 5 7)
        do
            f=rc/gen/${classifier_names[i]}_${j}_${size}w.sh
            [[ ! -e $f ]] || rm $f
            cp rc/res/book_window_master.txt $f

            py="python3 -m python.classifiers.book_net_window ${classifier_names[i]} ${model_file_names[i][j]} ${size}"
            echo $py >> $f

            sbatch -J "b_${i}_${j}_${size}" $f
        done
    done
done
