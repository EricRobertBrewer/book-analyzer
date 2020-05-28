#!/bin/bash

[[ -e rc/gen ]] || mkdir rc/gen

for model in "k_nearest_neighbors" "logistic_regression" "multi_layer_perceptron" "multinomial_naive_bayes" "random_forest" "svm"
do
    f=rc/gen/baselines_window_${model}.sh
    [[ ! -e $F ]] || rm $f
    cp rc/res/baselines_window_master.txt $f

    py="python3 -m python.classifiers.baselines_window ${model} 36100418 1"
    echo $py >> $f

    sbatch -J "bw_${model}" $f
done
