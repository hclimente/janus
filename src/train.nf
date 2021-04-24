#!/usr/bin/env nextflow

params.data = file('data/boyd_2019')
params.metadata = file('data/boyd_2019_PlateMap-KPP_MOA.xlsx')
params.split = 'crop'
params.csize = 128
params.gpus = 3

train_script = file('train.py')
dropout = [0, 0.05, 0.1, 0.25, 0.5]
margin = [0.001, 0.01, 0.1, 1]

process train {

    tag { "D=$D, M=$M ($I)" }
    publishDir ".", mode: 'copy', overwrite: true

    input:
        each D from dropout
        each I from 1..5
        each M from margin
        val split from params.split
        val csize from params.csize
        file train_script
        file train_data from params.data
        file train_metadata from params.metadata

    output:
        file "sn_*.tsv"
        file "sn_*_100.torch"
//        file 'train_*.pkl'
//        file 'test_*.pkl'

    """
    CUDA_VISIBLE_DEVICES=${(I-1) % Math.min(5, params.gpus)} ./$train_script --dropout $D --margin $M --seed $I --data $train_data --metadata $train_metadata --split $split --csize $csize
    mv train_1.pkl train_1_dropout_${D}_margin_${M}_seed_${I}.pkl
    mv train_2.pkl train_2_dropout_${D}_margin_${M}_seed_${I}.pkl
    mv test_1.pkl  test_1_dropout_${D}_margin_${M}_seed_${I}.pkl
    mv test_2.pkl  test_2_dropout_${D}_margin_${M}_seed_${I}.pkl
    """
}

