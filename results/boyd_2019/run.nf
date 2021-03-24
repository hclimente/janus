#!/usr/bin/env nextflow

params.data = file('../../data/boyd_2019')
params.metadata = file('../../data/boyd_2019_PlateMap-KPP_MOA.xlsx')

train_script = file('../../train.py')
dropout = [0, 0.05, 0.1, 0.25, 0.5]
margin = [0.001, 0.01, 0.1, 1]

process train {

    tag { "D=$D, M=$M ($I)" }
    publishDir ".", overwrite: true, mode: "copy"

    input:
        each D from dropout
        each I from 1..5
        each M from margin
        file train_script
        file train_data from params.data
        file train_metadata from params.metadata

    output:
        file "sn_*.tsv"

    """
    CUDA_VISIBLE_DEVICES=${I % 3 + 1} ./$train_script --dropout $D --margin $M --seed $I --data $train_data --metadata $train_metadata
    """
}

