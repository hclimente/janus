#!/usr/bin/env nextflow

params.data = file('data/boyd_2019')
params.metadata = file('data/boyd_2019_PlateMap-KPP_MOA.xlsx')
params.split = 'well'
params.gpus = 3

train_script = file('fc_train.py')
params.out = '.'
dropout = [0.1, 0.5]
margin = [0.1, 1]

process train {

    tag { "D=$D, M=$M ($I)" }
    publishDir "$params.out", mode: 'move', overwrite: true

    input:
        each D from dropout
        each I from 1..3
        each M from margin
        val split from params.split
        file train_script
        file train_data from params.data
        file train_metadata from params.metadata

    output:
        // file "sn_*.tsv"
        file "fc_*.ckpt"
        file "tr_seed_${I}.tsv"
        file "te_seed_${I}.tsv"

    """
    ./$train_script --dropout $D --margin $M --seed $I --data $train_data --metadata $train_metadata --split $split --gpus $params.gpus --accelerator ddp --auto_lr_find True
    """
}

