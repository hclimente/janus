for d in 0 0.05 0.1 0.25 0.5
do
    for m in 0.001 0.01 0.1 1 10 100
    do
        for i in {1..5}
        do
            ../../train.py --dropout $d --margin $m --seed $i
        done
    done
done
