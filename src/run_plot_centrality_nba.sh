#!/bin/bash
arr="random dice sacide structack_dg_comm structack_pr_katz"
for atk in $arr
do
    for rate in 0.05 0.10 0.15 0.20 0.25 0.30
    do
        for seed in 42 0 1 2 100
        do
        python -u Compute_clossness_centrality.py --dataset nba --attack_type ${atk} --ptb_rate ${rate} --seed ${seed}
        done
    done
done


