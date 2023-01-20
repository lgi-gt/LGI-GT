# for seeds in 42 314159 271828 2020 2022 2333 23333 12138 666 886 
for seeds in 42 314159 271828 2020 2022 2333 23333 12138 666 886 
do
  python -u main_pattern_rw1.py --device 0 --num_rw_steps 7 --dim-pe 16 --tlayer-dropout 0.3 --attn-dropout 0.3 --scheduler none --seeds $seeds 
done
