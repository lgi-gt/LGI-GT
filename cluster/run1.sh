# for seeds in 42 314159 271828 2020 2022 2333 23333 12138 666 886 
for seeds in 42 314159 271828 2020 2022 2333 23333 12138 666 886 
do
  python -u main_rw1.py --device 1 --lr 0.001 --attn-dropout 0.2 --tlayer-dropout 0.2 --scheduler linear --seeds $seeds 
done
