# for seeds in 42 314159 271828 2020 2022 2333 23333 12138 666 886 
for seeds in 42 314159 271828 2020 2022 2333 23333 12138 666 886 
do
  python -u main_rw1.py --device 0 --attn-dropout 0.5 --tlayer-dropout 0.1 --lr 0.001 --num-heads 8 --scheduler cosine --seeds $seeds 
done
