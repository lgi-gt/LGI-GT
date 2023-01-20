
# for seeds in 42 314159 271828 2020 2022 2333 23333 12138 666 886 
for seeds in 2333 23333 12138 666 886 
do
  python -u main_segment.py --gconv-dim 256 --tlayer-dim 256 --gconv-dropout 0 --attn-dropout 0 --tlayer-dropout 0.4 --scheduler linear --warmup 5 --lr 0.0002 --readout cls --segment_pooling sum --device 1 --seeds $seeds 
done
