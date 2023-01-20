


# for seeds in 42 314159 271828 2020 2022 2333 23333 12138 666 886 
for seeds in 42 314159 271828 2020 2022 
do
  python -u main_clip.py --gconv-dim 256 --tlayer-dim 256 --gconv-dropout 0 --attn-dropout 0 --tlayer-dropout 0.4 --scheduler linear --warmup 5 --lr 0.0002 --readout cls --clipping_mode tail --device 2 --seeds $seeds 
done


