# run=0
# time=`date +%Y-%m-%d_%H%M%S

# 1 
# sleep 3h

# for seeds in 2333 23333 12138 666 886 
for seeds in 42 314159 271828 2020 2022 
do
  nohup python main_parallel.py --readout mean --gconv-dim 256 --tlayer-dim 256 --num-heads 8 --no-clustering --masked-attention --batch-size 32 --epochs 200 --warmup 20 --gconv-dropout 0.5 --tlayer-dropout 0.5 --seeds $seeds --device 1 > /dev/null 2>&1 & 
  
  sleep 5s
  # echo $run 
#   ((run=$run+1)) 
done
