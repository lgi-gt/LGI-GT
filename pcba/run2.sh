
# for seeds in 42 314159 271828 2020 2022 2333 23333 12138 666 886 
for seeds in 12138 666 
do
  python -u main_gps_parallel_gat.py --seeds $seeds --device 2 
done
