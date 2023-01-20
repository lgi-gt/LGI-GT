
# for seeds in 42 314159 271828 2020 2022 2333 23333 12138 666 886 
for seeds in 2333 23333 12138 666 886 
do
  python -u main_gps_parallel_transformerconv.py --seeds $seeds --device 0 
done
