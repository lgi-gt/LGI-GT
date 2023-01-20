
# sleep 1h # 

# parser.add_argument('--epochs', type=int, default=200) 
# parser.add_argument('--warmup', type=int, default=10) 
# parser.add_argument('--weight-decay', type=float, default=1e-4) 
# parser.add_argument('--transformer-dropout', type=float, default=0.1) 

# hidden_dim=128, num_layers=4, ginconv dropout ratio=0.0, 
# dropout_ratio between ginconv and transformer = 0.5, all LayerNorm, no BatchNorm1d, 
# readout=mean, no relu between ginconv and transformer 
# long skip connection to the output


for seeds in 42 314159 271828 2020 2022 
do
  nohup python main_parallel.py --batch-size 64 --lr 0.0001 --warmup 50 --scheduler-type linear --seeds $seeds --device 5 > /dev/null 2>&1 &  
  sleep 5s 
done


