

import torch 
import numpy as np 
import os 
import time 
from tqdm import tqdm 

import torch.nn.functional as F 
from torch.optim import Adam, AdamW 
from torch.optim.lr_scheduler import StepLR 

# from torch_geometric.data.dataloader import DataLoader # for pyg == 1.7.0 
from torch_geometric.loader import DataLoader # for pyg == 2.0.4 

from ogb.graphproppred import Evaluator 

from transformers.optimization import get_cosine_schedule_with_warmup,\
    get_linear_schedule_with_warmup 

from utils import decode_arr_to_seq 


def hold_out(model, dataset, args, idx2vocab): 
    epochs: int = args.epochs 
    lr: float = args.lr 
    warmup: int = args.warmup 
    weight_decay: float = args.weight_decay 
    # device: str = "cuda:" + str(args.device) # "cuda:0" 
    device = torch.device('cpu') if args.use_cpu else torch.device('cuda') 

    exp_result_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '.', 'exp_result') 
    if not os.path.exists(exp_result_path): 
        os.makedirs(exp_result_path) 
    
    log_path = os.path.join(exp_result_path, 'log') 
    if not os.path.exists(log_path): 
        os.makedirs(log_path) 
    
    result_file = os.path.join(exp_result_path, dataset.name + '.txt') 
    current_time = time.localtime() 

    current_time_log = time.strftime("%Y-%m-%d %H:%M:%S", current_time) 
    current_time_filename =  time.strftime("%Y-%m-%d_%H%M%S", current_time) 
    log_file = os.path.join(log_path, dataset.name + '_' + current_time_filename + '.log') 

    if args.save_state: 
        state_path = os.path.join(exp_result_path, 'state', type(model).__name__, dataset.name+'_'+current_time_filename) 
        if not os.path.exists(state_path): 
            os.makedirs(state_path) 

    result_fp = open(result_file, 'a') 
    log_fp = open(log_file, 'w') 

    record_basic_info(log_fp, current_time_log, args) 
    log_fp.write("\n\n") 

    model.to(device) 
    print(f'#Params: {sum(p.numel() for p in model.parameters())}') 

    evaluator = Evaluator(dataset.name) 

    split_idx = dataset.get_idx_split() 

    if args.baseline == 'SAT': 
        filter_mask = np.array([dataset[i].num_nodes for i in split_idx['train']]) <= 1000
        train_set = dataset[split_idx["train"]][filter_mask] 
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers) 

        filter_mask = np.array([dataset[i].num_nodes for i in split_idx['valid']]) <= 1000 
        val_set = dataset[split_idx['valid'][filter_mask]] 
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers) 

        filter_mask = np.array([dataset[i].num_nodes for i in split_idx['test']]) <= 1000 
        test_set = dataset[split_idx['test'][filter_mask]] 
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers) 

    elif args.baseline == 'GraphTrans': 
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers) 
        val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers) 
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers) 

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 

    if args.scheduler == 'cosine': 
        scheduler = get_cosine_schedule_with_warmup( 
            optimizer, warmup*len(train_loader), epochs*len(train_loader)) 
    elif args.scheduler == 'linear': 
        scheduler = get_linear_schedule_with_warmup( 
            optimizer, warmup*len(train_loader), epochs*len(train_loader)) 
    elif args.scheduler is None: 
        scheduler = None 

    # p_bar = tqdm(range(0, epochs), bar_format='{l_bar}{bar}| [{n_fmt}/{total_fmt}{postfix}]') 

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter() 

    train_curve = [] 
    val_curve = [] 
    test_curve = [] 

    # for epoch in p_bar: 
    for epoch in range(0, epochs): 
        train_loss = train(model, device, train_loader, optimizer, scheduler) 

        train_perf = eval(model, device, train_loader, evaluator, arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab), desc="Eval Train ")
        val_perf = eval(model, device, val_loader, evaluator, arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab), desc="Eval   Val ")
        test_perf = eval(model, device, test_loader, evaluator, arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab), desc="Eval  Test ") 

        train_curve.append(train_perf[dataset.eval_metric]) 
        val_curve.append(val_perf[dataset.eval_metric]) 
        test_curve.append(test_perf[dataset.eval_metric]) 

        epoch_log = f"| Epoch{epoch: 5d} " 
        train_log = f"| Train e: {train_curve[-1]: 6.4f}, T l: {train_loss: 6.4f} " 
        val_log = f"| Val e: {val_curve[-1]: 6.4f} " 
        test_log = f"| Test e: {test_curve[-1]: 6.4f} " 
        file_log = epoch_log + train_log + val_log + test_log 
        # bar_log = train_log + val_log + test_log 
        log_fp.write(file_log + "\n") 
        log_fp.flush() 
        # p_bar.set_description(bar_log) # tqdm cannot set multi-line description, so only bar_log 
        print(file_log) 

        # scheduler.step() 

    if torch.cuda.is_available():
            torch.cuda.synchronize()
    end_time = time.perf_counter() 
    run_time = end_time - start_time 

    best_val_epoch = np.argmax(np.array(val_curve)) 
    best_val = val_curve[best_val_epoch]

    test_score = test_curve[best_val_epoch] 

    result_log = (f"\nRun time: {run_time}\n"  
                 f"Best Epoch: {best_val_epoch}\n" 
                 f"Val: {best_val:6.4f}\n"  
                 f"Test Score: {test_score:6.4f}\n\n" ) 

    record_basic_info(result_fp, current_time_log, args) 
    result_fp.write(result_log) 
    result_fp.flush() 

    print(result_log) 

    if args.save_state: 
            torch.save(model.state_dict(), state_path+'/state_dict.pt') 

    # return get_best_ogb(...) #TODO for model selection 

def record_basic_info(fp, current_time, args): 
    fp.write("\n\n===============================================\n") 
    fp.write(current_time) 
    fp.write("\n") 
    fp.write("\n" + args.cmd_str + '\n') 
    for key, value in args.__dict__.items(): 
        if key == "configs": # keep configs as list of my own data structure Config
            for config in value: 
                fp.write("\n" + str(config)) 
        elif key == "info":
            fp.write("\n" + str(value) + "\n") 
        elif key == "config": # just config file stream string
            fp.write("\n" + value) 
        elif key == "cmd_str" or key == 'device': 
            continue 
        else: 
            fp.write("\n"+key+": "+str(value)) 
    fp.write("\n\n") 
    fp.flush() 

def get_best(result, use_val, use_val_loss, same_best_epoch): # for model selection 
    test_score = torch.tensor(result["test_score"]) 
    if use_val: 
        best_val_score, best_val_loss  = \
        torch.tensor(result["best_val_score"]), torch.tensor(result["best_val_loss"]) 
        return best_val_score.mean().item()*100, best_val_loss.mean().item(), test_score.mean().item()*100, test_score.std().item()*100  
    else: # NOTE that it's wrong to tune hyper-parameters one test set directly :) 
        if same_best_epoch: # like GIN did 
            test_score_mean = test_score.mean(dim=0)  
            best_test_score, best_epoch = test_score_mean.max(dim=0) 
            return None, None, best_test_score.item()*100, test_score[:, best_epoch].std().item()*100 
        else: 
            test_score_value, test_score_best_epoch = test_score.max(dim=1) 
            return None, None, test_score_value.mean().item()*100, test_score_value.std().item()*100 


def train(model, device, loader, optimizer, scheduler): 
    model.train() 
    # p_bar = tqdm(loader, desc="Train Iteration") 
    loss_accum = 0
    # for step, batch in enumerate(p_bar):
    for step, batch in enumerate(loader): # for log print into file instead of terminal while using nohup 
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred_list = model(batch)
            optimizer.zero_grad()

            loss = 0
            for i in range(len(pred_list)):
                loss += F.cross_entropy(pred_list[i].to(torch.float32), batch.y_arr[:,i])

            loss = loss / len(pred_list)
            
            loss.backward()
            optimizer.step()

            loss_accum += loss.item() 
            # p_bar.set_description(f"Train (loss = {loss.item():.4f}, smoothed = {loss_accum / (step + 1):.4f})") 
            if step % 100 == 0: # for log print into file instead of terminal while using nohup 
                print(f"Train (loss = {loss.item():.4f}, smoothed = {loss_accum / (step + 1):.4f})") 

        if scheduler:
            scheduler.step() 

    # print('Average training loss: {}'.format(loss_accum / (step + 1)))
    return loss_accum / (step + 1) 

def eval(model, device, loader, evaluator, arr_to_seq, desc):
    model.eval()
    seq_ref_list = []
    seq_pred_list = []
    print(desc+"...") # for log print into file instead of terminal while using nohup
    # p_bar = tqdm(loader, desc=desc+"| Iteration") 
    # for step, batch in enumerate(p_bar): 
    for step, batch in enumerate(loader): # for log print into file instead of terminal while using nohup 
        batch = batch.to(device) 

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred_list = model(batch)

            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim = 1).view(-1,1))
            mat = torch.cat(mat, dim = 1)
            
            seq_pred = [arr_to_seq(arr) for arr in mat] 
            
            # PyG = 1.4.3
            # seq_ref = [batch.y[i][0] for i in range(len(batch.y))]

            # PyG >= 1.5.0
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

    return evaluator.eval(input_dict)