
import torch 
from torch.utils.data import random_split
from sklearn.model_selection import StratifiedKFold
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



def hold_out_gt_parallel(model, dataset, args, current_time): 

    epochs: int = args.epochs 
    # early_stop: int = args.early_stop 
    # patience: int = args.patience 
    batch_size: int = args.batch_size 
    # dim: int = args.dim 
    scheduler_type: str = args.scheduler 
    lr: float = args.lr 
    warmup: int = args.warmup 
    # lr_decay_factor: float = args.lr_decay_factor 
    # lr_decay_step: int = args.lr_decay_step 
    weight_decay: float = args.weight_decay 
    # device: str = "cuda:" + str(args.device) # "cuda:0" 
    device = torch.device("cuda") 

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

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model.to(device) 
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
    if scheduler_type == 'cosine': 
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
        scheduler = get_cosine_schedule_with_warmup( 
                optimizer, warmup*len(train_loader), epochs*len(train_loader)) 
    elif scheduler_type == 'linear': 
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
        scheduler = get_linear_schedule_with_warmup( 
            optimizer, warmup*len(train_loader), epochs*len(train_loader)) 

    p_bar = tqdm(range(0, epochs), bar_format='{l_bar}{bar}| [{n_fmt}/{total_fmt}{postfix}]') 

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter() 

    # evaluator = Evaluator(dataset.name) 
    evaluator = None 
    train_curve = [] 
    val_curve = [] 
    test_curve = [] 

    for epoch in p_bar: 
        train_loss = train_gt(model, optimizer, scheduler, train_loader, device) 
        train_perf, val_perf, test_perf = eval(model, device, train_loader, evaluator), \
                                            eval(model, device, val_loader, evaluator), \
                                            eval(model, device, test_loader, evaluator)
        train_curve.append(train_perf) 
        val_curve.append(val_perf)
        test_curve.append(test_perf) 

        epoch_log = f"| Epoch{epoch: 5d} " 
        train_log = f"| Train e: {train_curve[-1]*100: 5.2f}, T l: {train_loss: 6.4f} " 
        val_log = f"| Val e: {val_curve[-1]*100: 5.2f} " 
        test_log = f"| Test e: {test_curve[-1]*100: 5.2f} " 
        file_log = epoch_log + train_log + val_log + test_log 
        bar_log = train_log + val_log + test_log 
        log_fp.write(file_log + "\n") 
        log_fp.flush() 
        p_bar.set_description(bar_log) # tqdm cannot set multi-line description, so only bar_log 

        # scheduler.step() 

    if torch.cuda.is_available():
            torch.cuda.synchronize()
    end_time = time.perf_counter() 
    run_time = end_time - start_time 

    best_val_epoch = np.argmax(np.array(val_curve))

    best_val = val_curve[best_val_epoch]*100
    test_score = test_curve[best_val_epoch]*100

    # result_log = f"\nRun time: {run_time:5f} | "   \
    last_train = f"Last T e: {train_curve[-1]*100: 5.2f}, T l: {train_loss: 6.4f} | " 
    best_record = f"Best Epoch: {best_val_epoch:3d} | " \
                + f"Best Val: {best_val:5.2f} | "  \
                + f"Test Score: {test_score:5.2f}\n" 
    result_log = "\n" + last_train + best_record 

    record_basic_info(result_fp, current_time_log, args) 
    result_fp.write(result_log) 
    result_fp.flush() 

    print_log = "\n" + best_record 
    print(print_log) 

    if args.save_state: 
            torch.save(model.state_dict(), state_path+'/state_dict.pt') 

    # return get_best_ogb(...) #TODO for model selection 
    
    return result_log 


def hold_out_gt(model, dataset, args): 
    epochs: int = args.epochs 
    # early_stop: int = args.early_stop 
    # patience: int = args.patience 
    batch_size: int = args.batch_size 
    # dim: int = args.dim 
    scheduler_type: str = args.scheduler 
    lr: float = args.lr 
    warmup: int = args.warmup 
    # lr_decay_factor: float = args.lr_decay_factor 
    # lr_decay_step: int = args.lr_decay_step 
    weight_decay: float = args.weight_decay 
    # device: str = "cuda:" + str(args.device) # "cuda:0" 
    device = torch.device("cuda") 

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

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model.to(device) 
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
    if scheduler_type == 'cosine': 
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
        scheduler = get_cosine_schedule_with_warmup( 
                optimizer, warmup*len(train_loader), epochs*len(train_loader)) 
    elif scheduler_type == 'linear': 
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
        scheduler = get_linear_schedule_with_warmup( 
            optimizer, warmup*len(train_loader), epochs*len(train_loader)) 

    p_bar = tqdm(range(0, epochs), bar_format='{l_bar}{bar}| [{n_fmt}/{total_fmt}{postfix}]') 

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter() 

    # evaluator = Evaluator(dataset.name) 
    evaluator = None 
    train_curve = [] 
    val_curve = [] 
    test_curve = [] 

    for epoch in p_bar: 
        train_loss = train_gt(model, optimizer, scheduler, train_loader, device) 
        train_perf, val_perf, test_perf = eval(model, device, train_loader, evaluator), \
                                            eval(model, device, val_loader, evaluator), \
                                            eval(model, device, test_loader, evaluator)
        train_curve.append(train_perf) 
        val_curve.append(val_perf)
        test_curve.append(test_perf) 

        epoch_log = f"| Epoch{epoch: 5d} " 
        train_log = f"| Train e: {train_curve[-1]*100: 5.2f}, T l: {train_loss: 6.4f} " 
        val_log = f"| Val e: {val_curve[-1]*100: 5.2f} " 
        test_log = f"| Test e: {test_curve[-1]*100: 5.2f} " 
        file_log = epoch_log + train_log + val_log + test_log 
        bar_log = train_log + val_log + test_log 
        log_fp.write(file_log + "\n") 
        log_fp.flush() 
        p_bar.set_description(bar_log) # tqdm cannot set multi-line description, so only bar_log 

        # scheduler.step() 

    if torch.cuda.is_available():
            torch.cuda.synchronize()
    end_time = time.perf_counter() 
    run_time = end_time - start_time 

    best_val_epoch = np.argmax(np.array(val_curve))

    best_val = val_curve[best_val_epoch]*100
    test_score = test_curve[best_val_epoch]*100

    # result_log = f"\nRun time: {run_time:5f} | "   \
    last_train = f"Last T e: {train_curve[-1]*100: 5.2f}, T l: {train_loss: 6.4f} | " 
    best_record = f"Best Epoch: {best_val_epoch:3d} | " \
                + f"Best Val: {best_val:5.2f} | "  \
                + f"Test Score: {test_score:5.2f}\n" 
    result_log = "\n" + last_train + best_record 

    record_basic_info(result_fp, current_time_log, args) 
    result_fp.write(result_log) 
    result_fp.flush() 

    print_log = "\n" + best_record 
    print(print_log) 

    if args.save_state: 
            torch.save(model.state_dict(), state_path+'/state_dict.pt') 

    # return get_best_ogb(...) #TODO for model selection 
    
    return result_log 


def hold_out(model, dataset, args): 
    epochs: int = args.epochs 
    early_stop: int = args.early_stop 
    patience: int = args.patience 
    batch_size: int = args.batch_size 
    # dim: int = args.dim 
    lr: float = args.lr 
    lr_decay_factor: float = args.lr_decay_factor 
    lr_decay_step: int = args.lr_decay_step 
    weight_decay: float = args.weight_decay 
    # device: str = "cuda:" + str(args.device) # "cuda:0" 
    device = torch.device("cuda") 

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
    log_fp = open(log_file, 'a') 

    record_basic_info(log_fp, current_time_log, args) 
    log_fp.write("\n\n") 

    # for ogb
    # split_idx = dataset.get_idx_split() 
    # train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)

    # for PROTEINS, NCI1 
    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # model.reset_parameters() 
    model.to(device) 

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor) 

    p_bar = tqdm(range(1, epochs + 1), bar_format='{l_bar}{bar}| [{n_fmt}/{total_fmt}{postfix}]') 

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter() 

    # evaluator = Evaluator(dataset.name) 
    evaluator = None 
    train_curve = [] 
    val_curve = [] 
    test_curve = [] 

    for epoch in p_bar: 
        train_loss = train(model, optimizer, train_loader, device) 
        train_perf, _ = test(model, train_loader, device)
        val_perf, _ = test(model, val_loader, device)
        test_perf, _ = test(model, test_loader, device)
                                            
        train_curve.append(train_perf) 
        val_curve.append(val_perf) 
        test_curve.append(test_perf)  

        epoch_log = f"| Epoch{epoch: 5d} " 
        train_log = f"| Train e: {train_curve[-1]*100: 5.2f}, T l: {train_loss: 6.4f} " 
        val_log = f"| Val e: {val_curve[-1]*100: 5.2f} " 
        test_log = f"| Test e: {test_curve[-1]*100: 5.2f} " 
        file_log = epoch_log + train_log + val_log + test_log 
        bar_log = train_log + val_log + test_log 
        log_fp.write(file_log + "\n") 
        log_fp.flush() 
        p_bar.set_description(bar_log) # tqdm cannot set multi-line description, so only bar_log 

        scheduler.step() 

    if torch.cuda.is_available():
            torch.cuda.synchronize()
    end_time = time.perf_counter() 
    run_time = end_time - start_time 

    best_val_epoch = np.argmax(np.array(val_curve))

    best_val = val_curve[best_val_epoch]*100
    test_score = test_curve[best_val_epoch]*100

    # result_log = f"\nRun time: {run_time:5f} | "   \
    last_train = f"Last T e: {train_curve[-1]*100: 5.2f}, T l: {train_loss: 6.4f} | " 
    best_record = f"Best Epoch: {best_val_epoch:3d} | " \
                + f"Best Val: {best_val:5.2f} | "  \
                + f"Test Score: {test_score:5.2f}\n" 
    result_log = "\n" + last_train + best_record 

    record_basic_info(result_fp, current_time_log, args) 
    result_fp.write(result_log) 
    result_fp.flush() 

    print_log = "\n" + best_record 
    print(print_log) 

    if args.save_state: 
            torch.save(model.state_dict(), state_path+'/state_dict.pt') 

    # return get_best_ogb(...) #TODO for model selection 
    return result_log  


def cross_validation_gt( 
    model, 
    dataset, 
    args 
    ):

    use_val: bool = args.use_val 
    use_val_loss: bool = args.use_val_loss 
    same_best_epoch: bool = args.same_best_epoch 
    epochs: int = args.epochs 
    early_stop: int = args.early_stop 
    patience: int = args.patience 
    batch_size: int = args.batch_size 
    # dim: int = args.dim 
    scheduler_type: str = args.scheduler_type 
    lr: float = args.lr 
    warmup: int = args.warmup 
    # lr_decay_factor: float = args.lr_decay_factor 
    # lr_decay_step: int = args.lr_decay_step 
    weight_decay: float = args.weight_decay 
    # device: str = "cuda:" + str(args.device) # "cuda:0" 
    device = torch.device("cuda") 
    r"""
    if use_val is False, then only consider ... #TODO
    """

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
    # record_basic_info(result_fp, current_time, args) 
    record_basic_info(log_fp, current_time_log, args) 
    
    kf = StratifiedKFold(10, shuffle=True, random_state=args.seeds) 
    # dataset = dataset.shuffle() # matters in some cases 
    labels = np.array([data.y.item() for data in dataset] ) 
    train_val_indices, test_indices = [], [] 
    for train_val_index, test_index in kf.split(np.zeros(len(dataset)), labels ): 
        train_val_indices.append(torch.from_numpy(train_val_index) )
        test_indices.append(torch.from_numpy(test_index) )
    
    val_indices = [test_indices[i-1] for i in range(10)] 

    result = { 
        'best_val_loss': [], # size is (10,) works only use_val 
        'best_val_acc': [], # (10,) works only use_val 
        'best_epoch': [], # (10,) works only use_val 
        'test_acc': [], # (10, epochs) if use_val else (10,) 
        'run_time': [] # (10,) 
    }

    model.to(device) 

    for k in range(10):
        val_index = val_indices[k] 
        test_index = test_indices[k] 
        train_val_index = train_val_indices[k] 
        train_val_list = train_val_index.tolist() 
        val_list = val_index.tolist()
        train_index = [ele for ele in train_val_list 
            if ele not in val_list ] 
        
        train_dataset = dataset[train_index] 
        val_dataset = dataset[val_index] 
        test_dataset = dataset[test_index] 
        train_val_dataset = dataset[train_val_index] 

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True) 
        train_val_loader = DataLoader(train_val_dataset, batch_size, shuffle=True) 
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False) 
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False) 
        
        model.reset_parameters() 
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
        if scheduler_type == 'cosine': 
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup*len(train_loader), epochs*len(train_loader)) 
        elif scheduler_type == 'linear': 
            scheduler = get_linear_schedule_with_warmup(
                optimizer, warmup*len(train_loader), epochs*len(train_loader)) 

        p_bar = tqdm(range(1, epochs + 1), bar_format='{l_bar}{bar}| [{n_fmt}/{total_fmt}{postfix}]') 

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter() 

        # val_acc_list, val_loss_list = [], [] 
        test_acc_list = [] 
        best_val_acc = 0. 
        best_val_loss = 99999999. 
        # best_epoch = -1 
        patience_count = 0 
        for epoch in p_bar: 
            # train_loss, train_acc = None, None 
            # val_acc, val_loss = None, None 
            if use_val:
                train_loss = train_gt(model, optimizer, scheduler, train_loader, device) 
                train_acc, _ = test(model, train_loader, device) 
                val_acc, val_loss = test(model, val_loader, device) 

                if use_val_loss and val_loss < best_val_loss: 
                    best_val_loss = val_loss 
                    # if val_acc > best_val_acc: 
                    #     best_val_acc = val_acc 
                    best_val_acc = val_acc # now best_val_acc reduces to recording the val acc correlated to the best val loss
                    best_epoch = epoch 
                    test_acc, _ = test(model, test_loader, device) 
                    if early_stop: 
                        patience_count = 0 
                elif not use_val_loss and val_acc > best_val_acc: 
                    best_val_acc = val_acc 
                    # if val_loss < best_val_loss: 
                    #     best_val_loss = val_loss 
                    best_val_loss = val_loss # now best_val_loss the variable reduces to recording the val loss correlated to the best acc 
                    best_epoch = epoch 
                    test_acc, _ = test(model, test_loader, device) 
                    if early_stop: 
                        patience_count = 0 
                else: 
                    if early_stop: 
                        patience_count += 1 
            else: 
                train_loss = train(model, optimizer, train_val_loader, device) 
                train_acc, _ = test(model, train_val_loader, device) 
                test_acc, _ = test(model, test_loader, device) 
                test_acc_list.append(test_acc) 
            # scheduler.step()

            fold_log = f"F: {k: 2d} " 
            epoch_log = f"| Epoch{epoch: 5d} " 
            train_log = f"| T a: {train_acc*100: 5.2f}, T l: {train_loss: 6.4f} "
            test_log = f"| E a: {test_acc*100: 5.2f} |" 

            # log = None 
            if use_val: 
                val_log = f"| V a: {val_acc*100: 5.2f}, V l: {val_loss: 6.4f} "
                file_log = fold_log + epoch_log + train_log + val_log + test_log 
                bar_log = fold_log + train_log + val_log + test_log 
            else: 
                file_log = fold_log + epoch_log + train_log + test_log 
                bar_log = fold_log + train_log + test_log 

            log_fp.write(file_log + "\n") 
            log_fp.flush() 
            p_bar.set_description(bar_log) # tqdm cannot set multi-line description, so only bar_log 

            if use_val and early_stop and patience_count >= patience: 
                break # break 'for epoch' 

            # scheduler.step() 

            # end one epoch 
        # end all epochs 

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter() 
        result["run_time"].append(end_time - start_time) 
        
        if use_val: # no matter whether use early stop or not 
            result['best_val_acc'].append(best_val_acc)
            result['best_val_loss'].append(best_val_loss) 
            result['best_epoch'].append(best_epoch) 
            result['test_acc'].append(test_acc) 
        else: 
            result['test_acc'].append(test_acc_list) 

        if args.save_state: 
            torch.save(model.state_dict(), state_path+'/fold_'+str(k)+'.pt') 

        # record_one_fold(result_fp, k, use_val, use_val_loss, 
        #       best_val_acc, best_val_loss, best_epoch, result['test_acc']) 
    # end all folds 

    record_basic_info(result_fp, current_time_log, args) 
    record_all_folds(result_fp, result, use_val, use_val_loss, same_best_epoch) 
    return get_best(result, use_val, use_val_loss, same_best_epoch) 
# end 10-fold cross validation 


def cross_validation( 
    model, 
    dataset, 
    args 
    ):

    use_val: bool = args.use_val 
    use_val_loss: bool = args.use_val_loss 
    same_best_epoch: bool = args.same_best_epoch # work when args.use_val is False 
    epochs: int = args.epochs 
    early_stop: int = args.early_stop 
    patience: int = args.patience 
    batch_size: int = args.batch_size 
    # dim: int = args.dim 
    lr: float = args.lr 
    lr_decay_factor: float = args.lr_decay_factor 
    lr_decay_step: int = args.lr_decay_step 
    weight_decay: float = args.weight_decay 
    # device: str = "cuda:" + str(args.device) # "cuda:0" 
    device = torch.device("cuda") 
    r"""
    if use_val is False, then only consider ... #TODO
    """

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
    # record_basic_info(result_fp, current_time, args) 
    record_basic_info(log_fp, current_time_log, args) 
    
    kf = StratifiedKFold(10, shuffle=True, random_state=args.seeds) 
    # dataset = dataset.shuffle() # matters in some cases 
    labels = np.array([data.y.item() for data in dataset] ) 
    train_val_indices, test_indices = [], [] 
    for train_val_index, test_index in kf.split(np.zeros(len(dataset)), labels ): 
        train_val_indices.append(torch.from_numpy(train_val_index) ) 
        test_indices.append(torch.from_numpy(test_index) ) 
    
    val_indices = [test_indices[i-1] for i in range(10)] 

    result = { 
        'best_val_loss': [], # size is (10,) works only use_val 
        'best_val_acc': [], # (10,) works only use_val 
        'best_epoch': [], # (10,) works only use_val 
        'test_acc': [], # (10, epochs) if use_val else (10,) 
        'run_time': [] # (10,) 
    }

    model.to(device) 

    for k in range(10):
        val_index = val_indices[k] 
        test_index = test_indices[k] 
        train_val_index = train_val_indices[k] 
        train_val_list = train_val_index.tolist() 
        val_list = val_index.tolist()
        train_index = [ele for ele in train_val_list 
            if ele not in val_list ] 
        
        train_dataset = dataset[train_index] 
        val_dataset = dataset[val_index] 
        test_dataset = dataset[test_index] 
        train_val_dataset = dataset[train_val_index] 

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True) 
        train_val_loader = DataLoader(train_val_dataset, batch_size, shuffle=True) 
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False) 
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False) 
        
        model.reset_parameters() 
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
        scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor) 

        p_bar = tqdm(range(1, epochs + 1), bar_format='{l_bar}{bar}| [{n_fmt}/{total_fmt}{postfix}]') 

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter() 

        # val_acc_list, val_loss_list = [], [] 
        test_acc_list = [] 
        best_val_acc = 0. 
        best_val_loss = 99999999. 
        # best_epoch = -1 
        patience_count = 0 
        for epoch in p_bar: 
            # train_loss, train_acc = None, None 
            # val_acc, val_loss = None, None 
            if use_val:
                train_loss = train(model, optimizer, train_loader, device) 
                train_acc, _ = test(model, train_loader, device) 
                val_acc, val_loss = test(model, val_loader, device) 

                if use_val_loss and val_loss < best_val_loss: 
                    best_val_loss = val_loss 
                    # if val_acc > best_val_acc: 
                    #     best_val_acc = val_acc 
                    best_val_acc = val_acc # now best_val_acc reduces to recording the val acc correlated to the best val loss
                    best_epoch = epoch 
                    test_acc, _ = test(model, test_loader, device) 
                    if early_stop: 
                        patience_count = 0 
                elif not use_val_loss and val_acc > best_val_acc: 
                    best_val_acc = val_acc 
                    # if val_loss < best_val_loss: 
                    #     best_val_loss = val_loss 
                    best_val_loss = val_loss # now best_val_loss the variable reduces to recording the val loss correlated to the best acc 
                    best_epoch = epoch 
                    test_acc, _ = test(model, test_loader, device) 
                    if early_stop: 
                        patience_count = 0 
                else: 
                    if early_stop: 
                        patience_count += 1 
            else: 
                train_loss = train(model, optimizer, train_val_loader, device) 
                train_acc, _ = test(model, train_val_loader, device) 
                test_acc, _ = test(model, test_loader, device) 
                test_acc_list.append(test_acc) 

            fold_log = f"F: {k: 2d} " 
            epoch_log = f"| Epoch{epoch: 5d} " 
            train_log = f"| T a: {train_acc*100: 5.2f}, T l: {train_loss: 6.4f} "
            test_log = f"| E a: {test_acc*100: 5.2f} |" 

            # log = None 
            if use_val: 
                val_log = f"| V a: {val_acc*100: 5.2f}, V l: {val_loss: 6.4f} "
                file_log = fold_log + epoch_log + train_log + val_log + test_log 
                bar_log = fold_log + train_log + val_log + test_log 
            else: 
                file_log = fold_log + epoch_log + train_log + test_log 
                bar_log = fold_log + train_log + test_log 

            log_fp.write(file_log + "\n") 
            log_fp.flush() 
            p_bar.set_description(bar_log) # tqdm cannot set multi-line description, so only bar_log 

            if use_val and early_stop and patience_count >= patience: 
                break # break 'for epoch' 

            scheduler.step() 
            # end one epoch 
        # end all epochs 

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter() 
        result["run_time"].append(end_time - start_time) 
        
        if use_val: # no matter whether use early stop or not 
            result['best_val_acc'].append(best_val_acc)
            result['best_val_loss'].append(best_val_loss) 
            result['best_epoch'].append(best_epoch) 
            result['test_acc'].append(test_acc) 
        else: 
            result['test_acc'].append(test_acc_list) 

        if args.save_state: 
            torch.save(model.state_dict(), state_path+'/fold_'+str(k)+'.pt') 

        # record_one_fold(result_fp, k, use_val, use_val_loss, 
        #       best_val_acc, best_val_loss, best_epoch, result['test_acc']) 
    # end all folds 

    record_basic_info(result_fp, current_time_log, args) 
    record_all_folds(result_fp, result, use_val, use_val_loss, same_best_epoch) 
    return get_best(result, use_val, use_val_loss, same_best_epoch) 
# end 10-fold cross validation 


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
        elif key in ('use_val_loss', 'early_stop') and not args.use_val: 
            continue 
        elif key == 'patience' and (not args.use_val or not args.early_stop):
            continue 
        else: 
            fp.write("\n"+key+": "+str(value)) 
    fp.write("\n\n") 
    fp.flush() 


def record_one_fold(result_fp, k, use_val, use_val_loss, 
                        best_val_acc, best_val_loss, best_epoch, test_acc) : 
    result_fp.write(f"Fold {k: 2d}: ") 
    if use_val: # either val_acc or val_loss 
        if use_val_loss: 
            result_fp.write(f"Best Val Loss {best_val_loss:6.4f}, "
                            f"Epoch {best_epoch:4d}, "
                            f"Test Acc {test_acc*100:5.2f}\n") 
        else: 
            result_fp.write(f"Best Val Acc {best_val_acc*100:5.2f}, "
                            f"Epoch {best_epoch:4d}, "
                            f"Test Acc {test_acc*100:5.2f}\n") 
    else: 
        test_acc_one_fold = torch.tensor(test_acc) 
        test_acc_value, test_acc_best_epoch = test_acc_one_fold.max(dim=0) 
        result_fp.write(f"Best Test Acc {test_acc_value.item()*100:5.2f}, "
                        f"Epoch {test_acc_best_epoch.item():4d}\n") 
    result_fp.flush() 


def get_best(result, use_val, use_val_loss, same_best_epoch): # for model selection 
    test_acc = torch.tensor(result["test_acc"]) 
    if use_val: 
        best_val_acc, best_val_loss  = \
        torch.tensor(result["best_val_acc"]), torch.tensor(result["best_val_loss"]) 
        return best_val_acc.mean().item()*100, best_val_loss.mean().item(), test_acc.mean().item()*100, test_acc.std().item()*100  
    else: # NOTE that it's wrong to tune hyper-parameters one test set directly :) 
        if same_best_epoch: # like GIN did 
            test_acc_mean = test_acc.mean(dim=0)  
            best_test_acc, best_epoch = test_acc_mean.max(dim=0) 
            return None, None, best_test_acc.item()*100, test_acc[:, best_epoch].std().item()*100 
        else: 
            test_acc_value, test_acc_best_epoch = test_acc.max(dim=1) 
            return None, None, test_acc_value.mean().item()*100, test_acc_value.std().item()*100 


def record_all_folds(result_fp, result, use_val, use_val_loss, same_best_epoch): 
    run_time = torch.tensor(result["run_time"]) 
    result_fp.write(f"\nAverage Time: {run_time.mean().item(): .2f} seconds\n") 
    best_val_acc, best_val_loss, \
    result_test_acc_mean, result_test_acc_std \
        = get_best(result, use_val, use_val_loss, same_best_epoch) 
    result_log = None 
    if use_val: 
        for k in range(10): 
            record_one_fold(result_fp, k, use_val, use_val_loss,
                result["best_val_acc"][k], result["best_val_loss"][k], 
                result['best_epoch'][k], result["test_acc"][k]) 
        if use_val_loss: 
            result_log = f"Average Best Val Loss: {best_val_loss:6.4f}\n" \
                         f"Result Test Acc: {result_test_acc_mean:5.2f} ± {result_test_acc_std:5.2f}\n"
        else: 
            result_log = f"Average Best Val Acc: {best_val_acc:5.2f}\n" \
                         f"Result Test Acc: {result_test_acc_mean:5.2f} ± {result_test_acc_std:5.2f}\n"
        # result_fp.write(f"Average Best Val Acc {best_val_acc:8.4f}\n" 
        #                 f"Result Test Acc {result_test_acc_mean:5.2f} ± {result_test_acc_std:5.2f}\n") 
    else: 
        for k in range(10): 
            record_one_fold(result_fp, k, use_val, use_val_loss, 
                None, None, None, result["test_acc"][k]) 
        if same_best_epoch: # like GIN did 
            result_log = f"Best Average Test Acc: {result_test_acc_mean:5.2f} " \
                            f"± {result_test_acc_std:5.2f}\n"  
                        # f"Best Epoch: {best_epoch.item():4d}\n") 
        else: 
            result_log = f"Average Best Test Acc: {result_test_acc_mean:5.2f} " \
                            f"± {result_test_acc_std:5.2f}\n" 
    result_fp.write(result_log) 
    result_fp.flush() 

    print(result_log) 


def construct_model(args): # only if models are build here 
    # TODO or never do 
    pass 


def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0 
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device) 
        out = model(data) 
        loss = F.cross_entropy(out, data.y) 
        loss.backward() 
        optimizer.step() 
        total_loss += loss.item() * data.num_graphs 
        
    return total_loss / len(loader.dataset) 


def train_ogb(model, optimizer, loader, device):
    model.train()

    total_loss = 0 
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device) 
        # if data.x is None:
        #     data.x = torch.ones((data.batch.shape[0], 1), device=device) # for data whose x is NONE
        out = model(data) 
        loss = F.binary_cross_entropy_with_logits(out, data.y.to(torch.float)) 
        loss.backward() 
        optimizer.step() 
        total_loss += loss.item() * data.num_graphs 
        
    return total_loss / len(loader.dataset)


def train_gt(model, optimizer, scheduler, loader, device):
    model.train() 

    total_loss = 0 
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device) 
        # if data.x is None:
        #     data.x = torch.ones((data.batch.shape[0], 1), device=device) # for data whose x is NONE
        out = model(data) 
        loss = F.cross_entropy(out, data.y) 
        loss.backward() 
        optimizer.step() 
        scheduler.step() 
        total_loss += loss.item() * data.num_graphs 
        
    return total_loss / len(loader.dataset) 


def train_ogb_gt(model, optimizer, scheduler, loader, device):
    model.train()

    total_loss = 0 
    for data in loader:
        optimizer.zero_grad() 
        data = data.to(device) 
        # if data.x is None:
        #     data.x = torch.ones((data.batch.shape[0], 1), device=device) # for data whose x is NONE
        out = model(data) 
        loss = F.binary_cross_entropy_with_logits(out, data.y.to(torch.float)) 
        loss.backward() 
        optimizer.step() 
        scheduler.step() 
        total_loss += loss.item() * data.num_graphs 
        
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader, device): 
    model.eval()

    total_correct = 0 
    total_loss = 0
    for data in loader:
        data = data.to(device) 
        # if data.x is None:
        #     data.x = torch.ones((data.batch.shape[0], 1), device=device) # for data whose x is NONE 

        # out = model.forward_downstream(data.x, data.edge_index, data.batch) 
        # out = model(data.x, data.edge_index, data.batch) 
        out = model(data) 
        total_correct += int((out.argmax(-1) == data.y).sum()) 
        total_loss += F.cross_entropy(out, data.y).item() * data.num_graphs 
    return total_correct / len(loader.dataset), total_loss / len(loader.dataset)  


# @torch.no_grad() 
# def eval(model, device, loader, evaluator):
#     model.eval()
#     y_true = []
#     y_pred = []

#     # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
#     for batch in loader:
#         batch = batch.to(device)

#         if batch.x.shape[0] == 1:
#             pass
#         else:
#             with torch.no_grad():
#                 pred = model(batch)

#             y_true.append(batch.y.view(pred.shape).detach().cpu())
#             y_pred.append(pred.detach().cpu())

#     y_true = torch.cat(y_true, dim = 0).numpy()
#     y_pred = torch.cat(y_pred, dim = 0).numpy()

#     input_dict = {"y_true": y_true, "y_pred": y_pred}

#     return evaluator.eval(input_dict)


@torch.no_grad()
def eval(model, device, loader, evaluator):
    model.eval()

    correct = 0
    for step, batch in enumerate(loader): 
        batch = batch.to(device)

        pred = model(batch)
        pred = pred.max(dim=1)[1]
        correct += pred.eq(batch.y).sum().item()
    return correct / len(loader.dataset) 
