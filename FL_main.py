import os
import copy
import time
import numpy as np
from tqdm import tqdm
import torch
from options import args_parser
from client import client
from utils import get_dataset, average_weights, exp_details, Initialize_Model, plot_dis
from opacus.validators import ModuleValidator
from opacus import GradSampleModule

if __name__ == '__main__':
    np.random.seed(1)
    args = args_parser()
    start_time = time.time()
    exp_details(args)
    data_path = '..\data'
    save_path = '..\save'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, '{}_{}_C[{}]_F[{}]_iid[{}]_Epoch[{}]_Lep[{}]_B[{}]_DP[{}]_SE[{}]_STime[{'
                                        '}]_TAcc[{}]_MinZ[{}].npy'. format(args.dataset, args.model, args.num_clients, args.frac, args.iid, args.epochs,
                                    args.local_ep, args.local_bs, args.is_dp, int(args.se_game), args.start_time,
                                    args.target_acc, args.min_z))
    # BUILD MODEL
    if args.is_dp:
        global_model = GradSampleModule(Initialize_Model(args))
        errors = ModuleValidator.validate(global_model, strict=False)
        if not errors:
            global_model = ModuleValidator.fix(global_model)
    else:
        global_model = Initialize_Model(args)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    global_weights = global_model.state_dict()
    # Training
    train_loss, avg_test_accuracy = [], []
    sample_count = [[], []]
    # load dataset for each client, stored as an dict, keys=Client id, value=sub-dataset belong to that client
    TrainSet_per_client, TestSet_per_client = get_dataset(args,data_path)
    # instantiate client objects stored in client_lst
    client_lst = []
    for cid in range(args.num_clients):
        client_lst.append(client(cid, args, global_model, train_set=TrainSet_per_client[cid],
                                 test_set=TestSet_per_client[cid]))
    # FL training
    m = max(int(args.frac * args.num_clients), 1)
    best_epoch, best_avg_acc, best_acc_list, state = 0, 0, [], {}
    for epoch in tqdm(range(args.epochs)):
        local_weights_lst, local_losses_lst, local_acc_lst = [], [], []

        selected_clients = np.random.choice(range(args.num_clients), m, replace=False)
        for cid in selected_clients:
            w, loss, acc = client_lst[cid].update_model(
                epoch, global_weights)
            local_weights_lst.append(copy.deepcopy(w))
            local_losses_lst.append(copy.deepcopy(loss))
            local_acc_lst.append(copy.deepcopy(acc))

        if args.is_dp:
            print('Round: {}|\tClient:{}|\tLoss: {}|\tAccuracy: {}|\tε={}'.format(
                epoch, selected_clients,
                [round(n, 3) for n in local_losses_lst],
                [round(n, 3) for n in local_acc_lst],
                [round(client_lst[i].eps[epoch], 3) for i in selected_clients]))
        else:
            print('Round: {}|\tClient:{}|\tLoss: {}|\tAccuracy: {}|'.format(
                epoch, selected_clients,
                [round(n, 3) for n in local_losses_lst],
                [round(n, 3) for n in local_acc_lst]))
        # aggregate local model weights to get global model weights
        global_weights = average_weights(local_weights_lst)

        # calculate average loss
        avg_tarin_loss = sum(local_losses_lst) / len(local_losses_lst)
        train_loss.append(avg_tarin_loss)
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for cid in range(args.num_clients):
            acc, loss = client_lst[cid].inference(global_weights)
            list_acc.append(round(acc, 3))
            list_loss.append(loss)
        avg_test_accuracy.append(sum(list_acc) / len(list_acc))
        state['Avg_test_acc'] = avg_test_accuracy
        if avg_test_accuracy[-1] > best_avg_acc:
            best_avg_acc = avg_test_accuracy[-1]
            best_acc_list = list_acc
            best_var = np.var(list_acc)
            best_epoch = epoch
            state.update({
                'best_avg_acc': best_avg_acc,
                'best_var': best_var,
                'configures': args,
                # 'client_lst': client_lst,
                'sample_count': sample_count,
                'state_dict': global_weights,  # 保存模型参数
            })

        print('\nBest_acc:{:.2%}, epoch:{:d}, variance:{:.2%} \nacc_lst:{}'.
              format(best_avg_acc, best_epoch, best_var, best_acc_list))
    client_record=[]
    for i in client_lst:
        client = {}
        client['cid'] = i.cid
        client['deleta'] = i.DELTA
        client['acc'] = i.acc
        client['eps'] = i.eps
        client['loss'] = i.loss
        client_record.append(client)
    state.update({'client_record': client_record})
    torch.save(state, save_path)
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
