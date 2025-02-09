import os
from unittest import result
import torch
import torch.nn as nn
from torch import optim
import argparse
import time
import logging
import copy
from utils import summarize_results, callback_get_label
from focal_loss.focal_loss import FocalLoss
#from torchsampler import ImbalancedDatasetSampler
from ImbalancedDatasetSampler import *
from dataset import BubbleData, BubbleDatav2
from torch.utils.data import DataLoader
from models import *
from geoopt.optim import RiemannianAdam
torch.multiprocessing.set_sharing_strategy('file_system')
import csv

if not os.path.exists("logs"):
    os.mkdir("logs")

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")


device = torch.device("cuda")
parser = argparse.ArgumentParser(
    description="Neural Bubble Predictor -- Trainer")

parser.add_argument(
    "--model",
    default="mobius_gru_attn",
    type=str,
    help="Model to use for training [mobius_gru_attn](default: mobius_gru_attn)",
)
parser.add_argument(
    "--data",
    default="text",
    type=str,
    help="Data to use for training [price, text](default: simple)",
)
parser.add_argument(
    "--lr",
    default=0.003,
    type=float,
    help="Learning rate to use for training(default: 0.001)",
)
parser.add_argument(
    "--num_epochs",
    default=100,
    type=int,
    help="Number of epochs to run for training(default: 50)",
)

parser.add_argument(
    "--decay",
    default=1e-5,
    type=float,
    help="Weight decay to use for training(default: 1e-5)",
)
parser.add_argument(
    "--batch_size",
    default=128,
    type=int,
    help="Batch Size use for training the model(default: 16)",
)

parser.add_argument(
    "--num_lookahead",
    default=10,
    type=int,
    help="Number of Lookahead days(default: 10)",
)
parser.add_argument(
    "--data_lookahead",
    default=10,
    type=int,
    help="For loading the dataset(default: 10)",
)

parser.add_argument(
    "--num_lookback",
    default=5,
    type=int,
    help="Number of Lookback days(default: 5)",
)

parser.add_argument(
    "--hidden_dim",
    default=8,
    type=int,
    help="Number of Hidden Dims for LSTM(default: 8)",
)

parser.add_argument(
    "--do_sampling",
    default=False,
    action="store_true",
    help="Whether to do sampling or not(default: False)",
)
parser.add_argument(
    "--focal_loss",
    default=False,
    action="store_true",
    help="Whether to use any custom loss(default: False)",
)
parser.add_argument(
    "--stride",
    default="3",
    type=str,
    help="Stride of this file(default: 3)",
)


def train_model(criterion, ce_criterion, num_epochs=25):
    since = time.time()

    best_model_wts = [copy.deepcopy(
        model.state_dict())]
    best_mcc = -1
    results_dict = {
        "train": [],
        "val": []
    }

    all_parameters  = []
    lst_mcc         = [] #results["MCC"],
    lst_em          = [] #results["EM"],
    lst_acc_span    = [] #results["acc_span"],
    lst_prec_span   = [] #results["precision_span"],
    lst_recall_span = [] #results["recall_span"],
    lst_f1_span     = [] #results["f1_span"],
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        logging.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        logging.info("-" * 10)

        start_idx_true_list = []
        start_idx_pred_list = []
        end_idx_true_list = []
        end_idx_pred_list = []
        true_bubble_list = []
        num_bubble_true_list = []
        num_bubble_pred_list = []
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for batch_data in dataloaders[
                phase
            ]:
                if args.data == "price":
                    inputs = batch_data[0].unsqueeze(
                        -1).to(torch.float).to(device)
                elif args.data == "text":
                    inputs = batch_data[0].to(device).float()
                else:
                    raise NotImplementedError

                if len(batch_data) > 5:
                    len_feats = batch_data[6]
                else:
                    len_feats =(torch.ones(size=(batch_size, 1))
                                * num_lookback).squeeze(-1)

                start_idx = batch_data[1].to(device).float()
                end_idx = batch_data[2].to(device).float()
                start_idx = start_idx[:, :num_days]
                end_idx = end_idx[:, :num_days]
                end_idx[:, -1] = end_idx[:, -1] + \
                   (1 -((torch.sum(start_idx, dim=1) == torch.sum(end_idx, dim=1)).int()))
                num_bubbles = torch.sum(start_idx, dim=1).long()
                true_bubble = batch_data[4]
                true_bubble = true_bubble[:, :num_days]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    num_bubbles_preds, outputs = model(inputs, len_feats)
                    start_preds = outputs[:, :, 0]
                    end_preds = outputs[:, :, 1]

                    loss1 = criterion(start_preds, start_idx)
                    loss2 = criterion(end_preds, end_idx)
                    loss3 = ce_criterion(num_bubbles_preds, num_bubbles)

                    loss = loss1 + loss2 + loss3

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                print(f"Epoch - {epoch} Batch Loss:{loss}")

                if phase == "val":
                    true_bubble_list.append(true_bubble)
                    start_idx_pred_list.append(start_preds)
                    end_idx_pred_list.append(end_preds)
                    num_bubble_true_list.append(num_bubbles)
                    num_bubble_pred_list.append(num_bubbles_preds)

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == "val":

                results = summarize_results(
                    true_bubble_list,
                    start_idx_pred_list,
                    end_idx_pred_list,
                    num_bubble_true_list,
                    num_bubble_pred_list,
                )
                results_dict[phase].append(results)
                
                logging.info(
                    "{} Loss:{:.4f},MCC:{:.4f},EM:{:.4f} EM(only_bubble):{:.4f} \n Accu(Span):{:.4f},Precision(Span):{:.4f},Recall(Span):{:.4f},F1(Span):{:.4f} \n Acc(Bubble):{:.4f},Precision(Bubble):{:.4f},Recall(Bubble):{:.4f},F1(Bubble):{:.4f}".format(
                        phase,
                        epoch_loss,
                        results["MCC"],
                        results["EM"],
                        results["EM_only_bubble"],
                        results["acc_span"],
                        results["precision_span"],
                        results["recall_span"],
                        results["f1_span"],
                        results["acc_bubble"],
                        results["precision_nbubble"],
                        results["recall_nbubble"],
                        results["f1_nbubble"],
                    )
                )

                ##add parameters:
                lst_mcc.append(results["MCC"])
                lst_em.append(results["EM"])
                lst_acc_span.append(results["acc_span"])
                lst_prec_span.append(results["precision_span"])
                lst_recall_span.append(results["recall_span"])
                lst_f1_span.append(results["f1_span"])

                
                mcc = results["MCC"]

                # deep copy the model
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_f1_span = results["f1_span"]
                    best_accu_span = results["acc_span"]
                    best_em = results["EM"]
                    best_em_only_bubble = results["EM_only_bubble"]
                    best_accu_num_bubbles = results["acc_bubble"]
                    best_f1_num_bubbles = results["f1_nbubble"]
                    best_model_wts = [
                        copy.deepcopy(model.state_dict()),
                    ]
                    '''
                    torch.save(
                        {"best_model_wts": best_model_wts[0],
                        #  "best_dec_wts": best_model_wts[1],
                        "best_val_mcc": best_mcc,
                        "args": args,
                        "model_name": model.name if hasattr(model, "name") else args.model,
                        "dataused_path": data_used,
                        "results": results_dict},
                        "saved_models/" + start_time + f"_lookback{num_lookback}_lookahead_{num_days}.pkl"
                    )
                    '''

    time_elapsed = time.time() - since
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    logging.info("Best test MCC:{:4f}".format(best_mcc))
    logging.info("Best test best_f1_span:{:4f}".format(best_f1_span))
    logging.info("Best test best_accu_span:{:4f}".format(best_accu_span))
    logging.info("Best test best_em_only_bubble:{:4f}".format(
        best_em_only_bubble))
    logging.info("Best test best_accu_num_bubbles:{:4f}".format(
        best_accu_num_bubbles))
    logging.info("Best test best_f1_num_bubbles:{:4f}".format(
        best_f1_num_bubbles))

    print("Best test MCC:{:4f}".format(best_mcc))
    print("Best test best_f1_span:{:4f}".format(best_f1_span))
    print("Best test best_accu_span:{:4f}".format(best_accu_span))
    print("Best test best_em_only_bubble:{:4f}".format(best_em_only_bubble))
    print("Best test best_accu_num_bubbles:{:4f}".format(
        best_accu_num_bubbles))
    print("Best test best_f1_num_bubbles:{:4f}".format(best_f1_num_bubbles))

    print(start_time)
    '''
    torch.save(
        {"best_model_wts": best_model_wts[0],
        "best_val_mcc": best_mcc,
        "args": args,
        "model_name": model.name if hasattr(model, "name") else args.model,
        "dataused_path": data_used,
        "results": results_dict},
        "saved_models/" + start_time + "_final.pkl"
    )
    '''

    all_parameters = {'lst_mcc': lst_mcc, 
                        'lst_em': lst_em, 
                        'lst_acc_span': lst_acc_span, 
                        'lst_prec_span': lst_prec_span, 
                        'lst_recall_span': lst_recall_span, 
                        'lst_f1_span': lst_f1_span,
                        'best_mcc': best_mcc,
                        'best_f1_span': best_f1_span, 
                        'best_accu_span': best_accu_span,
                        'best_em': best_em,
                        'best_em_only_bubble': best_em_only_bubble,
                        'best_accu_num_bubble': best_accu_num_bubbles, 
                        'best_f1_num_bubbles': best_f1_num_bubbles
                        }


    return model, all_parameters

def test_model(model, criterion, ce_criterion):

    start_idx_true_list, start_idx_pred_list = [], []
    end_idx_true_list, end_idx_pred_list = [], []
    true_bubble_list = []
    num_bubble_true_list, num_bubble_pred_list = [], []

    results_dict = { "test": [] }
    running_loss = 0.0

    all_parameters  = []
    lst_mcc         = [] #results["MCC"],
    lst_em          = [] #results["EM"],
    lst_acc_span    = [] #results["acc_span"],
    lst_prec_span   = [] #results["precision_span"],
    lst_recall_span = [] #results["recall_span"],
    lst_f1_span     = [] #results["f1_span"],
    lst_best_em_only_bubble = []
    lst_best_accu_num_bubbles = []
    lst_best_f1_num_bubbles = []

    for batch_data in dataloaders["test"]:
        if args.data == "price":
            inputs = batch_data[0].unsqueeze(
                -1).to(torch.float).to(device)
        elif args.data == "text":
            inputs = batch_data[0].to(device).float()
        else:
            raise NotImplementedError

        if len(batch_data) > 5:
            len_feats = batch_data[6]
        else:
            len_feats =(torch.ones(size=(batch_size, 1))
                         * num_lookback).squeeze(-1)

        start_idx = batch_data[1].to(device).float()
        end_idx = batch_data[2].to(device).float()
        start_idx = start_idx[:, :num_days]
        end_idx = end_idx[:, :num_days]
        end_idx[:, -1] = end_idx[:, -1] + \
           (1 -((torch.sum(start_idx, dim=1) == torch.sum(end_idx, dim=1)).int()))
        num_bubbles = torch.sum(start_idx, dim=1).long()
        true_bubble = batch_data[4]
        true_bubble = true_bubble[:, :num_days]

        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            num_bubbles_preds, outputs = model(inputs, len_feats)
            start_preds = outputs[:, :, 0]
            end_preds = outputs[:, :, 1]

            loss1 = criterion(start_preds, start_idx)
            loss2 = criterion(end_preds, end_idx)
            loss3 = ce_criterion(num_bubbles_preds, num_bubbles)

            loss = loss1 + loss2 + loss3


            print(f"Batch Loss:{loss}")

            true_bubble_list.append(true_bubble)
            start_idx_pred_list.append(start_preds)
            end_idx_pred_list.append(end_preds)
            num_bubble_true_list.append(num_bubbles)
            num_bubble_pred_list.append(num_bubbles_preds)

            running_loss += loss.item() * inputs.size(0)

            results = summarize_results(
                true_bubble_list,
                start_idx_pred_list,
                end_idx_pred_list,
                num_bubble_true_list,
                num_bubble_pred_list,
            )
            results_dict["test"].append(results)

    logging.info(f'Testing results \n'
                    f'Loss:{running_loss:.4f},MCC:{results["MCC"]:.4f},EM:{results["EM"]:.4f},EM(only_bubble):{results["EM_only_bubble"]:.4f} '
                    f'\nAccu(Span):{results["acc_span"]:.4f},Precision(Span):{results["precision_span"]:.4f},Recall(Span):{results["recall_span"]:.4f},F1(Span):{results["f1_span"]:.4f} '
                    f'\nAcc(Bubble):{results["acc_bubble"]:.4f},Precision(Bubble):{results["precision_nbubble"]:.4f},Recall(Bubble):{results["recall_nbubble"]:.4f},F1(Bubble):{results["f1_nbubble"]:.4f}')

    print(f'Testing results \n'
            f' Loss:{running_loss:.4f},MCC:{results["MCC"]:.4f},EM:{results["EM"]:.4f} EM(only_bubble):{results["EM_only_bubble"]:.4f} '
            f'\n Accu(Span):{results["acc_span"]:.4f},Precision(Span):{results["precision_span"]:.4f},Recall(Span):{results["recall_span"]:.4f},F1(Span):{results["f1_span"]:.4f} '
            f'\n Acc(Bubble):{results["acc_bubble"]:.4f},Precision(Bubble):{results["precision_nbubble"]:.4f},Recall(Bubble):{results["recall_nbubble"]:.4f},F1(Bubble):{results["f1_nbubble"]:.4f}')

    lst_mcc.append(results["MCC"])
    lst_em.append(results["EM"])
    lst_acc_span.append(results["acc_span"])
    lst_prec_span.append(results["precision_span"])
    lst_recall_span.append(results["recall_span"])
    lst_f1_span.append(results["f1_span"])
    lst_best_em_only_bubble.append(results["EM_only_bubble"])
    lst_best_accu_num_bubbles.append(results["acc_bubble"])
    lst_best_f1_num_bubbles.append(results["f1_nbubble"])
    

    all_parameters = {'best_mcc': lst_mcc, 
                        'best_em': lst_em, 
                        'best_accu_span': lst_acc_span, 
                        'lst_prec_span': lst_prec_span, 
                        'lst_recall_span': lst_recall_span, 
                        'best_f1_span': lst_f1_span,
                        'best_em_only_bubble': lst_best_em_only_bubble,
                        'best_accu_num_bubble': lst_best_accu_num_bubbles, 
                        'best_f1_num_bubbles': lst_best_f1_num_bubbles
                        }
    return all_parameters

def plotfigure(numbers, filename):

    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(numbers['lst_mcc'], 'bo-', label = 'mcc')
    plt.plot(numbers['lst_em'], 'ro-', label = 'em')
    plt.plot(numbers['lst_acc_span'], 'go-' , label = 'acc_span')
    plt.plot(numbers['lst_prec_span'], 'co-' ,label = 'prec_span')
    plt.plot(numbers['lst_recall_span'], 'mo-' ,label = 'recall_span')
    plt.plot(numbers['lst_f1_span'], 'yo-' ,label = 'f1_span')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("CryptoBubbles")
    plt.legend([ 'mcc','em','acc_span','prec_span','recall_span', 'f1_span'], loc='upper left')
    plt.savefig('figures/' + filename + '.png') 
    plt.show()
    

def savetable(numbers, numbers2, csvfilename):
    header = ['hyperparameters', 'type', 'best_mcc', 'best_em', 'best_f1_span','best_accu_span','best_em_only_bubble', 'best_accu_num_bubble','best_f1_num_bubbles' ]
    data = []

    data.append([ csvfilename, 'val', numbers['best_mcc'], numbers['best_em'], 
                                        numbers['best_f1_span'],numbers['best_accu_span'], 
                                        numbers['best_em_only_bubble'], numbers['best_accu_num_bubble'], 
                                        numbers['best_f1_num_bubbles']]) ##from val

    data.append([ csvfilename, 'test', numbers2['best_mcc'][0], numbers2['best_em'][0], 
                                        numbers2['best_f1_span'][0], numbers2['best_accu_span'][0], 
                                        numbers2['best_em_only_bubble'][0], numbers2['best_accu_num_bubble'][0], 
                                        numbers2['best_f1_num_bubbles'][0]]) ##from val
    data.append(['','','','','','' ])
    
    with open('tables2' + '.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        #writer.writerow(header)     # write the header
        writer.writerows(data)      # write multiple rows


'''

num_batches = [128]
num_epochs = [1]
num_hidden_dim = [8]
losses_list = [True, False]
do_sampling_list = [True]
'''

#num_batches = [32, 64, 128, 256, 512]
num_batches = [128]
#num_epochs = [1, 5, 10, 20, 50, 100]
num_epochs = 5
num_hidden_dim = [4]
losses_list = [False]
#do_sampling_list = [True, False]
do_sampling_list = [True]

#5_128_4_False_True_0.003_1e-05

'''
header = ['hyperparameters', 'type', 'best_mcc', 'best_em', 'best_f1_span','best_accu_span' ]
with open('tables' + '.csv', 'a', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)     # write the header
'''

for index in range(0, 10):
    for batch_size in num_batches:
        for hidden_dim in num_hidden_dim:
            for element in losses_list:
                for do_sampling in do_sampling_list:
                        args = parser.parse_args()
                        filename = str(num_epochs) + '_' + str(batch_size) + '_' + str(hidden_dim) + '_' + str(element) + '_' + str(do_sampling) + '_' + str(args.lr) + '_' + str(args.decay)
                        #print(filename)
                        
                        start_time = str(time.strftime("%Y%m%d-%H%M%S"))
                        print(start_time)
                        logging.basicConfig(
                            filename=f"logs/{start_time}.log", level=logging.INFO, format="%(message)s"
                        )
                        

                        load_embeds = True

                        if args.data == "price":
                            load_embeds = False
                            input_dim = 1
                        elif args.data == "text":
                            input_dim = 768
                        else:
                            raise NotImplementedError

                        #batch_size = args.batch_size
                        #num_epochs = args.num_epochs
                        num_days = args.num_lookahead
                        num_lookback = args.num_lookback
                        #hidden_dim = args.hidden_dim
                        LR = args.lr
                        num_span_classes = 5
                        args.focal_loss = element
                        #print(num_epochs, batch_size, hidden_dim, args.focal_loss, element)
                        
                        args.batch_size = batch_size
                        args.num_epochs = num_epochs
                        args.hidden_dim = hidden_dim
                        args.do_sampling = do_sampling

                        print(args)
                        logging.info(
                            f"Model:{args.model} Batch Size:{batch_size} Num Epochs:{num_epochs}  Hidden Dim:{hidden_dim} LR:{LR} Decay:{args.decay}\n "
                        )

                        logging.info(
                            f"Data:{args.data}  Data used: final_split_data_dtype_values_lookback_{args.num_lookback}_lookahead_{args.data_lookahead}_stride_{args.stride}.pkl \n Num Lookback:{num_lookback} Num Lookahead:{num_days} Sampling:{args.do_sampling} Focal Loss:{args.focal_loss} \n\n"
                        )
                        save_path = "/data/leticia/crypto/"
                        data_used = {
                            "train": [f"{save_path}train_data_price_only_lookback_{num_lookback}_lookahead_{args.num_lookahead}_stride_{args.stride}.pkl",
                                    f"{save_path}train_data_text_only_lookback_{num_lookback}_lookahead_{args.num_lookahead}_stride_{args.stride}.pkl"],
                            "val":  [f"{save_path}val_data_price_only_lookback_{num_lookback}_lookahead_{args.num_lookahead}_stride_{args.stride}.pkl",
                                    f"{save_path}val_data_text_only_lookback_{num_lookback}_lookahead_{args.num_lookahead}_stride_{args.stride}.pkl"],
                            "test": [f"{save_path}test_data_price_only_lookback_{num_lookback}_lookahead_{args.num_lookahead}_stride_{args.stride}.pkl",
                                    f"{save_path}test_data_text_only_lookback_{num_lookback}_lookahead_{args.num_lookahead}_stride_{args.stride}.pkl"]
                        }

                        print('train: price_data_path: ', data_used["train"][0], ', embed_folder_path: ', data_used["train"][1])
                        #trainset = BubbleData(
                        trainset = BubbleDatav2(
                            price_data_path=data_used["train"][0],
                            load_embeds=load_embeds,
                            embed_folder_path=data_used["train"][1],
                        )

                        print('val: price_data_path: ', data_used["val"][0], ', embed_folder_path: ', data_used["val"][1])
                        #valset = BubbleData(
                        valset = BubbleDatav2(
                            price_data_path=data_used["val"][0],
                            load_embeds=load_embeds,
                            embed_folder_path=data_used["val"][1],
                        )

                        testset = BubbleDatav2(
                            price_data_path=data_used["test"][0],
                            load_embeds=load_embeds,
                            embed_folder_path=data_used["test"][1],
                        )

                        if args.do_sampling:
                            trainloader = DataLoader(
                                trainset,
                                sampler = ImbalancedDatasetSampler(
                                    trainset, callback_get_label=callback_get_label
                                ),
                                batch_size=batch_size,
                                num_workers=2,
                                drop_last=True,
                            )
                        else:
                            trainloader = DataLoader(
                                trainset,
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=2,
                                drop_last=True,
                            )
                        valloader = DataLoader(
                            valset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
                        )
                        testloader = DataLoader(
                            testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
                        )
                        dataloaders = {"train": trainloader, "val": valloader, "test": testloader}

                        if args.data == "price":
                            maxlen = num_lookback
                        else:
                            maxlen = num_lookback * 15

                        model = MobiusEncDecGRUAttn(input_dim, hidden_dim,
                                                    num_span_classes, out_dim=2, num_days=num_days, maxlen=maxlen)


                        dataset_sizes = {"train": len(trainset), "val": len(valset), "test": len(testset)}

                        if args.focal_loss:
                            criterion1 = FocalLoss(alpha=2, gamma=5).cuda()
                        else:
                            criterion1 = nn.BCELoss()

                        model.cuda()
                        criterion2 = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=args.decay)
                        model, numbers = train_model(criterion1, criterion2, num_epochs)
                        numbers2 = test_model(model, criterion1, criterion2)
                        
                        plotfigure(numbers, filename)
                        savetable(numbers, numbers2, filename)




