import torch
import numpy as np
import random
import os
import time
from model_gan import *
import timeit
from sklearn.model_selection import KFold
import numpy as np
from torchsummary import summary


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    DATASET = "data_mol2"
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

    # device = torch.device('cpu')
    # print('The code uses CPU!!!')

    # device = torch.device('cpu')
    # print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('dataset/' + DATASET + '/ddi/')
    seq1 = load_tensor(dir_input + 'drug1', torch.FloatTensor)
    # adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    seq2 = load_tensor(dir_input + 'drug2', torch.FloatTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    # print(proteins[0],len(proteins[0]))
    # print("compounds",type(seq1))
    # print("adj",len(adjacencies))
    # print("protein",len(proteins))
    # print("interactions",len(interactions))

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(seq1, seq2, interactions))
    train_size = int(len(dataset) * 0.8)
    validate_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - validate_size - train_size
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size, validate_size,test_size])
    print(len(dataset))
    print(len(train_dataset))
    print(len(validate_dataset))
    print(len(test_dataset))
    path = "output_data_mol_wv_gan"
    print("结果",path)
    # print("dataset",len(dataset))

    # X = np.arange(24).reshape(12,2)

    # kf = KFold(n_splits=5,shuffle=True)  # 初始化KFold
    # for dataset_train , dataset_dev in kf.split(dataset):  # 调用split方法切分数据
    #     print('dataset_train:%s , test_index: %s ' %(dataset_train,dataset_dev))

    # dataset = shuffle_dataset(dataset, 1234)
    # dataset_train, dataset_dev = split_dataset(dataset, 0.8)

    """ create model ,trainer and tester """
    protein_dim = 34
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 64
    lr = 1e-4
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1.0
    iteration = 200
    kernel_size = 7

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout,Generator,device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,
                      PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    # model = nn.DataParallel(model)
    # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))
    model.to(device)
    print(model)
    # summary(model,input_size=(batch,protein_dim,atom_dim))
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)
 

    """Output files."""
    file_AUCs = 'output_data_mol_wv_gan/result/train'+ '.txt'
    test_file = 'output_data_mol_wv_gan/result/test'+ '.txt'
    file_model = 'output_data_mol_wv_gan/model/'+ 'model.pkl'
    AUCs = ('Epoch\tTime(sec)\tLoss_train  \tACC \t ROC\tAUC_PRC \tF1')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    
#     """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()
    max_ACC_dev = 0.0
    for epoch in range(1, iteration + 1):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(train_dataset, device)
        acc,roc,auc_prc,f1 = tester.test(validate_dataset)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, acc,roc,auc_prc,f1]
        tester.save_AUCs(AUCs, file_AUCs)
        # if AUC_dev > max_AUC_dev:
        if acc > max_ACC_dev:
            tester.save_model(model, file_model)
            max_ACC_dev = acc
        print('\t'.join(map(str, AUCs)))
    
    net = torch.load(file_model)
    # net = net.load_state_dict(torch.load(file_model))
    # state_dict = torch.load(PATH)
    net.to(device)
    # net.eval()
    # torch.no_grad()
    tt = Tester(net)
    ACC,AUROC,AUPRC,F1 = tt.test(test_dataset)
    title = ('ACC \t ROC\tAUC_PRC \tF1')
    print(title)
    parm = [ACC, AUROC, AUPRC, F1]
    tt.save_AUCs(parm,test_file)
    print("test acc:",ACC)
    print("test AUROC:",AUROC)
    print("test AUPRC:",AUPRC)
    print("test F1:",F1)
 
    
