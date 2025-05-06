import argparse
import torch
# from exp.exp_main import Exp_Main
# from models import WITRAN
from WITRAN import Model
import custom_repr

import random
import numpy as np

def main():
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description=
        'Models for Long-range Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--task_id', type=str, default='test', help='task id')
    parser.add_argument('--model', type=str, default='WITRAN')

    # data loader
    parser.add_argument('--data', type=str, default='electricity', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../LT-Datasets/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=1440, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length, no use for WITRAN')
    parser.add_argument('--pred_len', type=int, default=2880, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=0, help='num of heads, no use for WITRAN')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=0, help='num of decoder layers, no use for WITRAN')
    parser.add_argument('--d_ff', type=int, default=0, help='dimension of fcn, no use for WITRAN')
    parser.add_argument('--moving_avg', default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=5, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=25, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type4', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU  
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

    # For WITRAN
    parser.add_argument('--WITRAN_deal', type=str, default='None', 
        help='WITRAN deal data type, options:[None, standard]')
    parser.add_argument('--WITRAN_grid_cols', type=int, default=24, 
        help='Numbers of data grid cols for WITRAN')
    
    args = parser.parse_args()
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    args.seq_len = 96
    args.pred_len = 720
    args.enc_in = 7

    x_enc = torch.randn(args.batch_size,args.seq_len,args.enc_in)
    x_mark_enc = torch.randn(args.batch_size,args.seq_len,4)
    x_dec = torch.randn(args.batch_size,args.pred_len,args.enc_in)
    x_mark_dec = torch.randn(args.batch_size,args.pred_len,4)

    model = Model(args)
    outputs = model(x_enc,x_mark_enc,x_dec,x_mark_dec)
    print(outputs.shape)

    torch.cuda.empty_cache()
    


if __name__ == "__main__":
    main()

