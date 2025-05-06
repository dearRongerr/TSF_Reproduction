import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from layers.Embed import WITRAN_Temporal_Embedding

class WITRAN_2DPSGMU_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, water_rows, water_cols, res_mode='none'):
        super(WITRAN_2DPSGMU_Encoder, self).__init__()
        self.input_size = input_size # 11
        self.hidden_size = hidden_size # 32
        self.num_layers = num_layers # 3
        self.dropout = dropout # 
        self.water_rows = water_rows # 4 填
        self.water_cols = water_cols  # 24小时
        self.res_mode = res_mode # 
        # parameter of row cell
        self.W_first_layer = torch.nn.Parameter(torch.empty(6 * hidden_size, input_size + 2 * hidden_size))
        '''
            输入 hidden_size=32，input_size=11
            处理：torch.nn.Parameter(torch.empty(
            输出：self.W_first_layer [192,75]
        '''
        self.W_other_layer = torch.nn.Parameter(torch.empty(num_layers - 1, 6 * hidden_size, 4 * hidden_size))
        '''
        in(type shape): 
            num_layers: 网络的层数，例如 3。
            hidden_size: 隐藏状态的特征维度，例如 32。
            self.W_other_layer 的形状为 [num_layers - 1, 6 * hidden_size, 4 * hidden_size]。
            - num_layers - 1 = 2（因为第一层的权重单独定义为 self.W_first_layer）。
            - 6 * hidden_size = 192（用于生成 6 个门的信号）。
            - 4 * hidden_size = 128（输入特征维度，包括行隐藏状态、列隐藏状态和上一层的输出）。
        deal(function meaning):
            初始化其他层的权重矩阵，用于多层网络中从第二层开始的计算。
            - 每一层的权重矩阵形状为 [6 * hidden_size, 4 * hidden_size]。
            - 通过 torch.empty 创建未初始化的张量，并将其包装为可训练参数（torch.nn.Parameter）。
        out(type shape): 
            self.W_other_layer: 张量，形状为 [num_layers - 1, 6 * hidden_size, 4 * hidden_size]。
            - 例如，形状为 [2, 192, 128]。
        现实含义: 
            定义从第二层开始的权重矩阵，用于多层网络中每一层的线性变换。
            - 每一层的输入特征维度为 4 * hidden_size（包括行隐藏状态、列隐藏状态和上一层的输出）。
            - 每一层的输出特征维度为 6 * hidden_size（用于生成 6 个门的信号）。
            - 这些权重矩阵用于控制隐藏状态的更新和输出。
        '''
        self.B = torch.nn.Parameter(torch.empty(num_layers, 6 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def linear(self, input, weight, bias, batch_size, slice, Water2sea_slice_num):
        a = F.linear(input, weight)
        if slice < Water2sea_slice_num: 
            a[:batch_size * (slice + 1), :] = a[:batch_size * (slice + 1), :] + bias
        return a
    '''
    该函数接收的参数：
        input [128,75]、weight [192,75]、 bias [192]、 batch_size = 32、  slice = 0 、Water2sea_slice_num=4
    第一句：a = F.linear(input, weight)
        input [batch_size, input_features] = [128,75] 
        weight [output_features, input_features] = [192,75]
        a 输出张量 [batch_size, output_features] = [128,192]
    if 判断  slice=0, Water2sea_slice_num=4
        a[:batch_size * (slice + 1), :] = a[:batch_size * (slice + 1), :] + bias
        通过 slice 控制偏置加法的范围，逐步扩展到更多的行
    '''

    ''' if
        a[:batch_size * (slice + 1), :]：表示从 a 中取出前 batch_size * (slice + 1) 行的所有列
        当 slice = 0 时，取前 32 * (0 + 1) = 32 行、只对前 32 行加偏置
        当 slice = 1 时，取前 32 * (1 + 1) = 64 行、对前 64 行加偏置。
        当 slice = 2 时，取前 32 * (2 + 1) = 96 行、对前 96 行加偏置。
        当 slice = 3 时，取前 32 * (3 + 1) = 128 行、对所有 128 行加偏置
        a[:batch_size * (slice + 1), :] + bias：对取出的行加上偏置 bias；偏置 bias 会广播到每一行的所有列
        a[:batch_size * (slice + 1), :] = ...： 将加了偏置的结果重新赋值回 a 的前 batch_size * (slice + 1) 行

    '''

    def forward(self, input, batch_size, input_size, flag): 
        '''
           
            参数说明：
                - input: 输入张量，包含批次大小、行数、列数和特征维度 = [32,4,24,11]
                - batch_size: 批次大小 = 32
                - input_size: 特征维度（features） = 11
                - flag: 调整维度顺序 = 0
        
        '''
       
        if flag == 1: 
            input = input.permute(2, 0, 1, 3)
        else: 
            input = input.permute(1, 0, 2, 3)
        '''
            这里 flag = 0 执行 else 
            ∴ [32,4,24,11] ->  [4,32,24,11] 4 天 24 小时，每个小时 11 个特征；32 个特征
        
        '''
        
        Water2sea_slice_num, _, Original_slice_len, _ = input.shape

        '''
            获取调整维度后的形状 切片数（4、4 天）、原始切片长度（24 小时）
            Water2sea_slice_num = 4 ，Original_slice_len = 24
        '''

        
        Water2sea_slice_len = Water2sea_slice_num + Original_slice_len - 1

        '''
            计算扩展后的序列长度 Water2sea_slice_num = 4,Original_slice_len = 24
            Water2sea_slice_len = 24 + 4 - 1 = 27
            切片长度 = 切片数 + 原始切片长度 - 1
        '''

        hidden_slice_row = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
        '''
            初始化行和列的隐藏状态

            输入： Water2sea_slice_num = 4，batch_size = 32，hidden_size = 32
            处理：torch.zeros 
            输出： hidden_slice_row :  [4 * 32, 28] = [128, 32]
        '''

        hidden_slice_col = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)

        '''
            存储每个时间步的列隐藏状态
            输入：
                Water2sea_slice_num = 4，batch_size = 32，self.hidden_size = 32
            处理：torch.zeros
            输出：
                hidden_slice_col = [128, 32]        
        '''


        input_transfer = torch.zeros(Water2sea_slice_num, batch_size, Water2sea_slice_len, input_size).to(input.device)

        '''
        in(type shape): 
            Water2sea_slice_num = 4(切片数)，batch_size = 32，Water2sea_slice_len = 27（切片长度），input_size = 11（特征维度7+时间戳特征 4）
        处理: 
            torch.zeros
        out(type shape): 
            input_transfer [4,32,27,11]
        现实含义: 初始化扩展后的输入张量，填充原始输入数据
          
        '''

        for r in range(Water2sea_slice_num): 
            input_transfer[r, :, r:r+Original_slice_len, :] = input[r, :, :, :]

            '''
                将原始输入填充到扩展后的张量中，沿着扩展后的序列长度填充原始输入数据
                处理的对象:  
                    - input_transfer  [Water2sea_slice_num, batch_size, Water2sea_slice_len, input_size] [4, 32, 27, 11]
                    - input  [Water2sea_slice_num, batch_size, Original_slice_len, input_size] [4, 32, 24, 11]
                变量定义说明：
                    - Water2sea_slice_num = 4（行数）
                    - batch_size = 32
                    - Original_slice_len = 24（原始列数）
                    - input_size = 11（特征维度）
                    - Water2sea_slice_len = 27（扩展后的序列长度）

                维度变化：
                - 原始输入 input 的形状为  
                    * [4, 32, 24, 11]
                    *（行数、批次大小、列数、特征维度）
                    * 4 天，每天 24 小时
                    * [self.WITRAN_grid_enc_rows, batch_size, self.WITRAN_grid_cols, input_size]
                    * 原始输入 input 的形状为 [4, 32, 24, 11]，表示 4 天的数据，每天有 24 小时，每小时有 11 个特征 
                - for 循环
                次数 Water2sea_slice_num = 4
                    r = 0 input_transfer[0, :, 0:24, :] = input[0, :, :, :] [32, 24, 11]
                    r = 1 input_transfer[1, :, 1:25, :] = input[1, :, :, :] [32, 24, 11]
                    r = 2 input_transfer[2, :, 2:26, :] = input[2, :, :, :] [32, 24, 11]
                    r = 3 input_transfer[3, :, 3:27, :] = input[3, :, :, :] [32, 24, 11]
                    * 第 0 行（第 1 天的数据） 填充后，input_transfer[0, :, 0:24, :] 包含第 1 天的 24 小时数据。
                    * 第 1 行（第 2 天的数据）填充后，input_transfer[1, :, 1:25, :] 包含第 2 天的 24 小时数据，并与第 1 天的数据在时间维度上产生了 1 小时的重叠
                    * 第 2 行（第 3 天的数据）：填充后，input_transfer[2, :, 2:26, :] 包含第 3 天的 24 小时数据，并与第 2 天的数据在时间维度上产生了 2 小时的重叠。
                    * 第 3 行（第 4 天的数据）：填充后，input_transfer[3, :, 3:27, :] 包含第 4 天的 24 小时数据，并与第 3 天的数据在时间维度上产生了 3 小时的重叠。
                - 填充后 input_transfer 的形状为  
                    [4, 32, 27, 11]
                    （行数、批次大小、扩展后的序列长度、特征维度）  
                    [Water2sea_slice_num, batch_size, Water2sea_slice_len, input_size]            
            '''



        hidden_row_all_list = []   
        hidden_col_all_list = []   
        '''
            初始化保存隐藏状态的列表，用于存储所有行和列的隐藏状态
            hidden_row_all_list 保存每个时间步的行隐藏状态
            hidden_col_all_list 保存每个时间步的列隐藏状态
        
        '''

        
        for layer in range(self.num_layers): 
            '''
            self.num_layers = 3
            
            '''
            if layer == 0:
                '''
                layer = 0 第一层使用扩展后的输入张量作为输入
                '''
                a = input_transfer.reshape(Water2sea_slice_num * batch_size, Water2sea_slice_len, input_size) 
                '''
                    输入：Water2sea_slice_num=4，batch_size=32，Water2sea_slice_len27，input_size=11
                    处理：.reshape
                    输出：a [128,27,11]
                '''             
                W = self.W_first_layer
                '''
                    输入：self.W_first_layer = torch.nn.Parameter(torch.empty(6 * hidden_size, input_size + 2 * hidden_size))
                                
                        输入 hidden_size=32，input_size=11
                        处理：torch.nn.Parameter(torch.empty(
                        输出：self.W_first_layer [192,75]
        
                    输出：W = self.W_first_layer = [192,75]
                '''

            else:
                a = F.dropout(output_all_slice, self.dropout, self.training)
                '''
                in(type shape): 
                    - output_all_slice [128, 27, 64] 64=行隐藏状态32+列隐藏状态 32
                    - self.dropout = 0.05（Dropout 概率）
                    - self.training = True（当前处于训练模式）
                deal(function meaning):
                    使用 F.dropout 对上一层的输出进行 Dropout 操作，随机将部分神经元置为 0，以防止过拟合。
                out(type shape): 
                    a [128, 27, 64]（添加 Dropout 后的输出）
                现实含义: 
                  
                '''
                if layer == 1:
                    layer0_output = a   # 保存第一层的输出，用于残差连接
                    '''
                        输入：
                            a [128, 27, 64]（第一层的输出）
                        处理：
                            将第一层的输出保存到 layer0_output 中，用于后续层的残差连接。
                        输出：
                            layer0_output [128, 27, 64]
                    
                    '''
                W = self.W_other_layer[layer-1, :, :]   # 其他层的权重矩阵
                '''
                    输入：
                        self.W_other_layer [2, 192, 128]（其他层的权重矩阵，形状为 [num_layers-1, 6 * hidden_size, 4 * hidden_size]）
                        layer-1（当前层的索引减 1，用于选择对应层的权重矩阵）
                    处理：
                        提取当前层的权重矩阵 W。
                    输出：
                        W [192, 128]（当前层的权重矩阵）
                
                '''
                hidden_slice_row = hidden_slice_row * 0   # 重置行隐藏状态
                hidden_slice_col = hidden_slice_col * 0   # 重置列隐藏状态
                '''
                输入：
                    hidden_slice_row [128, 32]（上一层的行隐藏状态）
                    hidden_slice_col [128, 32]（上一层的列隐藏状态）
                处理：
                    将行隐藏状态和列隐藏状态重置为全 0。
                输出：
                    hidden_slice_row [128, 32]（全 0 的行隐藏状态）
                    hidden_slice_col [128, 32]（全 0 的列隐藏状态）
                现实含义：
                    在每一层开始时重置隐藏状态，以确保每一层的计算独立。
                '''
            
            B = self.B[layer, :]
            '''
            输入：
                self.B [3, 192]（偏置矩阵，形状为 [num_layers, 6 * hidden_size]）
                layer（当前层的索引）
            处理：
                提取当前层的偏置向量 B。
            输出：
                B [192]（当前层的偏置向量）
            '''

            # start every for all slice
            output_all_slice_list = []
            '''
                初始化保存当前层输出的列表
            
            '''

             # 遍历每个时间步（slice），逐步计算
            for slice in range (Water2sea_slice_len): # Water2sea_slice_len = 27 切片长度
                # 生成门控机制的输入
                gate = self.linear(torch.cat([hidden_slice_row, hidden_slice_col, a[:, slice, :]],dim = -1), 
                                   W, 
                                   B, 
                                   batch_size, 
                                   slice, 
                                   Water2sea_slice_num)
                ''' 
                    输入：
                        torch.cat([hidden_slice_row, hidden_slice_col, a[:, slice, :]],dim = -1)
                            hidden_slice_row [128,32]
                            hidden_slice_col [128,32]
                            a [128,27,11]   
                                a[:, slice, :].shape [128,11]
                        cat.shape [128,75]
                        W.shape = [192,75]
                        B (192,)
                        batch_size = 32
                        slice = 0
                        Water2sea_slice_num =4 切片数=4

                    处理：self.linear
                    输出：
                        gate
                        gate [128,192]  
                        
                    Water2sea_slice_num = 4, batch_size=32, hidden_size = 128

                    Water2sea_slice_len = 27
                    hidden_slice_row  [128,32]
                    hidden_slice_col  [128,32]
                    a[:, slice, :] [128,11]
                    self.linear
                    
                
                
                '''
                # 维度变化：
                # - hidden_slice_row 的形状为 [Water2sea_slice_num * batch_size, hidden_size]
                # - hidden_slice_col 的形状为 [Water2sea_slice_num * batch_size, hidden_size]
                # - a[:, slice, :] 的形状为 [Water2sea_slice_num * batch_size, input_size]
                # - 拼接后，输入 gate 的形状为 
               
                # gate  
                sigmod_gate, tanh_gate = torch.split(gate, 4 * self.hidden_size, dim = -1)
                '''
                in(type shape): 
                    gate [128,192]
                    4 * self.hidden_size = 4*32 = 128
                deal(function meaning):
                    torch.split 将张量沿指定维度分割成若干子张量、torch.split 将 gate 按列分割为两部分，用于不同的激活函数处理。
                out(type shape): 
                    sigmod_gate [128,128]
                    tanh_gate [128,64]

                现实含义: 
                    将 gate 张量按照最后一个维度（dim=-1）分割为两部分：
                    - sigmod_gate：前 128 列（4 * self.hidden_size），用于 Sigmoid 激活。
                    - tanh_gate：后 64 列（剩余部分），用于 Tanh 激活。
                '''
                sigmod_gate = torch.sigmoid(sigmod_gate)
                '''
                in(type shape): 
                    sigmod_gate [128,128]
                deal(function meaning):
                    torch.sigmoid
                out(type shape): 
                    sigmod_gate [128,128]
                现实含义: 
                        对 sigmod_gate 进行 Sigmoid 激活，将值映射到 (0, 1) 区间，用于生成门控信号。
                        - Sigmoid 激活函数的公式为：f(x) = 1 / (1 + exp(-x))
                        - 主要用于控制更新门和输出门的开关程度。
                '''
                tanh_gate = torch.tanh(tanh_gate)
                '''
                in(type shape): tanh_gate [128,64]
                deal(function meaning):torch.tanh
                out(type shape): tanh_gate [128,64]
                现实含义: Tanh 激活
                    对 tanh_gate 进行 Tanh 激活，将值映射到 (-1, 1) 区间，用于生成输入门信号。
                    - Tanh 激活函数的公式为：f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
                    - 主要用于生成输入门的候选值。                  
                '''

                
                update_gate_row, output_gate_row, update_gate_col, output_gate_col = sigmod_gate.chunk(4, dim = -1)
                '''
                in(type shape): 
                    
                    sigmod_gate [128,128]
                deal(function meaning):
                    .chunk(4, dim = -1)
                out(type shape): 
                    update_gate_row [128,32]
                    output_gate_row [128,32]
                    update_gate_col [128,32]
                    output_gate_col [128,32]
                现实含义: 
                    将 sigmod_gate 张量按照最后一个维度（dim=-1）分割为 4 部分：
                    - update_gate_row：更新行隐藏状态的门控信号。
                    - output_gate_row：输出行隐藏状态的门控信号。
                    - update_gate_col：更新列隐藏状态的门控信号。
                    - output_gate_col：输出列隐藏状态的门控信号。     
                  
                '''
                input_gate_row, input_gate_col = tanh_gate.chunk(2, dim = -1)
                '''
                in(type shape): 
                    tanh_gate [128,64]
                deal(function meaning):
                    .chunk(2, dim = -1)
                out(type shape): 
                    input_gate_row [128,32]
                    input_gate_col [128,32]
                现实含义: 
                    将 tanh_gate 张量按照最后一个维度（dim=-1）分割为 2 部分：
                    - input_gate_row：用于更新行隐藏状态的输入门信号。
                    - input_gate_col：用于更新列隐藏状态的输入门信号。
                  
                '''
                
                
                hidden_slice_row = torch.tanh((1-update_gate_row)*hidden_slice_row + update_gate_row*input_gate_row) * output_gate_row
                '''
                in(type shape): 
                    hidden_slice_row [128,32]
                    update_gate_row [128,32]
                    input_gate_row [128,32]
                    output_gate_row [128,32]

                deal(function meaning):
                    - 计算 (1 - update_gate_row) * hidden_slice_row + update_gate_row * input_gate_row，用于更新隐藏状态。
                    - 对结果应用 Tanh 激活函数，将值映射到 (-1, 1) 区间。
                    - 最后乘以 output_gate_row，控制输出的强度。               
                out(type shape): 
                    hidden_slice_row  [Water2sea_slice_num(4) * batch_size(32), hidden_size]
                    hidden_slice_row [128,32]
                现实含义: 
                    更新行隐藏状态：
                    - 使用更新门（update_gate_row）和输入门（input_gate_row）计算新的隐藏状态。
                    - 使用输出门（output_gate_row）控制隐藏状态的输出。
                  
                '''

                
                hidden_slice_col = torch.tanh(
                    (1-update_gate_col)*hidden_slice_col + update_gate_col*input_gate_col) * output_gate_col
                '''
                in(type shape): 
                    hidden_slice_col [128,32]
                    update_gate_col [128,32]
                    input_gate_col [128,32]
                    output_gate_col [128,32]
                deal(function meaning):
                    - 计算 (1 - update_gate_col) * hidden_slice_col + update_gate_col * input_gate_col，用于更新隐藏状态。
                    - 对结果应用 Tanh 激活函数，将值映射到 (-1, 1) 区间。
                    - 最后乘以 output_gate_col，控制输出的强度。
                out(type shape): 
                    hidden_slice_col [128,32]
                现实含义: 
                    更新列隐藏状态：
                    - 使用更新门（update_gate_col）和输入门（input_gate_col）计算新的隐藏状态。
                    - 使用输出门（output_gate_col）控制隐藏状态的输出。
                  
                '''
                
                
                output_slice = torch.cat([hidden_slice_row, hidden_slice_col], dim = -1)
                '''
                in(type shape): 
                    hidden_slice_row [128,32]
                    hidden_slice_col [128,32]
                deal(function meaning):
                    - 将行隐藏状态（hidden_slice_row）和列隐藏状态（hidden_slice_col）在最后一个维度（dim=-1）上拼接。
                out(type shape): 
                    output_slice [128,64]
                现实含义: 
                    将行隐藏状态和列隐藏状态拼接在一起，作为当前时间步的输出。
                    
                '''

                
                output_all_slice_list.append(output_slice)
                '''
                in(type shape): 
                    output_slice [128,64]
                deal(function meaning):
                     - 将当前时间步的输出（output_slice）添加到输出列表（output_all_slice_list）中。
                out(type shape): 
                     output_all_slice_list 包含多个时间步的输出，每个元素形状为 [128,64]。
                现实含义: 
                    保存当前时间步的输出，为后续的时间步计算和最终输出提供数据。
                  
                '''

                # save row hidden # 保存行隐藏状态
                if slice >= Original_slice_len - 1:
                    need_save_row_loc = slice - Original_slice_len + 1
                    '''
                    in(type shape): 
                        slice: 当前时间步索引，范围为 [0, Water2sea_slice_len - 1]
                        Original_slice_len: 原始序列长度 = 24
                    deal(function meaning):
                        计算需要保存的行隐藏状态的位置索引 need_save_row_loc。
                        当 slice >= Original_slice_len - 1 时，计算公式为：
                            need_save_row_loc = slice - Original_slice_len + 1
                    out(type shape): 
                        need_save_row_loc: 整数，表示需要保存的行隐藏状态的起始位置。
                    现实含义: 
                        确定当前时间步对应的行隐藏状态在批次中的位置，用于保存该时间步的行隐藏状态。
                      
                    '''
                    hidden_row_all_list.append(
                        hidden_slice_row[need_save_row_loc*batch_size:(need_save_row_loc+1)*batch_size, :])
                    '''
                    in(type shape): 
                        hidden_slice_row [128, 32]（Water2sea_slice_num * batch_size, hidden_size）
                        need_save_row_loc: 当前需要保存的行隐藏状态的起始位置。
                        batch_size: 批次大小 = 32。
                    deal(function meaning):
                        从 hidden_slice_row 中提取当前时间步对应的行隐藏状态：
                        hidden_slice_row[need_save_row_loc*batch_size:(need_save_row_loc+1)*batch_size, :]
                        提取范围为 [need_save_row_loc * batch_size, (need_save_row_loc + 1) * batch_size]。
                        将提取的行隐藏状态添加到 hidden_row_all_list 中。
                    out(type shape): 
                         hidden_row_all_list: 列表，其中每个元素的形状为 [batch_size, hidden_size]。
                    现实含义: 
                        保存当前时间步的行隐藏状态，用于后续的计算或输出。
                      
                    '''
                    

                # save col hidden # 保存列隐藏状态
                if slice >= Water2sea_slice_num - 1:
                    hidden_col_all_list.append(
                        hidden_slice_col[(Water2sea_slice_num-1)*batch_size:, :])
                    # hidden_col_all_list 中每个元素的形状为 [batch_size, hidden_size]
                    '''
                    in(type shape): 
                        hidden_slice_col [128, 32]（Water2sea_slice_num * batch_size, hidden_size）
                        Water2sea_slice_num: 切片数 = 4
                         batch_size: 批次大小 = 32
                    deal(function meaning):
                        从 hidden_slice_col 中提取当前时间步对应的列隐藏状态：
                        hidden_slice_col[(Water2sea_slice_num-1)*batch_size:, :]
                        提取范围为 [(Water2sea_slice_num - 1) * batch_size, :]
                        将提取的列隐藏状态添加到 hidden_col_all_list 中。

                    out(type shape): 
                        hidden_col_all_list: 列表，其中每个元素的形状为 [batch_size, hidden_size]。
                    现实含义: 
                        保存当前时间步的列隐藏状态，用于后续的计算或输出。
                    
                    '''

                # hidden transfer # 滚动列隐藏状态
                hidden_slice_col = torch.roll(hidden_slice_col, shifts=batch_size, dims = 0)
                '''
                in(type shape): 
                    hidden_slice_col [128, 32]（Water2sea_slice_num * batch_size, hidden_size）
                    batch_size: 批次大小 = 32
                deal(function meaning):
                    使用 torch.roll 滚动 hidden_slice_col 的数据：
                    - shifts=batch_size：沿第 0 维滚动 batch_size 个位置。
                    - dims=0：指定滚动的维度为第 0 维。
                    滚动操作会将前 batch_size 行的数据移动到最后，其他行依次向前移动。
                out(type shape): 
                    hidden_slice_col [128, 32]
                现实含义: 
                    滚动列隐藏状态，使得下一时间步的列隐藏状态能够正确对应到当前时间步的计算。
                  
                '''
            # 计算当前层的输出
            if self.res_mode == 'layer_res' and layer >= 1: # layer-res  # 如果启用残差连接
                output_all_slice = torch.stack(output_all_slice_list, dim = 1) + layer0_output
                '''
                in(type shape):
                    hidden_slice_col [128, 32]（Water2sea_slice_num * batch_size, hidden_size） 
                    batch_size: 批次大小 = 32
                deal(function meaning):
                    使用 torch.roll 滚动 hidden_slice_col 的数据：
                    - shifts=batch_size：沿第 0 维滚动 batch_size 个位置。
                    - dims=0：指定滚动的维度为第 0 维。
                    滚动操作会将前 batch_size 行的数据移动到最后，其他行依次向前移动。
                out(type shape):
                    hidden_slice_col [128, 32] 
                现实含义: 
                    滚动列隐藏状态，使得下一时间步的列隐藏状态能够正确对应到当前时间步的计算。
                  
                '''
            else:
                output_all_slice = torch.stack(output_all_slice_list, dim = 1)
            # output_all_slice 的形状为 [Water2sea_slice_num * batch_size, Water2sea_slice_len, 2 * hidden_size]
            '''
            in(type shape): 
                output_all_slice_list: 列表，包含多个时间步的输出，每个元素的形状为 [Water2sea_slice_num * batch_size, 2 * hidden_size]
                - Water2sea_slice_num * batch_size = 128
                - 每个时间步的输出形状为 [128, 64]（2 * hidden_size = 64）
                - 列表长度为 Water2sea_slice_len = 27
            deal(function meaning):
                使用 torch.stack 将 output_all_slice_list 中的所有时间步的输出沿第 1 维堆叠，形成一个三维张量。
                - dim=1：指定堆叠的维度为第 1 维（时间步维度）。
            out(type shape): 
                output_all_slice: 张量，形状为 [Water2sea_slice_num * batch_size, Water2sea_slice_len, 2 * hidden_size]
                - [128, 27, 64]
            现实含义: 
                将所有时间步的输出堆叠在一起，形成一个包含所有时间步输出的张量。
                - 每一行表示一个样本的输出。
                - 时间步维度（第 1 维）表示序列的长度。
                - 最后一维（2 * hidden_size）表示拼接后的行隐藏状态和列隐藏状态的特征。
              
            '''
        # 将所有行隐藏状态堆叠
        hidden_row_all = torch.stack(hidden_row_all_list, dim = 1)
        # hidden_row_all 的形状为 [batch_size, num_layers, Water2sea_slice_num, hidden_size]
        '''
        in(type shape): 
        deal(function meaning):
        out(type shape): 
        现实含义: 
          
        '''

        # 将所有列隐藏状态堆叠
        hidden_col_all = torch.stack(hidden_col_all_list, dim = 1)
        # hidden_col_all 的形状为 [batch_size, num_layers, Original_slice_len, hidden_size]
        '''
        in(type shape): 
        deal(function meaning):
        out(type shape): 
        现实含义: 
          
        '''
        hidden_row_all = hidden_row_all.reshape(batch_size, self.num_layers, Water2sea_slice_num, hidden_row_all.shape[-1])
        '''
        in(type shape): 
        deal(function meaning):
        out(type shape): 
        现实含义: 
          
        '''
        hidden_col_all = hidden_col_all.reshape(batch_size, self.num_layers, Original_slice_len, hidden_col_all.shape[-1])
        '''
        in(type shape): 
        deal(function meaning):
        out(type shape): 
        现实含义: 
          
        '''

        # 返回结果
        if flag == 1:
            return output_all_slice, hidden_col_all, hidden_row_all
        else:
            return output_all_slice, hidden_row_all, hidden_col_all

class Model(nn.Module):
    def __init__(self, configs, WITRAN_dec='Concat', WITRAN_res='none', WITRAN_PE='add'):
        super(Model, self).__init__()
        self.standard_batch_size = configs.batch_size
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout
        self.WITRAN_dec = WITRAN_dec
        self.WITRAN_deal = configs.WITRAN_deal
        self.WITRAN_res = WITRAN_res
        self.PE_way = WITRAN_PE
        self.WITRAN_grid_cols = configs.WITRAN_grid_cols
        self.WITRAN_grid_enc_rows = int(configs.seq_len / self.WITRAN_grid_cols)
        self.WITRAN_grid_dec_rows = int(configs.pred_len / self.WITRAN_grid_cols)
        self.device = configs.gpu
        if configs.freq== 'h':
            Temporal_feature_dim = 4
        # Encoder
        self.encoder_2d = WITRAN_2DPSGMU_Encoder(self.enc_in + Temporal_feature_dim, self.d_model, self.num_layers, 
            self.dropout, self.WITRAN_grid_enc_rows, self.WITRAN_grid_cols, self.WITRAN_res)
        # Embedding
        self.dec_embedding = WITRAN_Temporal_Embedding(Temporal_feature_dim, configs.d_model,
            configs.embed, configs.freq, configs.dropout)
        
        if self.PE_way == 'add':
            if self.WITRAN_dec == 'FC':
                self.fc_1 = nn.Linear(self.num_layers * (self.WITRAN_grid_enc_rows + self.WITRAN_grid_cols) * self.d_model, 
                    self.pred_len * self.d_model)
            elif self.WITRAN_dec == 'Concat':
                self.fc_1 = nn.Linear(self.num_layers * 2 * self.d_model, self.WITRAN_grid_dec_rows * self.d_model)
            self.fc_2 = nn.Linear(self.d_model, self.c_out)
        else:
            if self.WITRAN_dec == 'FC':
                self.fc_1 = nn.Linear(self.num_layers * (self.WITRAN_grid_enc_rows + self.WITRAN_grid_cols) * self.d_model, 
                    self.pred_len * self.d_model)
            elif self.WITRAN_dec == 'Concat':
                self.fc_1 = nn.Linear(self.num_layers * 2 * self.d_model, self.WITRAN_grid_dec_rows * self.d_model)
            self.fc_2 = nn.Linear(self.d_model * 2, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if self.WITRAN_deal == 'standard': # 根据统计选择 选择是否标准化
            seq_last = x_enc[:,-1:,:].detach()
            x_enc = x_enc - seq_last
        
        print("编码器输入：",x_enc.shape)
        print("编码器时间戳：",x_mark_enc.shape)
        print("解码器输入：",x_dec.shape)
        print("解码器时间戳：",x_mark_dec.shape)
        x_input_enc = torch.cat([x_enc, x_mark_enc], dim = -1)
        print("cat x_enc 编码器输入  & x_mark_enc 编码器时间戳 ",x_input_enc.shape)
        batch_size, _, input_size = x_input_enc.shape
        x_input_enc = x_input_enc.reshape(batch_size, self.WITRAN_grid_enc_rows, self.WITRAN_grid_cols, input_size)
        print("x_input_enc.reshape 一维转二维：",x_input_enc.shape)
        print("96 小时 = 4 天 × 24 小时")

        if self.WITRAN_grid_enc_rows <= self.WITRAN_grid_cols:
            flag = 0
        else: # need permute
            flag = 1
        print("self.encoder_2d = WITRAN_2DPSGMU_Encoder 跳到 WITRAN_2DPSGMU_Encoder forward")
        _, enc_hid_row, enc_hid_col = self.encoder_2d(x_input_enc, batch_size, input_size, flag)
        print("self.dec_embedding = WITRAN_Temporal_Embedding 跳到  WITRAN_Temporal_Embedding forward")
        dec_T_E = self.dec_embedding(x_mark_dec)

        if self.WITRAN_dec == 'FC':
            hidden_all = torch.cat([enc_hid_row, enc_hid_col], dim = 2)
            hidden_all = hidden_all.reshape(hidden_all.shape[0], -1)
            last_output = self.fc_1(hidden_all)
            last_output = last_output.reshape(last_output.shape[0], self.pred_len, -1)
            
        elif self.WITRAN_dec == 'Concat':
            enc_hid_row = enc_hid_row[:, :, -1:, :].expand(-1, -1, self.WITRAN_grid_cols, -1)
            output = torch.cat([enc_hid_row, enc_hid_col], dim = -1).permute(0, 2, 1, 3)
            output = output.reshape(output.shape[0], 
                output.shape[1], output.shape[2] * output.shape[3])
            last_output = self.fc_1(output)
            last_output = last_output.reshape(last_output.shape[0], last_output.shape[1], 
                self.WITRAN_grid_dec_rows, self.d_model).permute(0, 2, 1, 3)
            last_output = last_output.reshape(last_output.shape[0], 
                last_output.shape[1] * last_output.shape[2], last_output.shape[3])
            
        if self.PE_way == 'add':
            last_output = last_output + dec_T_E
            if self.WITRAN_deal == 'standard':
                last_output = self.fc_2(last_output) + seq_last
            else:
                last_output = self.fc_2(last_output)
        else:
            if self.WITRAN_deal == 'standard':
                last_output = self.fc_2(torch.cat([last_output, dec_T_E], dim = -1)) + seq_last
            else:
                last_output = self.fc_2(torch.cat([last_output, dec_T_E], dim = -1))
        
        return last_output