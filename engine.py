import torch.optim as optim
from model import *
import util
class trainer():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        # print("-----")
        # print(device)
        # print(num_nodes)
        # print(dropout)
        # print(supports)
        # print(gcn_bool)
        # print(addaptadj)
        # print(aptinit)
        # print()
        # print(in_dim)
        # print(seq_length)
        # print(nhid)
        # print(nhid)
        # print(nhid * 8)
        # print(nhid * 16)
        # print("-----")
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                           aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid,
                           dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)

        # self.fhooks = {}
        # self.bhooks = {}
        # for name, module in self.model.named_modules():
        #     # self.fhooks[name] = module.register_forward_hook(util.hook_f)
        #     self.bhooks[name] = module.register_backward_hook(util.hook_b)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.bob_loss
        # self.loss = util.masked_mae
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # print(input.shape)
        # print(input[0,0, ...])
        input = nn.functional.pad(input,(1,0,0,0))
        # print(input.shape)
        # print(input[0, 0, ...])
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = output

        loss = self.loss(predict, real, w=1000)
        # print(loss)
        loss.backward()
        # print(self.clip)
        # if self.clip is not None:
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        # print("234"+234)
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = output
        loss = self.loss(predict, real, w=1000)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
