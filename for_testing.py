import torch
import torch.nn as nn
import numpy as np
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        module_list = [nn.Linear(1, 1, bias=False) for i in range(1, 5)]
        self.module = nn.ModuleList(module_list)

    def forward(self, x):
        result = []
        for m in self.module:
            x = m(x)
            result.append(x)
        return result


def run():
    net = Net()
    # net.requires_grad_(False)

    criterion = nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), lr=0.1)

    input = torch.ones(1)

    # print(net)
    # print(len(output))
    print("#" * 10 + "Input" + "#" * 10)
    print(input.is_leaf, input)

    print("#" * 10 + "GT" + "#" * 10)
    gt = []
    for i in range(1, 5):
        gt.append(torch.ones(1))
        print(gt[i - 1])

    print("#" * 10 + "Init Params" + "#" * 10)
    for p in net.parameters():
        p.data = torch.ones_like(p.data) * 2
        print(p, p.grad)

    output = net(input)
    print("#" * 10 + "Result" + "#" * 10)
    for i in range(len(output)):
        # output[i].requires_grad_(True)
        print(output[i].is_leaf, output[i])

    optim.zero_grad()
    print("#" * 10 + "After zero_grad" + "#" * 10)
    for p in net.parameters():
        # p.requires_grad_(True)
        print(p, p.grad)

    loss = []
    for i in range(len(output)):
        loss.append(criterion(output[i], gt[i]))

    avg_loss = torch.mean(torch.tensor(loss, requires_grad=True))
    # avg_loss = 0
    # for lo in loss:
    #     avg_loss += lo
    # avg_loss /= len(loss)
    # print(len(loss))

    # avg_loss = 0
    print("#" * 10 + "Every loss" + "#" * 10)
    for l in loss:
        l.retain_grad()
        print(l.is_leaf, l.item(), l.grad, l.grad_fn)

    # avg_loss /= len(loss)
    print("#" * 10 + "Avg loss" + "#" * 10)
    # avg_loss.retain_grad()
    print(avg_loss)
    print(avg_loss.is_leaf, avg_loss.data, avg_loss.grad_fn)

    avg_loss.backward()
    optim.step()

    print("#" * 10 + "Gradient update" + "#" * 10)
    for p in net.parameters():
        print(p.is_leaf, p.data, p.grad, p.grad_fn)

    print("#" * 10 + "Output" + "#" * 10)
    for i in range(len(output)):
        output[i].requires_grad_(True)
        print(output[i].is_leaf, output[i])

    print("#" * 10 + "Result Loss Info" + "#" * 10)
    for l in loss:
        print(l.is_leaf, l.item(), l.grad, l.grad_fn)


if __name__ == '__main__':
    import pandas as pd
    from tqdm.auto import tqdm
    top1=pd.read_excel('top1.xlsx')
    top2=pd.read_excel('top2.xls')
    print(len(top1['id']))
    new_top1 = top1['id'].unique()
    print(len(new_top1))

    print(len(top2['id']))
    new_top2 = top2['id'].unique()

    print(len(new_top2))
    tag1= 0
    tag2=0
    dep = []
    for i,id1_ in tqdm(top1.iterrows()):
        id1 = str(id1_['id'])
        if id1 in new_top2:
            dep.append(id1)
    print(len(dep))
    print(top2)
    for did in dep:
        top2['exit'][top2['id']==did]=1

    # print(i,j,i+j)

    df = pd.DataFrame(top2)
    df.to_excel('top2-1.xlsx',index=False)
    # a = np.random.random(size=(512, 512, 3))
    # b = np.mean(np.mean(a, axis=0), axis=0)
    #
    # print(b)
    # print(b.shape)
    # net = Net()
    # # net.requires_grad_(False)
    #
    # criterion = nn.MSELoss()
    # optim = torch.optim.SGD(net.parameters(), lr=0.1)
    #
    # input = torch.ones(1)

    # gt = []
    # for i in range(1, 5):
    #     gt.append(torch.ones(1))
    #
    # for p in net.parameters():
    #     p.data = torch.ones_like(p.data) * 2
    #
    # output = net(input)
    #
    # optim.zero_grad()
    #
    # loss = []
    # avg_loss = 0
    # for i in range(len(output)):
    #     loss.append(criterion(output[i], gt[i]))
    #     avg_loss += criterion(output[i], gt[i])
    # avg_loss/=len(output)
    # # avg_loss = torch.mean(torch.tensor(loss, requires_grad=True))
    #
    # # avg_loss = torch.zeros(1)
    # # for lo in loss:
    # #     avg_loss += lo
    # # avg_loss /= len(loss)
    #
    # avg_loss.backward()
    # optim.step()
    #
    # print("#" * 10 + "Gradient update" + "#" * 10)
    # for p in net.parameters():
    #     print(p.is_leaf, p.data, p.grad, p.grad_fn)
