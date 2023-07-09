import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class NodeConvolution(Module):

    def __init__(self, kernel, input_size, pooling_size=2):
        super(NodeConvolution, self).__init__()
        self.pooling_size = pooling_size
        self.weight1 = Parameter(torch.FloatTensor(kernel, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)

    def forward(self, input_tensor):
        batch_list = list()
        for i in range(input_tensor.shape[0]):
            tensor = input_tensor[i]
            batch_size = tensor.shape[0]
            steps = batch_size // self.pooling_size
            left_tensor = False
            if batch_size % self.pooling_size != 0:
                steps += 1
                left_tensor = True
            tensor_list = list()
            for j in range(steps):
                if left_tensor is True and j == steps - 1:
                    tensor_ = torch.zeros(self.pooling_size, tensor.shape[1], requires_grad=True).cuda()
                    for t in range(batch_size % self.pooling_size):
                        tensor_[t] = tensor[self.pooling_size * j + t]
                    tensor_list.append(torch.sum(tensor_ * self.weight1, dim=0).requires_grad_(True))
                else:
                    tensor_list.append(
                        torch.sum(
                            tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size] * self.weight1,
                            dim=0).requires_grad_(True))

            batch_list.append(torch.stack(tensor_list, dim=0))

        return torch.stack(batch_list, dim=0).requires_grad_(True)


class GraphPooling(Module):

    def __init__(self, pooling_size=3):
        super(GraphPooling, self).__init__()
        self.pooling_size = pooling_size

    def forward(self, input_tensor, graph):
        batch_list = []
        for i in range(input_tensor.shape[0]):
            tensor = input_tensor[i]
            batch_size = tensor.shape[0]
            steps = batch_size // self.pooling_size
            left_tensor = False
            if batch_size % self.pooling_size != 0:
                steps += 1
                left_tensor = True
            tensor_list = list()
            for j in range(steps):
                shot_boundary = graph[i][j]
                tensor_list.append(
                    torch.div(torch.sum(input_tensor[shot_boundary[0]:shot_boundary[1] + 1], dim=0).requires_grad_(True),
                              shot_boundary[1] - shot_boundary[0] + 1).requires_grad_(True) + 6e-3)
                if left_tensor is True and j == steps - 1:
                    tensor_ = torch.zeros(self.pooling_size, tensor.shape[1], requires_grad=True).cuda()
                    for t in range(batch_size % self.pooling_size):
                        tensor_[t] = tensor[self.pooling_size * j + t]

                    tensor_list.append(torch.div(torch.sum(tensor_, dim=0).requires_grad_(True), self.pooling_size))
                else:
                    tensor_list.append(
                        torch.div(torch.sum(tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size],
                                            dim=0).requires_grad_(True), self.pooling_size))
            batch_list.append(torch.stack(tensor_list, dim=0).requires_grad_(True))

        return torch.stack(batch_list, dim=0).requires_grad_(True)


class GraphAttentionPooling(Module):

    def __init__(self, in_features, pooling_size=3):
        super(GraphAttentionPooling, self).__init__()
        self.in_features = in_features
        self.W = nn.Linear(in_features, 1, bias=True)
        self.pooling_size = pooling_size

    def forward(self, batch_tensor):
        batch_list = list()
        for i in range(batch_tensor.shape[0]):
            tensor = batch_tensor[i]
            batch_size = tensor.shape[0]
            steps = batch_size // self.pooling_size
            left_tensor = False
            if batch_size % self.pooling_size != 0:
                steps += 1
                left_tensor = True

            tensor_list = list()
            for j in range(steps):
                if left_tensor is True and j == steps - 1:
                    tensor_ = torch.zeros(self.pooling_size, tensor.shape[1], requires_grad=True).cuda()
                    for t in range(batch_size % self.pooling_size):
                        tensor_[t] = tensor[self.pooling_size * j + t]
                    att_w = F.softmax(self.W(tensor_), dim=0).requires_grad_(True)
                    tensor_list.append(tensor_.T @ att_w)
                else:
                    att_w = F.softmax(
                        self.W(tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size]),
                        dim=0).requires_grad_(True)
                    tensor_list.append(
                        tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size].T.matmul(att_w))
            batch_list.append(torch.stack(tensor_list, dim=0))
        return torch.stack(batch_list, dim=0).requires_grad_(True)


'''
   test = torch.isnan(adj)

        for i in range(test.shape[1]):
            for j in range(test.shape[2]):
                if test[0][i][j].item():
                    print("gere")
'''


class DCGNPropagate(nn.Module):
    def __init__(self, input_tensor, output):
        super(DCGNPropagate, self).__init__()
        self.Weight = Parameter(torch.FloatTensor(input_tensor, output))
        self.Bias = Parameter(torch.FloatTensor(output))

    def forward(self, adj, x):
        graphed_net = torch.einsum('adb,abc->adc', adj, x)

        x = torch.matmul(graphed_net, self.Weight) + self.Bias

        return x


class DCGN(nn.Module):
    def __init__(self, input_tensor, nclass, pooling_size=4):
        super(DCGN, self).__init__()

        self.nodewiseconvolution = NodeConvolution(pooling_size, input_tensor, pooling_size=pooling_size)
        self.WisePooling = GraphAttentionPooling(input_tensor, pooling_size=pooling_size)
        self.Propagate1 = DCGNPropagate(input_tensor, 1100)

        self.CosSimillarity = nn.CosineSimilarity(dim=1)
        self.activation1 = nn.LeakyReLU()

        self.NodeConvolution2 = NodeConvolution(pooling_size, 1100, pooling_size=pooling_size)
        self.AttentionPooling2 = GraphAttentionPooling(1100, pooling_size=pooling_size)
        self.Propagate2 = DCGNPropagate(1100, nclass)
        self.activation2 = nn.LeakyReLU()

    def forward(self, x):

        adj = self.WisePooling(x)
        x = self.nodewiseconvolution(x)  # 2,256
        adj = self.get_adjacent(adj).requires_grad_(True).cuda()  # 2,2

        x = self.Propagate1(adj, x)
        x = self.activation1(x)

        adj2 = self.AttentionPooling2(x)  # 2,64
        x = self.NodeConvolution2(x)  # 2,64
        adj2 = self.get_adjacent(adj2).requires_grad_(True).cuda()
        x = self.Propagate2(adj2, x)
        x = self.activation2(x)
        return x

    def get_adjacent(self, matrix):
        batch_list = list()
        for t in range(matrix.shape[0]):
            tensor = matrix[t]
            matrix_frame = tensor.shape[0]  # 4,2,1024
            AdjacentMatrix = torch.zeros(matrix_frame, matrix_frame)  # 2 X 2

            chunks = torch.chunk(tensor, matrix_frame, dim=0)
            for i in range(matrix_frame):
                for j in range(matrix_frame - i):
                    AdjacentMatrix[j][i] = self.CosSimillarity(chunks[i], chunks[j])
                    if not i == j:
                        AdjacentMatrix[j][i] = AdjacentMatrix[i][j]
            IdentityMatrix = torch.eye(AdjacentMatrix.shape[0], requires_grad=True)

            AdjacentMatrix += IdentityMatrix
            AdjacentMatrix = AdjacentMatrix.requires_grad_(True)
            D_hat = torch.sum(AdjacentMatrix, dim=0)
            D_hat = torch.linalg.inv(torch.sqrt(torch.diag(D_hat)))

            batch_list.append(D_hat @ AdjacentMatrix @ D_hat)

        return torch.stack(batch_list, dim=0).requires_grad_(True)
