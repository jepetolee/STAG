import torch
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import snntorch as snn
import snntorch.surrogate as surrogate
from SNNRESNET import ResNet,Bottleneck
class WisePooling(Module):

    def __init__(self):
        super(WisePooling, self).__init__()

    def forward(self, input, graph):
        tensor_list = list()
        for j in range(graph.shape[0]):
            shot_boundary = graph[j]
            tensor_list.append(
                torch.div(torch.sum(input[shot_boundary[0]:shot_boundary[1] + 1], dim=0).requires_grad_(True),
                          shot_boundary[1] - shot_boundary[0] + 1).requires_grad_(True) + 6e-3)
        return torch.stack(tensor_list, dim=0).requires_grad_(True)


class WiseConvolution(Module):

    def __init__(self, input_size, output_size):
        super(WiseConvolution, self).__init__()
        self.WiseConv = nn.Linear(input_size, output_size)

    def forward(self, input, graph):
        tensor_list = list()
        for j in range(graph.shape[0]):
            shot_boundary = graph[j]

            tensor_list.append(
                torch.sum(self.WiseConv(input[shot_boundary[0]:shot_boundary[1] + 1]), dim=0).requires_grad_(True))

        return torch.stack(tensor_list, dim=0).requires_grad_(True)


class NodeConvolution(Module):

    def __init__(self, kernel, input_size, pooling_size=2):
        super(NodeConvolution, self).__init__()
        self.pooling_size = pooling_size
        self.weight1 = Parameter(torch.FloatTensor(kernel, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_list = list()
        for i in range(input.shape[0]):
            tensor = input[i]
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
                    for i in range(batch_size % self.pooling_size):
                        tensor_[i] = tensor[self.pooling_size * j + i]
                    tensor_list.append(torch.sum(tensor_ * self.weight1, dim=0).requires_grad_(True))
                else:
                    tensor_list.append(
                        torch.sum(
                            tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size] * self.weight1,
                            dim=0).requires_grad_(True))
            batch_list.append(torch.stack(tensor_list, dim=0))

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
                    for i in range(batch_size % self.pooling_size):
                        tensor_[i] = tensor[self.pooling_size * j + i]
                    att_w = F.softmax(self.W(tensor_), dim=0).requires_grad_(True)
                    tensor_list.append(tensor_.T @ att_w)
                else:
                    att_w = F.softmax(
                        self.W(tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size]),
                        dim=0).requires_grad_(True)
                    tensor_list.append(
                        tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size].T @ att_w)
            batch_list.append(torch.stack(tensor_list, dim=0))
        return torch.stack(batch_list, dim=0).requires_grad_(True)

# v1 1024 256 64 16   4
# v2 1024 512 128 32  4
# v3 1024 512 256 128 8

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()
        self.feature_extract = ResNet(Bottleneck, [3, 4, 6,8, 3])
        self.dcgn = DCGN(1960, 2)

    def forward(self, input, device):
        batch_size = input.shape[0]
        arrays = list()
        for i in range(batch_size):
            arrays.append(self.feature_extract(input[i]))
        x = torch.stack(arrays, dim=0)
        x = x.view(batch_size, 50, -1)
        return self.dcgn(x, device)



class DCGNPropagate(nn.Module):
    def __init__(self, input, output):
        super(DCGNPropagate, self).__init__()
        self.Weight = Parameter(torch.FloatTensor(input, output))
        self.Bias = Parameter(torch.FloatTensor(output))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Weight.size(1))
        self.Weight.data.uniform_(-stdv, stdv)

    def forward(self, adj, x):
        batch_list = list()
        for i in range(x.shape[0]):
            batch_list.append(adj[i] @ x[i] @ self.Weight + self.Bias)
        return torch.stack(batch_list, dim=0).requires_grad_(True)


class DCGN(nn.Module):
    def __init__(self, input, nclass, pooling_size=4):
        super(DCGN, self).__init__()

        self.nodewiseconvolution = NodeConvolution(pooling_size, input, pooling_size=pooling_size)
        self.WisePooling = GraphAttentionPooling(input, pooling_size=pooling_size)
        self.Propagate1 = DCGNPropagate(input, 245)
        self.Lif1 = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.NodeConvolution2 = NodeConvolution(pooling_size, 245, pooling_size=pooling_size)
        self.AttentionPooling2 = GraphAttentionPooling(245, pooling_size=pooling_size)
        self.Propagate2 = DCGNPropagate(245, 35)
        self.Lif2 = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True)
        self.classifier = nn.Sequential( nn.Linear(6*35,32),
                                         snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True),
                                         nn.Linear(32,nclass))


    def forward(self, x, device):
        membrane1 = self.Lif1.init_leaky()
        membrane2 = self.Lif2.init_leaky()

        adj = self.WisePooling(x)
        x = self.nodewiseconvolution(x)  # 2,256
        adj = self.get_adjacent(adj).to(device).requires_grad_(True)  # 2,2
        x = self.Propagate1(adj, x)
        x = self.Lif1(x,membrane1)

        adj = self.AttentionPooling2(x)  # 2,64
        x = self.NodeConvolution2(x)  # 2,64
        adj = self.get_adjacent(adj).to(device).requires_grad_(True)  # 2,32.
        x = self.Propagate2(adj, x)
        x = self.Lif2(x,membrane2)

        x = x.view(-1, 6*28)
        x = self.classifier(x)
        return x

    def cosine_similarity_adjacent(self, matrix1, matrix2):
        squaresum1 = torch.sum(torch.square(matrix1), dim=1)  # 1024 to 1

        squaresum2 = torch.sum(torch.square(matrix2), dim=1)  # 1024 to 1

        multiplesum = torch.sum(torch.multiply(matrix1, matrix2), dim=1)

        Matrix1DotProduct = torch.sqrt(squaresum1)
        Matrix2DotProduct = torch.sqrt(squaresum2)
        cosine_similarity = torch.div(multiplesum, torch.multiply(Matrix1DotProduct, Matrix2DotProduct))
        return cosine_similarity

    def get_adjacent(self, matrix):
        batch_list = list()
        for i in range(matrix.shape[0]):
            tensor = matrix[i]
            matrix_frame = tensor.shape[0]  # 4,2,1024
            AdjacentMatrix = torch.zeros(matrix_frame, matrix_frame)  # 2 X 2

            chunks = torch.chunk(tensor, matrix_frame, dim=0)
            for i in range(matrix_frame):
                for j in range(matrix_frame - i):
                    AdjacentMatrix[j][i] = self.cosine_similarity_adjacent(chunks[i], chunks[j])
                    if not i == j:
                        AdjacentMatrix[j][i] = AdjacentMatrix[i][j]
            I = torch.eye(AdjacentMatrix.shape[0], requires_grad=True)

            AdjacentMatrix += I
            AdjacentMatrix = AdjacentMatrix.requires_grad_(True)
            D_hat = torch.sum(AdjacentMatrix, dim=0)
            D_hat = torch.linalg.inv(torch.sqrt(torch.diag(D_hat)))
            batch_list.append(D_hat @ AdjacentMatrix @ D_hat)
        return torch.stack(batch_list, dim=0).requires_grad_(True)