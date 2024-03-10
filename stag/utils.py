import torch
import torchvision.transforms as T
import torch.nn as nn
import numpy as np


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def normalize_parameters(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = nn.functional.normalize(param.data, p=2, dim=0)

def parameter_size(model):
    print(sum(p.numel() for p in model.parameters()))


def printing_parameters(model, keyword='def', record=False):
    for name, param in model.named_parameters():
        print(name)
        print(param)
        num = param.detach().numpy()
        if record:
            np.savetxt('./txt/' + str(keyword) + name + '.txt', num.reshape(-1, num.shape[-1]))


def valueing_model(model):
    for name,parameter in model.named_parameters():
        print(name, torch.any(torch.isnan(parameter)),parameter.shape)


def initialize_parameters(model):
    for name, parameter in model.named_parameters():
        if len(parameter.shape) == 1:
            parameter.data =torch.randn(parameter.shape).detach()
        else:
            nn.init.xavier_uniform_(parameter)


def DrawingParameters(LearningNet, crypto_name):
    transform = T.ToPILImage()

    for name, parameter in LearningNet.named_parameters():
        if len(parameter.shape) == 4:
            mini = parameter.min().min().min().min()
            changed_img = (parameter - mini)
            changed_img = changed_img / changed_img.max() * 255
            MetaDatas = changed_img.mean(dim=0).chunk(3, dim=0)
            DataList = []
            for Data in MetaDatas:
                DataList.append(Data.mean(dim=0))
            MetaData = torch.stack(DataList, dim=0)
            img = transform(MetaData)
            img.save('./ModelPrint/' + crypto_name + '/' + name + '.png')
        elif len(parameter.shape) == 3:

            if parameter.shape[2] == 3:
                changed_img = (parameter.permute(2, 0, 1) - parameter.permute(2, 0, 1).min().min())
                changed_img = changed_img / changed_img.max() * 255

                img = transform(changed_img)
            else:
                changed_img = (parameter - parameter.min())
                changed_img = changed_img / changed_img.max() * 255
                img = transform(changed_img)

            img.save('./ModelPrint/' + crypto_name + '/' + name + '.png')
        elif len(parameter.shape) == 2:
            mini = parameter.min()
            changed_img = (parameter - mini.min())
            changed_img = changed_img / changed_img.max() * 255
            img = transform(changed_img.unsqueeze(dim=0))
            img.save('./ModelPrint/' + crypto_name + '/' + name + '.png')
        elif len(parameter.shape) == 1:
            changed_img = (parameter - parameter.min())
            changed_img = changed_img / changed_img.max() * 255
            img = transform(changed_img.reshape(1, 1, -1))
            img.save('./ModelPrint/' + crypto_name + '/' + name + '.png')


class GlobalAdamW(torch.optim.AdamW):
    def __init__(self, params, lr, weight_decay=1e-5):
        super(GlobalAdamW, self).__init__(params, lr=lr, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0

                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class GlobalSGD(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay=1e-5):
        super(GlobalSGD, self).__init__(params, lr=lr, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
