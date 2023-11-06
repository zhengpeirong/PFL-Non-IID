import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import MySGD
from flcore.clients.clientbase import Client


class clientNLN(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = MySGD(
            self.init_params_lr_by_layer(), lr=self.learning_rate)
        # NOTE：原base的初始：
        
        self.conf = conf  # 配置信息
        self.local_model = copy.deepcopy(model)  # 本地模型
        self.client_id = id  # 客户端id
        self.train_dataset = train_dataset  # 训练集
        self.train_dataset_idcs = train_dataset_idcs # 训练集下标
        self.dataset_size = self.train_dataset_idcs.size # 不需要自行数数self.get_dataset_size()
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf['batch_size'],
                                                        num_workers=self.conf["num_workers"],
                                                        # pin_memory=True,
                                                        # drop_last=True,
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            self.train_dataset_idcs))
        # SubsetRandomSampler 随机采样from a given list of indices train_dataset_idcs.
        if torch.cuda.is_available():
            self.local_model = self.local_model.cuda()
        self.conv_layers = [module for name, module in self.local_model.named_modules() if (isinstance(module, torch.nn.Conv2d) and module.kernel_size != (1, 1)) or isinstance(module, torch.nn.Linear) ]
        self.conv_layers_names = [name for name, module in self.local_model.named_modules() if (isinstance(module, torch.nn.Conv2d) and module.kernel_size != (1, 1)) or isinstance(module, torch.nn.Linear) ]
        self.num_conv = len(self.conv_layers) #权重层数量
        self.num_class = len(self.train_loader.dataset.class_to_idx) #类别数量
        self.param_dict = {}
        self.param_dict['weight_decay'] = self.conf["weight_decay"]
        self.param_dict['momentum'] = self.conf["momentum"]
        self.param_dict['lr'] = self.conf['lr']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # NOTE: fednln特定的初始化
        # 训练数据加载器
        self.activation = []# 保存每一个权重层的输出
        # 获取所有权重层，resnet18的17个卷积层和1个全连接层
        self.conv_layers = [module for name, module in self.local_model.named_modules() if (isinstance(module, torch.nn.Conv2d) and module.kernel_size != (1, 1)) or isinstance(module, torch.nn.Linear) ]
        self.conv_layers_names = [name for name, module in self.local_model.named_modules() if (isinstance(module, torch.nn.Conv2d) and module.kernel_size != (1, 1)) or isinstance(module, torch.nn.Linear) ]
        self.num_conv = len(self.conv_layers) #权重层数量
        self.num_class = len(self.train_loader.dataset.class_to_idx) #类别数量
        # result_list (类别10*权重层数量18)的列表，存储列表activation
        self.result_list = [[[] for _ in range(self.num_conv)] for _ in range(self.num_class)]
        self.TF = [[[] for _ in range(self.num_conv)] for _ in range(self.num_class)]
        self.channel_lr_multiplier = [[] for _ in range(self.num_conv)]
        self.verbose =  False # 仅仅用于快速测试
        self.label_proportions = {}
        self.inc = self.conf["mul"] # 让缩放因子的最小值变成1
        self.pre_mul = self.conf["pre_mul"]  # 等于1时，数学上不能计算，不预处理。
        self.last_layer_index = -1  # 用于打印最后一层的缩放因子
        self.dataset_size = self.train_dataset_idcs.size # 不需要自行数数self.get_dataset_size()
        self.param_dict.pop('lr')
    
    def init_params_lr_by_layer(self):
        r"""conv_layers_names->params_lr
        每次训练之前初始化，确定哪些参数需要训练，初始学习率缩放全为1
        主要是为了将params分组，这样每一层都可以设置不同的学习率，适合CL方法与RS方法"""
        params_lr = []
        param_dict = copy.deepcopy(self.param_dict)
        # HACK: 初始化时，自带lr
        if param_dict.get('lr') == None:
            param_dict['lr'] = self.conf['lr']
        for i, layer_name in enumerate(self.conv_layers_names): 
            found_layer_weight = False
            found_layer_bias = False
            for name, param in self.local_model.named_parameters():
                if found_layer_weight and found_layer_bias: continue # 'conv'层有weight和bias
                if layer_name == name[:len(layer_name)]:
                    if 'weight' in name:
                        params_lr.append({'params': param, **param_dict})
                        found_layer_weight = True 
                    elif 'bias' in name:
                        params_lr.append({'params': param, **param_dict})
                        found_layer_bias = True
        self.last_layer_index = len(params_lr)-1
        # 是否只设置权重层的学习率
        if not self.conf['only_train_convs']:
            for name, param in self.local_model.named_parameters():# 遍历本地模型的所有参数
                param_handled = False # 标志是否已经被上面的处理
                for handled_param_group in params_lr:
                    if param is handled_param_group["params"]:
                        param_handled = True
                        break
                if not param_handled:
                    params_lr.append({"params": param, **param_dict})
        return params_lr
    
    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.global_params, self.device)

        # self.model.cpu()

        # if self.learning_rate_decay:
        #     self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)

                gm = torch.cat([p.data.view(-1) for p in self.global_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm-pm, p=2)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
