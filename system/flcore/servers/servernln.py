from flcore.clients.clientnln import clientNLN
from flcore.servers.serverbase import Server
from threading import Thread


class FedNLN(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientNLN)


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        # NOTE: base的初始化：
        self.conf = conf
        self.num_class = len(eval_dataset.class_to_idx)
        self.global_model = self.init_model(conf['model_name'],self.num_class)
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, 
                                                       batch_size=self.conf["batch_size"],
                                                       num_workers=self.conf["num_workers"],
                                                       shuffle=True,
                                                    #    pin_memory=True,
                                                       )  # 创建测试集加载器用于测试最终的聚合模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = self.global_model.to(self.device)

        # NOTE: fednln特定的初始化
        self.conv_layers = [module for name, module in self.global_model.named_modules() if (isinstance(module, torch.nn.Conv2d) and module.kernel_size != (1, 1)) or isinstance(module, torch.nn.Linear) ]
        self.conv_layers_names = [name for name, module in self.global_model.named_modules() if (isinstance(module, torch.nn.Conv2d) and module.kernel_size != (1, 1)) or isinstance(module, torch.nn.Linear) ]
        self.num_conv = len(self.conv_layers)
        self.num_class = len(self.eval_loader.dataset.class_to_idx)
        # self.result_list = [[torch.Tensor() for _ in range(column)] for _ in range(row)]
        self.result_list = [[np.array([], dtype=np.float64) for _ in range(self.num_conv)] for _ in range(self.num_class)]
        # 空壳子：channel_frequency仅仅用来创建新的相同维度的变量。
        self.channel_frequency = [[ np.array([], dtype=np.float32) for _ in range(self.num_conv)] for _ in range(self.num_class)]
        self.TF = copy.deepcopy(self.channel_frequency)
        self.TF_IDF = copy.deepcopy(self.channel_frequency)
        self.verbose = self.conf["debug_mode"]
        self.IDF = [np.array([]) for _ in range(self.num_conv)] 
        self.layer_mul = np.ones(self.num_conv,dtype=np.float32)+0.01
        conv_layers_dimension_list= [module.weight.shape[0] for name, module in self.global_model.named_modules() if (isinstance(module, torch.nn.Conv2d) and module.kernel_size != (1, 1)) or isinstance(module, torch.nn.Linear) ]
        self.formula_layer_mul =  1 + np.log10(np.array(conv_layers_dimension_list) )* 0.75
        #mul的计算，可以用fitting.py拟合
        for index, out_features in enumerate(conv_layers_dimension_list):
            self.formula_layer_mul[index] =  1 + 0.45 * (index+1)/self.num_conv + 0.3*np.log10(np.array(out_features))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_model(self,model_name,num_class):
        if hasattr(myCNN, model_name): # 先查看是否本地已经构建。
            model = eval('myCNN.{}()'.format(model_name)) 
        else:
            model = eval('models.{}()'.format(model_name))  # 从配置文件获得模型名称并创建服务器模型
            if "resnet" in model_name:
                # 获取原始模型的全连接层的输入维度
                in_features = model.fc.in_features
                # 定义新的全连接层
                new_fc = nn.Linear(in_features,num_class)  # 10是CIFAR-10数据集的类别数量
                # 替换ResNet18的最后一层
                model.fc = new_fc
        self.init_weights(model)  # 初始化权重
        return model
    def init_weights(self,model):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                print(f'{m.__class__.__name__}.weight({m.weight.shape}) 已初始化')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                    print(f'{m.__class__.__name__}.bias({m.bias.shape}) 已初始化')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.GroupNorm):
                torch.nn.init.constant_(m.weight, 1.0)
                torch.nn.init.zeros_(m.bias)
                print(f'{m.__class__.__name__}.weight({m.weight.shape}) 已初始化')
                print(f'{m.__class__.__name__}.bias({m.bias.shape}) 已初始化')
    
    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientProx)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
