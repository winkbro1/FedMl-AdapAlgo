import copy
import logging
import random
import time
import numpy as np
import torch
import math
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client
class MecAlgoAPI(object):
    def __init__(self,args,device,dataset,model):
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset

        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = create_model_trainer(model, args)
        self.model = model

        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
        )

    def _setup_clients(self,train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        global gradtest
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        tau = self.args.epochs
        eta = self.args.learning_rate
        print("---------",tau)
        comm_round = self.args.comm_round
        sum_time =0
        aggregate_time = 0
        control_param = 10
        param_a = 0.0
        time_per_round = 0
        node_train_time = 0
        R_TIME = 500
        PHI = 0.2
        beta = 0.0
        #R_TIME = 420
        nodes_train_time = []
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))
            w_locals = []
            nodes_time = []

            grad_local_per_node_list = []
            beta_i_list = []


            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            logging.info("client_indexes = " + str(client_indexes))

            if round_idx >= 1:
                """
                if self._Within_budget(R_TIME,sum_time,tau,nodes_train_time,aggregate_time,round_idx,comm_round):
                    tau = self._compute_new_tau(tau,param_a,eta,beta,PHI,control_param)
                    if not self._Within_budget(R_TIME,sum_time,tau,nodes_train_time,aggregate_time,round_idx,comm_round):                   #tau = self._compute_new_tau(beta,eta)
                        tau = self._decrease_tau(R_TIME,sum_time,nodes_train_time,aggregate_time,round_idx,comm_round)
                    print("round:",round_idx+1,"yes")
                    """
                if not self._Within_budget(R_TIME,sum_time,tau,nodes_train_time,aggregate_time,round_idx,comm_round):
                    tau = self._decrease_tau(R_TIME,tau,sum_time,nodes_train_time,aggregate_time,round_idx,comm_round)
                    #print("round:",round_idx+1,"No")
                #else:
                    #print("round:",round_idx+1,"Yes")

            #elif (round_idx >= 1) and(round_idx == self.args.comm_round-1 ):
                #tau = tau -2
            print("round---->",round_idx+1,"current tau:",tau)
            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )
                # train on new dataset
                node_start_time = time.time()
                w = client.train(copy.deepcopy(w_global),tau,round_idx)
                node_time = time.time()-node_start_time
                nodes_time.append(node_time)
                grad_local_per_node = client.gradient(w)


                # self.logging.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                w_globaltest = self._aggregate(w_locals)

                grad_local_per_node_list.append(copy.deepcopy(grad_local_per_node))  #本地更新之后每个client的损失函数梯度
                gradtest = client.gradient(w_globaltest) #全局聚合之后的梯度
                #print("w_globaltest ------->",w_globaltest)

            aggregate_start_time = time.time()
            w_global = self._aggregate(w_locals)
            aggregate_time = (time.time()-aggregate_start_time) * 1000
            #print("global gard----->",gradtest)

            #开始计算beta
            beta = self._compute_beta(grad_local_per_node_list,gradtest,w_locals,w_global)
            #print("local beta------>",round(beta,6))
            gradtest = 0

            node_train_time = np.mean(nodes_time) * 1000
            #time_per_round = node_train_time + aggregate_time
            #print(nodes_time) 输出每个client训练时间的列表
            #print("round:", round_idx+1, "-------","time_per_round---->",node_train_time + aggregate_time )
            sum_time = sum_time + (tau * node_train_time) +aggregate_time
            param_a = aggregate_time / node_train_time
            if round_idx == 0:
                node_train_time = np.mean(nodes_time[1:]) * 1000
            nodes_train_time.append(node_train_time)
            print("sum time---->", sum_time,"mean node train time----->",node_train_time)
            self.model_trainer.set_model_params(w_global)
            tau = self._compute_new_tau(param_a, eta, beta, PHI, control_param)


            #test result
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)



    def _compute_beta(self,grad_local_list,grad_global,w_locals,w_global):
        i = 0
        beta = 0
        training_num =0
        for j in range(len(w_locals)-1):
            training_num = training_num + w_locals[j+1][0]
        for i in range(len(grad_local_list)-1):
            dif_of_grad = np.linalg.norm(grad_local_list[i+1] - grad_global)
            dif_of_w = np.linalg.norm(w_locals[i+1][1]['linear.bias'] - w_global['linear.bias'])
            beta_i = (dif_of_grad * w_locals[i+1][0]) / (dif_of_w * training_num)
            beta = beta + beta_i
        return beta


    def _compute_new_tau(self,param_a,eta,beta,PHI,control_param):
        max_tau = 1
        tau = self.args.epochs
        current_g = self._compute_G_tau(1,param_a,eta,beta,PHI)
        for i in range(1,(tau*control_param)):
            if self._compute_G_tau(i,param_a,eta,beta,PHI) > current_g:
                max_tau = i
        return max_tau

    def _compute_h_tau(self,tau,beta,eta):
        '''h_delta = 0.005
        h_tau = tau
        h_beta = beta
        h_eta = eta
        h_tau = ((h_delta *100) / h_beta)(math.pow(((h_eta * h_beta)+1),h_tau)-1) - (h_eta * h_delta) * h_tau'''
        h_h_tau = 0.001
        return h_h_tau

    def _compute_G_tau(self,tau,param_a,eta,beta,PHI):
        g_tau = tau
        g_param_a = param_a
        g_eta = eta
        g_beta = beta
        g_h_tau = self._compute_h_tau(g_tau,g_beta,g_eta)
        g_1 = g_tau / (g_tau +g_param_a)
        g_2 = 1 - (g_beta*g_eta)/2
        g_3 = (g_h_tau * PHI)/g_tau
        #print("g1--->",g_1,"g2--->",g_2,"g3---",g_3)
        g =g_1 * (g_eta *g_2-g_3)
        #print("g is",g)
        return g

    def _decrease_tau(self,R_TIME,tau,sum_time,nodes_train_time,aggregate_time,round_idx,comm_round):
        custom_tau = tau
        round_cur = round_idx
        round_remain = comm_round - round_cur
        mean_node_train_time = np.mean(nodes_train_time)
        tau = (((R_TIME - sum_time)/round_remain)-aggregate_time)/mean_node_train_time
        tau = int(tau)
        if tau > custom_tau:
            tau = custom_tau
        print("new tau :",tau)
        return tau

    def _Within_budget(self,R_TIME,sum_time,tau,nodes_train_time,aggregate_time,round_idx,comm_round):
        round_cur = round_idx
        round_remain = comm_round - round_cur
        mean_node_train_time = np.mean(nodes_train_time)
        #print("round---->",round_cur,"mean_node_train_time------->",mean_node_train_time)
        #print("remain time:",R_TIME-sum_time,mean_node_train_time,"*",tau,"aggregate:",aggregate_time,"round remain:",round_remain)
        if (R_TIME - sum_time) < ((mean_node_train_time * tau) + aggregate_time) * round_remain :
            return False
        else :
            return True



    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # test on test dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

        stats = {"training_acc": train_acc, "training_loss": train_loss}

        logging.info(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}

        logging.info(stats)

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}

        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)