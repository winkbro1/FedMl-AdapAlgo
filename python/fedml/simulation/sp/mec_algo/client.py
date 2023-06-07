import torch
class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        return self.local_sample_number

    def train(self,w_global,tau_new,round_idx):
        w_local_pre = w_global
        self.model_trainer.set_model_params(w_local_pre)
        self.model_trainer.new_train(self.local_training_data, self.device, self.args,tau_new,round_idx)
        w_real_local = self.model_trainer.get_model_params()
        return w_real_local

    def gradient(self,w):
        self.model_trainer.set_model_params(w)
        grad = self.model_trainer.gradient(self.local_training_data,self.device,self.args)
        return grad

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics