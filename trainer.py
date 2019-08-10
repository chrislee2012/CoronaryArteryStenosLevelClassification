import torch
from torchvision import models, transforms
import torch.nn.functional as F
from torch.optim import Adam
from ignite.metrics import Accuracy, Loss, Precision, Recall, MetricsLambda
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from datasets.mpr_dataset import MPR_Dataset
from datasets.sampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import inspect
import importlib
import os
import yaml

class Trainer:
    def __init__(self, config):
        self.config = config

        os.makedirs(self.config['experiments_path'], exist_ok=True)
        self.id = len(os.listdir(self.config['experiments_path'])) + 1
        self.path = os.path.join(self.config['experiments_path'], "exp{}".format(self.id))
        os.makedirs(self.path, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.__save_config()
        self.__load_model()
        self.__load_optimizer()
        self.__load_loaders()
        self.__load_metrics()
        self.__create_pbar()
        self.__create_evaluator()
        self.__create_trainer()

    def __module_mapping(self, module_name):
        mapping = {}
        for name, obj in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
            mapping[name] = obj
        return mapping

    def __save_config(self):
        config_path = os.path.join(self.path, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def __load_model(self):
        mapping = self.__module_mapping('models')
        self.config['model']['parameters']['n_classes'] = len(self.config['data']['groups'])
        self.model = mapping[self.config['model']['name']](**self.config['model']['parameters'])

    def __load_optimizer(self):
        mapping = self.__module_mapping('torch.optim')
        self.optimizer = mapping[self.config['optimizer']['name']](self.model.parameters(), **self.config['optimizer']['parameters'])


    def __load_metrics(self):
        precision = Precision(average=False)
        recall = Recall(average=False)
        F1 = precision * recall * 2 / (precision + recall + 1e-20)
        F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)

        self.metrics = {'accuracy': Accuracy(),
                        "f1": F1,
                        "precision": precision.mean(),
                        "recall": recall.mean(),
                        'loss': Loss(F.cross_entropy)}

    def __load_loaders(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        root_dir = self.config["data"]["root_dir"]
        train_dataset = MPR_Dataset(root_dir, partition="train", config=self.config["data"], transform=transform)
        self.train_loader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=self.config["dataloader"]["batch_size"])
        self.val_loader = DataLoader(MPR_Dataset(root_dir, partition="val", config=self.config["data"], transform=transform), shuffle=False, batch_size=64)

    def __create_pbar(self):
        self.desc = "ITERATION - loss: {:.2f}"
        self.pbar = tqdm(
            initial=0, leave=False, total=len(self.train_loader),
            desc=self.desc.format(0)
        )

    def __create_trainer(self):
        self.trainer = create_supervised_trainer(self.model, self.optimizer, F.cross_entropy, device=self.device)

        @self.trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(self.train_loader) + 1

            if iter % 10 == 0:
                self.pbar.desc = self.desc.format(engine.state.output)
                self.pbar.update(10)

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            self.pbar.refresh()
            self.evaluator.run(self.train_loader)
            metrics = self.evaluator.state.metrics
            results = " ".join(["Avg {}: {:.2f}".format(name, metrics[name]) for name in metrics])
            tqdm.write("Training Results - Epoch: {} {}".format(engine.state.epoch, results))

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            self.evaluator.run(self.val_loader)
            metrics = self.evaluator.state.metrics
            results = " ".join(["Avg {}: {:.2f}".format(name, metrics[name]) for name in metrics])
            tqdm.write("Validation Results - Epoch: {}  {}".format(engine.state.epoch, results))
            self.pbar.n = self.pbar.last_print_n = 0


    def __create_evaluator(self):
        self.evaluator = create_supervised_evaluator(self.model, metrics=self.metrics, device=self.device)

    def run(self):
        self.trainer.run(self.train_loader, max_epochs=20)

if __name__ == "__main__":
    pass