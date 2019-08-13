import logging
import warnings
import inspect
import importlib
import os

logging.getLogger('werkzeug').setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ignite.metrics import Accuracy, Loss, Precision, Recall, ConfusionMatrix, MetricsLambda
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from datasets.mpr_dataset import MPR_Dataset

from tqdm import tqdm
import yaml
from tensorboard import program
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, config):
        self.config = config

        os.makedirs(self.config['experiments_path'], exist_ok=True)
        self.id = len(os.listdir(self.config['experiments_path'])) + 1
        self.path = os.path.join(self.config['experiments_path'], "exp{}".format(self.id))
        os.makedirs(self.path, exist_ok=True)

        self.device = self.config['device']
        self.n_class = len(self.config['data']['groups'])
        self.__save_config()
        self.__load_tensorboad()
        self.__load_model()
        self.__load_optimizer()
        self.__load_augmentation()
        self.__load_sampler()
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

    def __load_tensorboad(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.path, "logs"), flush_secs=30)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', '{}/logs'.format(self.path)])
        tb.launch()

    def __save_config(self):
        config_path = os.path.join(self.path, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def __load_model(self):
        mapping = self.__module_mapping('models')
        self.config['model']['parameters']['n_classes'] = self.n_class
        self.model = mapping[self.config['model']['name']](**self.config['model']['parameters'])

    def __load_optimizer(self):
        mapping = self.__module_mapping('torch.optim')
        self.optimizer = mapping[self.config['optimizer']['name']](self.model.parameters(),
                                                                   **self.config['optimizer']['parameters'])

    def __load_augmentation(self):
        mapping = self.__module_mapping('augmentations')
        self.augmentation = mapping[self.config['data']['augmentation']['name']](
            **self.config['data']['augmentation']['parameters'])

    def __load_metrics(self):
        precision = Precision(average=False)
        recall = Recall(average=False)
        F1 = precision * recall * 2 / (precision + recall + 1e-20)
        F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)
        confusion_matrix = ConfusionMatrix(self.n_class)
        # TODO: Add metric by patient
        self.metrics = {'accuracy': Accuracy(),
                        "f1": F1,
                        "confusion_matrix": confusion_matrix,
                        "precision": precision.mean(),
                        "recall": recall.mean(),
                        'loss': Loss(F.cross_entropy)}

    def __load_sampler(self):
        mapping = self.__module_mapping('samplers')
        self.sampler = mapping[self.config['dataloader']['sampler']]

    def __load_loaders(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        root_dir = self.config["data"]["root_dir"]
        train_dataset = MPR_Dataset(root_dir, partition="train", config=self.config["data"], transform=transform,
                                    augmentation=self.augmentation)
        self.train_loader = DataLoader(train_dataset, sampler=self.sampler(train_dataset),
                                       batch_size=self.config["dataloader"]["batch_size"])
        self.val_loader = DataLoader(
            MPR_Dataset(root_dir, partition="val", config=self.config["data"], transform=transform), shuffle=False,
            batch_size=64)

        self.test_loader = DataLoader(
            MPR_Dataset(root_dir, partition="test", config=self.config["data"], transform=transform), shuffle=False,
            batch_size=64)

    def __create_pbar(self):
        self.desc = "ITERATION - loss: {:.2f}"
        self.pbar = tqdm(
            initial=0, leave=False, total=len(self.train_loader),
            desc=self.desc.format(0)
        )

    def __create_trainer(self):
        self.trainer = create_supervised_trainer(self.model, self.optimizer, F.cross_entropy, device=self.device)

        # TODO: Add checkpoints
        @self.trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(self.train_loader) + 1
            if iter % 10 == 0:
                self.writer.add_scalar("batch/loss/train", engine.state.output, engine.state.iteration)
                self.pbar.desc = self.desc.format(engine.state.output)
                self.pbar.update(10)

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            self.pbar.refresh()
            self.evaluator.run(self.train_loader)
            metrics = self.evaluator.state.metrics
            for metric in metrics:
                if metric != "confusion_matrix":
                    self.writer.add_scalars("epoch/{}".format(metric), {'train': metrics[metric]}, engine.state.epoch)
                else:
                    df = pd.DataFrame(metric, range(3), range(3))
                    sn.heatmap(df, annot=True)
                    fig = plt.figure()
                    fig.canvas.draw()
                    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    self.writer.add_images("epoch/train/", data)


            results = " ".join(["Avg {}: {:.2f}".format(name, metrics[name]) for name in metrics if name != "confusion_matrix"])
            tqdm.write("Training Results - Epoch: {} {}".format(engine.state.epoch, results))

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            self.evaluator.run(self.val_loader)
            metrics = self.evaluator.state.metrics
            for metric in metrics:
                if metric != "confusion_matrix":
                    self.writer.add_scalars("epoch/{}".format(metric), {'validation': metrics[metric]}, engine.state.epoch)
                else:
                    df = pd.DataFrame(metric, range(3), range(3))
                    sn.heatmap(df, annot=True)
                    fig = plt.figure()
                    fig.canvas.draw()
                    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    self.writer.add_images("epoch/validation/", data)
            results = " ".join(["Avg {}: {:.2f}".format(name, metrics[name]) for name in metrics if name != "confusion_matrix"])
            tqdm.write("Validation Results - Epoch: {}  {}".format(engine.state.epoch, results))

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_test_results(engine):
            self.evaluator.run(self.test_loader)
            metrics = self.evaluator.state.metrics
            for metric in metrics:
                if metric != "confusion_matrix":
                    self.writer.add_scalars("epoch/{}".format(metric), {'test': metrics[metric]}, engine.state.epoch)
                else:
                    df = pd.DataFrame(metric, range(3), range(3))
                    sn.heatmap(df, annot=True)
                    fig = plt.figure()
                    fig.canvas.draw()
                    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    self.writer.add_images("epoch/test/", data)

            results = " ".join(["Avg {}: {:.2f}".format(name, metrics[name]) for name in metrics if name != "confusion_matrix"])
            tqdm.write("Test Results - Epoch: {}  {}".format(engine.state.epoch, results))
            self.pbar.n = self.pbar.last_print_n = 0

    def __create_evaluator(self):
        self.evaluator = create_supervised_evaluator(self.model, metrics=self.metrics, device=self.device)

        best_loss = ModelCheckpoint(os.path.join(self.path, "models/"), filename_prefix="model", score_name="val_loss",
                                    score_function=lambda engine: engine.state.metrics['loss'], n_saved=5,
                                    atomic=True, create_dir=True)

        self.evaluator.add_event_handler(Events.COMPLETED, best_loss, {"model": model})

    def run(self):
        self.trainer.run(self.train_loader, max_epochs=20)


if __name__ == "__main__":
    metric = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    df = pd.DataFrame(metric, range(3), range(3))
    sn.heatmap(df, annot=True)
    fig = plt.figure()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    print(data.shape)
