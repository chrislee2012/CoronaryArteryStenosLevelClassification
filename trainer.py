import torch
from torchvision import models, transforms
import torch.nn.functional as F
from torch.optim import Adam
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from datasets.mpr_dataset import MPR_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__load_model()
        self.__load_optimizer()
        self.__load_metrics()
        self.__load_loaders()
        self.__create_pbar()
        self.__create_evaluator()
        self.__create_trainer()

    def __load_model(self):
        self.model = models.resnet18(pretrained=True)

    def __load_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def __load_metrics(self):
        precision = Precision(average=False)
        recall = Recall(average=False)
        F1 = (precision * recall * 2 / (precision + recall)).mean()

        self.metrics = {'accuracy': Accuracy(),
                        "f1": F1,
                        "precision": precision.mean(),
                        "recall": recall.mean(),
                        'nll': Loss(F.cross_entropy)}

    def __load_loaders(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_loader = DataLoader(MPR_Dataset("data/", partition="train", config=self.config["data"]["filters"], transform=transform))
        self.val_loader = DataLoader(MPR_Dataset("data/", partition="train", config=self.config["data"]["filters"], transform=transform))

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
            # TODO: Improve code style
            avg_accuracy = metrics['accuracy']
            avg_nll = metrics['nll']
            avg_f1 = metrics['f1']
            avg_precision = metrics['precision']
            avg_recall = metrics['recall']
            tqdm.write(
                "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg precision: {:.2f} Avg recall: {:.2f} Avg f1: {:.2f} Avg loss: {:.2f} "
                    .format(engine.state.epoch, avg_accuracy, avg_precision, avg_recall, avg_f1, avg_nll)
            )

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            self.evaluator.run(self.val_loader)
            metrics = self.evaluator.state.metrics
            # TODO: Improve code style

            metrics_strs = []
            for name in metrics:
                metrics_strs.append("Avg {}: {:.2f}".format(name, metrics[name]))
            
            avg_accuracy = metrics['accuracy']
            avg_nll = metrics['nll']
            avg_f1 = metrics['f1']
            avg_precision = metrics['precision']
            avg_recall = metrics['recall']

            tqdm.write(
                "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg precision: {:.2f} Avg recall: {:.2f} Avg f1: {:.2f} Avg loss: {:.2f}"
                    .format(engine.state.epoch, avg_accuracy, avg_precision, avg_recall, avg_f1, avg_nll))

            self.pbar.n = self.pbar.last_print_n = 0

    def __create_evaluator(self):
        self.evaluator = create_supervised_evaluator(self.model, metrics=self.metrics, device=self.device)

    def run(self):
        self.trainer.run(self.train_loader, max_epochs=20)