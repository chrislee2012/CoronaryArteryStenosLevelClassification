import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision.utils as vutils

import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from tensorboardX import SummaryWriter
from tqdm import tqdm 
import cv2
from sklearn.metrics import f1_score, precision_score, recall_score

def eval_fun(labels, preds):
    rr = (np.intersect1d(labels, preds))
    precision = np.float(len(rr)) / len(preds)
    recall = np.float(len(rr)) / len(labels)
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)


def train_model(model, data_loaders, criterion, optimizer, scheduler, path_to_save_weights,exp_name, device, num_epochs=25):
    since = time.time()

    writer = SummaryWriter('runs/{}'.format(exp_name))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in    range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
                print('Training: ')
                phase_acc_name = 'trainAcc'
                phase_loss_name = 'trainLoss'
            else:
                print('Validation: ')
                model.eval()   # Set model to evaluate mode
                phase_acc_name = 'valAcc'
                phase_loss_name = 'valLoss'

            running_loss = 0.0
            running_corrects = 0.0
            running_f1_score = 0.0
            running_precision = 0.0
            running_recall = 0.0
            # Iterate over data.
            # ind = 0
            for inputs, labels, _, _, _ in tqdm(data_loaders[phase]):
                # ind +=1 

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # Tensorboad images
                # image_to_show = inputs[0].cpu().numpy().T
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(image_to_show,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)



                # print(image_to_show.shape)
                # final_img = torch.tensor(image_to_show)
                # final_img = vutils.make_grid(final_img)
                # writer.add_image('{}: {}'.format(phase, str(epoch)), final_img,epoch)

                # write.add_image(logit)
                # statistics
                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)
                running_f1_score += f1_score(labels.data.cpu().numpy(), preds.cpu().numpy())
                running_precision += precision_score(labels.data.cpu().numpy(), preds.cpu().numpy())
                running_recall += recall_score(labels.data.cpu().numpy(), preds.cpu().numpy())

                # if ind > 3:
                #     break  

            epoch_loss = running_loss / len(data_loaders[phase])

            epoch_acc = running_corrects.double() / (len(data_loaders[phase]) * 16)
            epoch_f1 = running_f1_score / len(data_loaders[phase])
            epoch_precision = running_precision / len(data_loaders[phase])
            epoch_recall = running_recall / len(data_loaders[phase])
            
            # Tensorboard scalsrs
            writer.add_scalar('{}_acc'.format(phase), epoch_acc, epoch)
            writer.add_scalar('{}_f1'.format(phase), epoch_f1, epoch)
            writer.add_scalar('{}_recall'.format(phase), epoch_recall, epoch)
            writer.add_scalar('{}_precision'.format(phase), epoch_precision, epoch)

            writer.add_scalar('{}_loss'.format(phase), epoch_loss, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f} F1: {} Precision: {} Recall: {}'.format(phase, epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, path_to_save_weights)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

