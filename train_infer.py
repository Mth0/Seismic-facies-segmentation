import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from functools import reduce
import pandas as pd
import os
import torch.nn.functional as F

def calculate_median_freq_weights(dataset, num_classes, epsilon=1e-8):
    """
    Calculates class weights using Median Frequency Balancing.
    Args:
        dataset (torch.utils.data.Dataset): A dataset object containing the images.
        num_classes (int): The total number of classes.
        epsilon (float): A small value to prevent division by zero.
    Returns:
        weights_tensor (torch.Tensor): A 1D tensor of weights (shape [num_classes]).
    """
    print(f"Calculating Median Frequency Balancing weights for {num_classes} classes...")

    class_counts = np.zeros(num_classes, dtype=np.float64)
    total_pixels = 0.0

    for _, mask in tqdm(dataset, desc="Counting pixels"):
        class_indices, counts = np.unique(mask.numpy(), return_counts=True)

        for idx, count in zip(class_indices, counts):
            if idx < num_classes:  # Ensure index is valid
                class_counts[idx] += count

        total_pixels += mask.numel()

    Calculates class frequencies
    class_frequencies = class_counts / (total_pixels + epsilon)

    # Calculates the median of all *non-zero* frequencies
    non_zero_freqs = class_frequencies[class_frequencies > 0]

    if len(non_zero_freqs) == 0:
        print("Warning: No classes found. Returning all weights as 1.0")
        return torch.ones(num_classes, dtype=torch.float32)

    median_frequency = np.median(non_zero_freqs)

    # Calculates the final weights
    weights = median_frequency / (class_frequencies + epsilon)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print("-" * 30)
    print(f"Class Frequencies: {class_frequencies}")
    print(f"Median Frequency: {median_frequency:.6f}")
    print(f"Calculated Weights: {weights_tensor}")
    print("-" * 30)

    return weights_tensor

def calculate_iou(preds_logits, targets, num_classes, epsilon=1e-6):
    """
    Calculates the Jaccard Index (Mean IoU) for semantic segmentation.
    Args:
        preds_logits (torch.tensor): The raw logits from the model.
        targets (torch.tensor): The ground truth labels (class indices).
        num_classes (int): The number of classes.
        epsilon (float): A small value to prevent division by zero.
    Returns:
        iou (torch.Tensor): The mean IoU score over all classes.
    """
    preds_classes = torch.argmax(preds_logits, dim=1) #[N, H, W]

    preds_flat = preds_classes.view(-1)
    targets_flat = targets.view(-1)

    iou_per_class = []

    for c in range(num_classes):
        preds_c = (preds_flat == c)
        targets_c = (targets_flat == c)

        # Calculates intersection and union
        intersection = (preds_c & targets_c).float().sum()
        union = (preds_c | targets_c).float().sum()

        # Calculates the IoU
        iou = (intersection + epsilon) / (union + epsilon)
        iou_per_class.append(iou)

    iou = torch.stack(iou_per_class)

    return iou

class FocalLoss(nn.Module):
    """
    A class for the focal loss implementation.
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean', ignore_index=-100):
        """
        Initialize the loss object.
        Args:
            alpha (torch.tensor): The weights for each class used in the weighted
            cross-entropy (WCE) term.
            gamma (float): The gamma argument for the exponentiation.
            reduction (str): reduction argument used in the WCE computation.
            ignore_index (int): ignore_index argument used in the WCE computation.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            ignore_index=self.ignore_index,
            weight=self.alpha
        )

        pt = torch.exp(-ce_loss)

        # Calculates focal loss
        focal_loss = (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TverskyLoss(nn.Module):
    """
    A class for the Tversky loss implementation.
    """
    def __init__(self, alpha=0.5, beta=0.5, epsilon=1e-6):
        """
        Initialize the loss object. In the case where
		alpha = beta = 0.5
	The Tversky loss is exactly the dice loss.
        Args:
            alpha (float): The False Positives (FP) weight.
            beta (float): The False Negatives (FN) weight.
            epsilon (float): A small value to prevent division by zero.
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, preds_logits, targets):
        num_classes = preds_logits.size(1)

        preds_probs = F.softmax(preds_logits, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)

        # True Positives (Intersection)
        tp = (preds_probs * targets_one_hot).sum(dim=dims)
        # False Positives
        fp = (preds_probs * (1 - targets_one_hot)).sum(dim=dims)
        # False Negatives
        fn = ((1 - preds_probs) * targets_one_hot).sum(dim=dims)

        tversky_index = (tp + self.epsilon) / (tp + self.alpha * fp + self.beta * fn + self.epsilon)

        # Calculates Tversky Loss (1 - Index)
        tversky_loss_per_class = 1. - tversky_index
        return tversky_loss_per_class.mean()

class HybridLoss(nn.Module):
    """
    A class for the (focal loss + Tversky loss) implementation.
    """
    def __init__(self,
                focal_loss,
                tv_loss,
                tv_weight=15,
                epoch_interval=(12, 50),
                ):
        """
        Initialize the loss object.
        Args:
            focal_loss (FocalLoss): The focal loss part from the hybrid one.
            tv_loss (TverskyLoss): The Tversky loss part from the hybrid one.
            tv_weight (float): The final tv_weight.
            ignore_index (int): ignore_index argument used in the WCE computation.

        In the beginning, the loss is
		focal loss + Tversky loss
	until the increasing stage starts. During this stage, the tv_loss increases
        and the loss function becomes
	        (focal loss) * 1/lambda + Tversky loss
        Where lambda is float number between 1 and tv_loss.
        """
        super(HybridLoss, self).__init__()
        self.focal_loss = focal_loss
        self.tv_loss = tv_loss
        self.tv_weight = tv_weight
        self.epoch_interval = epoch_interval

        self.num_increases = self.epoch_interval[1] - self.epoch_interval[0] + 1
        self.increase = (self.tv_weight - 1) / self.num_increases
        self.epoch = 0
        self.current_multiplier = 1

    def update_epoch(self):
        self.epoch += 1
        if self.epoch in range(self.epoch_interval[0], self.epoch_interval[1] + 1):
            self.current_multiplier = self.current_multiplier + self.increase
            print(f"weight increase for {self.current_multiplier}")

    def forward(self, inputs, targets):
        f = self.focal_loss(inputs, targets)
        d = self.tv_loss(inputs, targets)
        return f / self.current_multiplier + d


def train_and_validate_epoch(model, dataloader, criterion, optimizer=None, is_training=True, device="cuda"):
    """
    Performs one epoch of training or validation.
    Args:
        model (nn.Module): The model to be trained or validated.
        dataloader (torch.utils.data.DataLoader): The data that will be used
        in the computation step.
        criterion (nn.Module): The loss function to be used.
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        is_training (bool): A flag indicating if the model is training or not
        device (str): The device where the computations are going to be made.
    Returns:
        epoch_loss (float): The epoch loss value.
        epoch_acc (float): The epoch accuracy value.
        (float): The epoch average IoU per class.
    """
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_pixels = 0
    total_images = 0
    train_iou = None

    progress_bar = tqdm(dataloader, desc='Train' if is_training else 'Validation', leave=False)

    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.squeeze(1).long().to(device)

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)["out"]
            loss = criterion(outputs, labels)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_images += inputs.size(0)

        _, preds = torch.max(outputs, 1)

        # Calculates accuracy
        correct_predictions += torch.sum(preds == labels.data)
        total_pixels += inputs.size(0) * inputs.size(2) * inputs.size(3)

        progress_bar.set_postfix(loss=loss.item())

        if train_iou is None:
            train_iou = calculate_iou(outputs, labels, num_classes=6, epsilon=1e-6)
        else:
            train_iou += calculate_iou(outputs, labels, num_classes=6, epsilon=1e-6)

    # Calculates final epoch metrics
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct_predictions.double() / total_pixels

    return epoch_loss, epoch_acc.item(), train_iou / len(dataloader)


def plot_stats(history):
    """
    Plots the results of a training based on its history.
    Args:
        history (dict): A history containing informations about loss, accuracy, IoU and
        execution time.
    """

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(history['train_loss'])),
                history['train_loss'], label='Train Loss')
    plt.plot(history['train_loss'], label='Train Loss')

    val_loss = np.array(history['val_loss'])
    non_null_idx = np.where(val_loss != None)[0]
    plt.scatter(non_null_idx,
                val_loss[non_null_idx], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(history['train_acc'])),
                history['train_acc'], label='Train Accuracy')
    plt.plot(history['train_acc'], label='Train Accuracy')

    val_acc = np.array(history['val_acc'])
    non_null_idx = np.where(val_acc != None)[0]

    plt.scatter(non_null_idx,
                val_acc[non_null_idx], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_model(NUM_EPOCHS, model, train_dataloader,
                criterion, optimizer, val_dataloader=None,
                device="cuda", model_name="seismic",
               val_peridiocity=3, update_epoch=False,
               UNFREEZE_EPOCH=35,
               FINE_TUNE_LR=0.1):
    """
    Performs the training of a model.
    Args:
        NUM_EPOCHS (int): Number of epochs.
        model (nn.Module): The model to be trained.
        train_dataloader (torch.utils.data.DataLoader): The train data
            that will be used in training.
        criterion (nn.Module): The loss function to be used.
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        val_dataloader (torch.utils.data.DataLoader): The validation data
            that will be used in training. If 'None', no validation is made.
        device (str): The device where the computations are going to be made.
        model_name (str): Name used as the file name during saving model step.
        val_periodicity (int): Interval for validation computations.
        update_epoch (bool): A flag indicating if the weights of tv_loss will increase.
        UNFREEZE_EPOCH (int): The epoch where the pre-trained part of the model
        will start to be trained.
        FINE_TUNE_LR (float): How much times the learning rate will decrease
		after the unfreeze.
    Returns:
        history (dict): A history containing informations about loss, accuracy, IoU and
        execution time.
        model (nn.Module): The model after training.
    """
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': [],
               'train_times': [], 'val_times': [],
               'train_iou':[], 'val_iou':[]}

    aux_val_time = []

    print(f"Starting training for {NUM_EPOCHS} epochs...")

    # --- Main Training Loop ---
    for epoch in range(NUM_EPOCHS):
        if epoch == UNFREEZE_EPOCH:
            print(f"--- Epoch {epoch}: Unfreezing backbone layers! ---")

            for param in model.backbone.parameters():
                param.requires_grad = True

            print(f"--- Dropping Learning Rate to an order of {FINE_TUNE_LR} for fine-tuning ---")
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * FINE_TUNE_LR

        start_time = time.perf_counter()

        # Training phase
        train_loss, train_acc, train_iou = train_and_validate_epoch(model.to(device), train_dataloader, criterion,
                             optimizer=optimizer, is_training=True, device="cuda")
        torch.cuda.empty_cache()
        end_time = time.perf_counter()
        history['train_times'].append(end_time - start_time)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_iou'].append(train_iou)

        train_iou_str = reduce(lambda x, y: f"{x}, " + f"\n{y:.2f}", train_iou)

        # Validation phase (if needed)
        if epoch % val_peridiocity == 0 and val_dataloader is not None:
            start_time = time.perf_counter()

            with torch.no_grad():
                val_loss, val_acc, val_iou = train_and_validate_epoch(model.to(device), val_dataloader, criterion,
                                 optimizer=None, is_training=False, device="cuda")

	    history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_iou'].append(val_iou)

            end_time = time.perf_counter()
            history['val_times'].append(end_time - start_time)
            aux_val_time.append(history['val_times'][-1])

            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}  | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            val_iou_str = reduce(lambda x, y: f"{x}, " + f"\n{y:.2f}", val_iou)
            print("Train IoU: "+ train_iou_str + "\nVal IoU: "+ val_iou_str)

            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, f"./models/best_seismic_model_{model_name}.pth")
                print(f'-> Model saved! New best validation accuracy in epoch {epoch+1}: {best_acc:.4f}')

        else:
            if epoch % val_peridiocity == 0 and train_acc > best_acc:
                best_acc = train_acc
                checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, f"./models/best_seismic_model_{model_name}.pth")
                print(f'-> Model saved! New best train accuracy in epoch {epoch+1}: {best_acc:.4f}')

            history['val_loss'].append(None)
            history['val_acc'].append(None)
            history['val_iou'].append(None)
            history['val_times'].append(0)
            print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
            print("Train IoU: "+ train_iou_str)

        # Print epoch summary
        current_time, avg_time = history['train_times'][-1], np.mean(history['train_times'])
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS} | Train time: {current_time:.2f}s | time avg: {avg_time:.2f}s')
        if epoch % val_peridiocity == 0:
            current_time, avg_time = history['val_times'][-1], np.mean(aux_val_time)
            print(f'\nValidation time: {current_time:.2f}s | time avg: {avg_time:.2f}s')

        if update_epoch:
            criterion.update_epoch()
            print(f"Epoch: {criterion.epoch} | Multiplier: {criterion.current_multiplier}")


    if epoch % val_peridiocity != 0:
        if val_dataloader is not None:
            with torch.no_grad():
                    val_loss, val_acc, val_iou = train_and_validate_epoch(model.to(device), val_dataloader, criterion,
                                         optimizer=None, is_training=False, device="cuda")
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint = {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, f"./models/best_seismic_model_{model_name}.pth")
                print(f'-> Model saved! New best validation accuracy in epoch {epoch+1}: {best_acc:.4f}')
        else:
            if train_acc > best_acc:
                best_acc = train_acc
                checkpoint = {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, f"./models/best_seismic_model_{model_name}.pth")
                print(f'-> Model saved! New best train accuracy in epoch {epoch+1}: {best_acc:.4f}')

    print("\n--- Training Complete ---")
    if val_dataloader is not None:
        print(f"Best Validation Accuracy: {best_acc:.4f}")
    else:
        print(f"Best Train Accuracy: {best_acc:.4f}")
    print(f"Total time: {(sum(history['train_times']) + sum(history['val_times'])):.4f}")

    plot_stats(history)

    return history, model

def to_csv(history, model_name):
    """
    Given a history per epoch, saves it as a .csv file.
    Args:
        history (dict): A history containing informations about loss, accuracy, IoU and
            execution time.
        model_name (str): Name to be used as the .csv file name.
    """
    model_csv = pd.DataFrame(history)
    model_csv.to_csv(os.path.join("./models", model_name + ".csv"), index=False)

def infer(model, infer_dataloader, device="cuda"):
    """
    Do inference in dataset.
    Args:
        model (nn.Module): The model used in the inference.
        infer_dataloader (torch.utils.data.DataLoader): the data that the model will predict.
        device (str): The device where the computations are going to be made.
    Returns:
        preds (torch.tensor): The predictions made by the model in the inference data.
    """

    model.eval()
    preds = []
    progress_bar = tqdm(infer_dataloader,
                        desc='Inference',
                        leave=False)
    start_time = time.perf_counter()
    with torch.no_grad():
        for x_batch in progress_bar:
            preds.append(model(x_batch.to(device))["out"])
    preds = torch.cat(preds, dim=0)
    print(f"Time for inference: {time.perf_counter() - start_time}s")
    return preds

def submit(mobile_net, test_loader, device="cuda", file_name="predictions"):
    """
    Save the inference result in a .npz file.
    Args:
        model (nn.Module): The model used in inference.
        test_loader (torch.utils.data.DataLoader): the data that the model will predict.
        device (str): The device where the computations are going to be made.
        file_name (str): File name used in the saving step.
    """
    preds = infer(mobile_net, test_loader, device)
    submission = preds.argmax(dim=1).detach().cpu()
    submission = submission.numpy().astype(np.int8)
    print(submission.shape)
    np.savez_compressed(file_name + ".npz", predictions = submission)
