import numpy as np
import torch


class EarlyStopping:

    def __init__(self, start, patience, paths, verbose=False):

        """EarlyStopping: early stopping strategy

        :param patience: patience epoch
        :param paths: path to save checkpoint
        :param verbose: verbosity
        """
        self.start = start
        self.patience = patience
        self.paths = paths
        self.epoch = 0
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.verbose = verbose

    def __call__(self, loss, models):

        """
        :param loss: loss to save as checkpoint
        :param models: models to save as checkpoint
        """
        score = loss
        self.epoch += 1
        if self.verbose:
            print('Epoch: ', self.epoch)

        if not self.epoch < self.start:
            if self.best_score is None:  # First epoch
                self.best_score = score  # Set best_score as 1 epoch score
                self.checkpoint(loss, models)  # Save models
            elif score > self.best_score:  # If the best score was not updated
                self.counter += 1
                if self.verbose:
                    print(f'Early Stopping Counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:  # Counter was more than patience
                    self.early_stop = True  # Stop flag early_stop was set as True
            else:  # If the best score was updated
                self.best_score = score  # Update best_score
                self.checkpoint(loss, models)  # Save models
                self.counter = 0  # Reset counter

    def checkpoint(self, loss, models):

        """
        :param loss: loss to save as checkpoint
        :param models: models to save as checkpoint
        :return:
        """

        if self.verbose:
            print(f'Validation loss decreased ({self.loss_min:.5f} --> {loss:.5f}). Saving model...')

        for path, model in zip(self.paths, models):
            model_path = f'{path}.pth'
            torch.save(model.state_dict(), model_path)

        self.loss_min = loss
