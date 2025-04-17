class PatienceScheduler:
    def __init__(self, optimizer, patience=3, factor=0.5, min_lr=1e-6, mode="min"):
        """
        Args:
            optimizer: The optimizer whose learning rate will be adjusted.
            patience: Number of epochs with no improvement after which learning rate will be reduced.
            factor: Factor by which the learning rate will be reduced (new_lr = lr * factor).
            min_lr: A lower bound on the learning rate.
            mode: One of {'min', 'max'}. If 'min', the scheduler will reduce the LR when the monitored
                  value has stopped decreasing. If 'max', it will reduce the LR when the monitored
                  value has stopped increasing.
        """
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.mode = mode
        self.best_value = None
        self.num_bad_epochs = 0
        self.last_epoch = -1

    def step(self, current_value):
        """
        Called after each epoch to update the learning rate based on the current validation performance.

        Args:
            current_value: The current epoch's performance (e.g., validation loss or accuracy).
        """
        if self.best_value is None:
            # Initialize the best value with the first value.
            self.best_value = current_value

        if self._is_better(current_value, self.best_value):
            # If the current performance is better, update best_value and reset bad epochs.
            self.best_value = current_value
            self.num_bad_epochs = 0
        else:
            # Otherwise, increment the number of bad epochs.
            self.num_bad_epochs += 1

        # If bad epochs exceed patience, reduce learning rate.
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0

    def _is_better(self, current_value, best_value):
        """
        Determines if the current performance is better than the best recorded performance.

        Args:
            current_value: Current epoch's performance (e.g., loss or accuracy).
            best_value: Best performance recorded so far.
        """
        if self.mode == "min":
            return current_value < best_value
        elif self.mode == "max":
            return current_value > best_value
        else:
            raise ValueError("mode should be either 'min' or 'max'.")

    def _reduce_lr(self):
        """
        Reduce the learning rate of the optimizer by the factor provided, ensuring it doesn't go below min_lr.
        """
        for param_group in self.optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group["lr"] = new_lr
        print(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")