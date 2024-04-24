def adamatch_hyperparams(lr=0.00001, tau=0.9, wd=1e-2, scheduler=True):
    """
    Return a dictionary of hyperparameters for the AdaMatch algorithm.
    adamatch_hyperparams(lr=1e-3, tau=0.8, wd=5e-4, lr=0.00001,wd=1e-2,scheduler=False):
    Arguments:
    ----------
    lr: float
        Learning rate.

    tau: float
        Weight of the unsupervised loss.

    wd: float
        Weight decay for the optimizer.

    scheduler: bool
        Will use a StepLR learning rate scheduler if set to True.

    Returns:
    --------
    hyperparams: dict
        Dictionary containing the hyperparameters. Can be passed to the `hyperparams` argument on AdaMatch.
    """
    
    hyperparams = {'learning_rate': lr,
                   'tau': tau,
                   'weight_decay': wd,
                   'step_scheduler': scheduler
                   }

    return hyperparams