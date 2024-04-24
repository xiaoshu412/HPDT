from dataset import get_dataloaders
from network import Encoder, Classifier
from hyperparameters import adamatch_hyperparams
from HPDT import HPDT


data = get_dataloaders()
source_dataloader_train_norm,source_dataloader_train_weak, source_dataloader_train_strong, source_dataloader_test = data[0]
target_dataloader_train_norm,target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test = data[1]
# instantiate the network
n_classes = 7

encoder = Encoder()
classifier = Classifier(n_classes=n_classes)

# instantiate AdaMatch algorithm and setup hyperparameters
adamatch = HPDT(encoder, classifier)
hparams = adamatch_hyperparams()
epochs = 200 # my implementations uses early stopping
save_path = "./checkpoint.pt"

# train the model
adamatch.train(source_dataloader_train_norm,source_dataloader_train_weak, source_dataloader_train_strong,
               target_dataloader_train_norm,target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test,
               epochs, hparams, save_path)

# evaluate the model
adamatch.plot_metrics()

# returns accuracy on the test set
print(f"accuracy on test set = {adamatch.evaluate(target_dataloader_test)}")

# returns a confusion matrix plot and a ROC curve plot (that also shows the AUROC)
adamatch.plot_cm_roc(target_dataloader_test)

