import copy
import os,sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# export QT_QPA_PLATFORM=offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Check if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data_path = sys.argv[1]
data = pd.read_csv(data_path, header=0)
print(data.info())
# print(data.head())

# data = pd.concat([data.iloc[:, :15], data.iloc[:, 16:32], data.iloc[:, 15:16]], axis=1)
# index_to_drop = ['column00', 'ID_RUN', 'ID_LANE', 'PROJECT_NAME', 'ID_RG', 'CONFIRMED_BY', 'SEQUENCESCAPE_NAME', 'BAITSET_NAME',
#                  'GENOME_BUILD_BUILD', 'PROCESS_TYPE_DESCRIPTION', 'SPECIES_NAME', 'Library Design', 'BAITSET_GC_CONTENT', 'GENOME_BUILD_GC_CONTENT']
# data = data.drop(index_to_drop, axis=1)
# print(data.info())

print(data.info())
print(data.isnull().sum())

X = data.iloc[:, 0:17]
y = data.iloc[:, 17]


# Define two models
class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(17, 89)
        self.relu = nn.ReLU()
        self.output = nn.Linear(89, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(17, 37)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(37, 89)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(89, 37)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(37, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# Compare model sizes
model1 = Wide()
model2 = Deep()
print(sum([x.reshape(-1).shape[0] for x in model1.parameters()]))  # 11161
print(sum([x.reshape(-1).shape[0] for x in model2.parameters()]))  # 11041

def min_max_normalize(col):
    return (col - col.min()) / (col.max() - col.min())
X = X.apply(min_max_normalize)

print(X)
print(y)

# Binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
print(X)
print(y)

n_epochs = 30   # number of epochs to run
batch_size = 10  # size of each batch
# loss function and optimizer
loss_fn = nn.BCELoss()  # binary cross entropy

# Helper function to train one model
def model_train(model, X_train, y_train, X_val, y_val, loss_fn=nn.BCELoss(), n_epochs=300, batch_size=10, device=device):
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    model = model.to(device)

    # define optimizer
    optimizer=optim.Adam(model.parameters())
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc

# train-test split: Hold out the test set for final model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)

cv_scores_wide = []
for train, test in kfold.split(X_train, y_train):
    # create model, train, and get accuracy
    model = Wide()
    print(f"Fold----")
    acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test], loss_fn=loss_fn, n_epochs=n_epochs, batch_size=batch_size, device=device)

    print("Accuracy (wide): %.2f" % acc)
    cv_scores_wide.append(acc)
cv_scores_deep = []
for train, test in kfold.split(X_train, y_train):
    # create model, train, and get accuracy
    model = Deep()
    acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test], loss_fn=loss_fn, n_epochs=n_epochs, batch_size=batch_size, device=device)

    print("Accuracy (deep): %.2f" % acc)
    cv_scores_deep.append(acc)

# evaluate the model
wide_acc = np.mean(cv_scores_wide)
wide_std = np.std(cv_scores_wide)
deep_acc = np.mean(cv_scores_deep)
deep_std = np.std(cv_scores_deep)
print("Wide: %.2f%% (+/- %.2f%%)" % (wide_acc*100, wide_std*100))
print("Deep: %.2f%% (+/- %.2f%%)" % (deep_acc*100, deep_std*100))

# rebuild model with full set of training data
if wide_acc > deep_acc:
    print("Retrain a wide model")
    model = Wide()
else:
    print("Retrain a deep model")
    model = Deep()
acc = model_train(model, X_train, y_train, X_test, y_test)
print(f"Final model accuracy: {acc*100:.2f}%")

model.eval()
print(f"Final model parameters: {sum([x.reshape(-1).shape[0] for x in model.parameters()])}")
with torch.no_grad():
    # Test out inference with 5 samples
    for i in range(5):
        print(i)
        model = model.to(device)
        # print(f"model is on {model.device}")
        X_test = X_test.to(device)
        print(f"X_test is on {X_test.device}")
        y_pred = model(X_test[i:i+1])
        print(f"y_pred is on {y_pred.device}")
        print(f"{X_test[i].cpu().numpy()} -> {y_pred[0].cpu().numpy()} (expected {y_test[i].cpu().numpy()})")

    # Plot the ROC curve
    print("plotting ROC curve ----- ")
    y_pred = model(X_test).detach().cpu().numpy()
    # print(f"y_pred is on {y_pred.device} ===== ")
    y_test = y_test.detach().cpu().numpy()
    # print(f"y_test is on {y_test.device} ===== ")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig('roc-genomic.png')

del os.environ["QT_QPA_PLATFORM"]
