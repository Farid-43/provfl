import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from my_utils import utils
from models import model_sets


# baseline: attribute inference attacks
class train_NN():

    def __init__(self, trainloader, testloader, model):

        self.device = 'cuda'
        self.net = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()

        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=0.0001)
        self.net = self.net.to(self.device)
        
    # Training
    def train(self):

        self.net.train()
        for _, (inputs, targets) in enumerate(self.trainloader):
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to(self.device), targets.float().to(self.device)
            logits = self.net(inputs)
            logits = torch.squeeze(logits, 1)
            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                
    def test(self):
        self.net.eval()
        count = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                if isinstance(targets, list):
                    targets = targets[0]

                inputs, targets = inputs.to(self.device), targets.float().to(self.device)
                logits = self.net(inputs)
                logits = torch.squeeze(logits, 1)
                loss = self.criterion(logits, targets)

                pred = logits
                if count == 0:
                    y_true = targets
                    y_pred = pred
                else:
                    y_true = torch.cat((y_true, targets), axis=0)
                    y_pred = torch.cat((y_pred, pred), axis=0)
                count += 1

        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        accuracy_test = ((y_pred.round() == y_true).sum() / len(y_true))

        return accuracy_test


def baseline_NN(prop_feature, nonprop_feature, features, file='test'):
    pred_ratio = 0.0
    epoch = 50
    batch_size=32

    x_data = np.concatenate((prop_feature, nonprop_feature))
    y_data = np.concatenate((np.ones(len(prop_feature)), np.zeros(len(nonprop_feature))))
    print(x_data.shape, y_data.shape, features.shape)
    f_size = x_data.shape[1]
    data = utils.tensor_data_create(x_data, y_data)

    tra_len = int(len(data)*0.9)
    val_len = len(data) - tra_len
    train_dataset, val_dataset = torch.utils.data.random_split(data, [tra_len, val_len])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # train attack model 
    model = model_sets.MLPClassifier(num_feature=f_size)
    b2 = train_NN(train_loader, test_loader, model)
    
    for i in range(epoch):
        b2.train()
        if i % 10 == 0: 
            test_acc = b2.test()
    
    # infer prop ratio
    x = features
    y = np.ones(len(x))  # random

    test_feature = utils.tensor_data_create(x, y)
    testloader = torch.utils.data.DataLoader(test_feature, batch_size=64, shuffle=False, num_workers=2)
    
    count = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            logits = model(inputs)
            pred = torch.squeeze(logits, 1)

            if count == 0:
                y_true = targets
                y_pred = pred
            else:
                y_true = torch.cat((y_true, targets), axis=0)
                y_pred = torch.cat((y_pred, pred), axis=0)
            count += 1

    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    pred_ratio = ((y_pred.round() == y_true).sum() / len(y_true)) # predicted as 1

    return pred_ratio, test_acc

def baseline_stat(args, X, y, clf_name, features):
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    if clf_name == 'svm':
        clf = SVC()
    else:
        clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    pred_y = clf.predict(features)
    pred_ratio = sum(pred_y) * 1.0 / len(features)

    return acc, pred_ratio, time.time()-start_time