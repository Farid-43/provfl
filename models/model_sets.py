"""
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
import torch.nn as nn
import torch.nn.functional as F
from my_utils.utils import weights_init
import torch
from torch.cuda.amp import autocast as autocast

class Weight_layer(nn.Module):
    def __init__(self, feature_size, class_num):
        super(Weight_layer, self).__init__()
        self.fc1 = nn.Linear(feature_size, 32)
        self.fc2 = nn.Linear(32, class_num)
        self.apply(weights_init)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

class DiscreteClassifier(nn.Module):
    def __init__(self, num_feature, num_cls=2, dropout=0.1):
        super(DiscreteClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(num_feature,256),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(256,128),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(128,64),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(64, num_cls)
        
    def forward(self,x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)
    
class MLPClassifier(nn.Module):
    def __init__(self, num_feature, class_num=1):
        super(MLPClassifier, self).__init__()
        self.class_num = class_num
        self.fc1 = nn.Linear(num_feature, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, class_num)
        self.map = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        if self.class_num == 1:
            x = self.map(x)
        return x

class ClassificationModelGuest(nn.Module):

    def __init__(self, local_model):#), hidden_dim, num_classes):
        super().__init__()
        self.local_model = local_model

    def forward(self, input_X):
        z = self.local_model(input_X).flatten(start_dim=1)
        return z

class MLPBottomModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPBottomModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 32, bias=True),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(32, output_dim, bias=True),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class TrainableTopModel(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier_head1 = nn.Linear(hidden_dim, 16)
        self.classifier_head2 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z0, z1):
        out = torch.cat([z0, z1], dim=1)
        out = self.relu(self.classifier_head1(out))
        out = self.sigmoid(self.classifier_head2(out))
        return out

class TrainableTopModel_Multi_Party(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier_head1 = nn.Linear(hidden_dim, 16)
        self.classifier_head2 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = torch.cat(z, dim=1)
        out = self.relu(self.classifier_head1(out))
        out = self.sigmoid(self.classifier_head2(out))
        return out

def update_top_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def update_bottom_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return