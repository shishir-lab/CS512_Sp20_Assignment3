import load_data
import torch
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
from Q6_Classifier import LSTMClassifier
#from matplotlib import pyplot as plt

# Constants
output_size = 9  # number of class
input_size = 12
batch_size = 32
hidden_size = 70  # LSTM output size of each time step
basic_epoch = 100
Adv_epoch = 50
Prox_epoch = 100
saved_model_name = "prox_lstm_model"


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


# Training model
def train_model(model, train_iter, mode):
    total_epoch_loss = 0
    total_epoch_acc = 0
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        input = batch[0]
        target = batch[1]
        target = torch.autograd.Variable(target).long()
        r = 0
        optim.zero_grad()
        prediction = model(input, r, batch_size=input.size()[0], mode=mode)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects / (input.size()[0])
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(train_iter), total_epoch_acc / len(train_iter)


# Test model
def eval_model(model, test_iter, mode):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    r = 0
    with torch.no_grad():
        for idx, batch in enumerate(test_iter):
            input = batch[0]
            target = batch[1]
            target = torch.autograd.Variable(target).long()
            prediction = model(input, r, batch_size=input.size()[0], mode=mode)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects.double() / (input.size()[0])
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / len(test_iter), total_epoch_acc / len(test_iter)



print("Loading Data")
train_iter, test_iter = load_data.load_data('JV_data.mat', batch_size)
loss_fn = F.cross_entropy


# 2. load the saved model to Prox_model, which is an instance of LSTMClassifier
Prox_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
# Prox_model.load_state_dict(torch.load(saved_model_name), strict=False)


print(''' Training Prox_model with dropout after convolution and batchnorm after proxlstm ''')
best_prox = None
for epoch in range(Prox_epoch):
    print('{}th epoch training'.format(epoch))
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Prox_model.parameters()), lr=1e-3, weight_decay=1e-3)
    train_loss, train_acc = train_model(Prox_model, train_iter, mode='ProxLSTM')
    val_loss, val_acc = eval_model(Prox_model, test_iter, mode='ProxLSTM')
    if best_prox == None:
        best_prox = val_acc
        torch.save(Prox_model.state_dict(), saved_model_name)
    elif val_acc > best_prox:
        best_prox = val_acc
        torch.save(Prox_model.state_dict(), saved_model_name)

    print('Epoch {}, Train loss: {}, Train Acc : {}, Test Loss: {}, Test Acc: {}, Best: {}'.format(epoch, train_loss, train_acc, val_loss, val_acc, best_prox))

