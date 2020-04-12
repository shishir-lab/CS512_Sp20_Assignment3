
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
from Classifier import LSTMClassifier
from matplotlib import pyplot as plt

# Constants
output_size = 9   # number of class
input_size = 12

# Hyperparameters, feel free to tune

batch_size = 32
hidden_size = 70  # LSTM output size of each time step
basic_epoch = 100
Adv_epoch = 50
Prox_epoch = 100
saved_model_name = 'basic_lstm_model'


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
        prediction = model(input, r,batch_size = input.size()[0], mode = mode)
        loss = loss_fn(prediction, target)
        if mode == 'AdvLSTM':
            ''' Add adversarial training term to loss'''
            r = compute_perturbation(loss, model)
            adv_prediction = model(input, r, batch_size = input.size()[0], mode = mode)
            adv_loss = loss_fn(adv_prediction, target)
            loss = loss + adv_loss

        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/(input.size()[0])
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


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
            prediction = model(input, r, batch_size=input.size()[0], mode = mode)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects.double()/(input.size()[0])
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / len(test_iter), total_epoch_acc / len(test_iter)




def compute_perturbation(loss, model):
    # # Use autograd
    grads = grad(loss, model.lstmInput, create_graph=True)
    g = grads[0].permute(0,2,1).detach()    
    r = g/F.normalize(g, dim=2, p=2) # batch_size x seq x embed_dim
    return r.permute(0,2,1) #batch_size x embed_dim x seq to match lstmInput

    

best_basic=None

print("Training Model")
''' Training basic model '''
print("Loading Data")
train_iter, test_iter = load_data.load_data('JV_data.mat', batch_size)
print("Training Plain version")
model = LSTMClassifier(batch_size, output_size, hidden_size, input_size, epsilon=0.5)
loss_fn = F.cross_entropy

for epoch in range(basic_epoch):
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-3)
        train_loss, train_acc = train_model(model, train_iter, mode = 'plain')
        val_loss, val_acc = eval_model(model, test_iter, mode ='plain')
        if best_basic==None:
            best_basic=val_acc
            torch.save(model.state_dict(), saved_model_name)
        elif val_acc>best_basic:
            best_basic=val_acc
            torch.save(model.state_dict(), saved_model_name)
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%, Best: {best_basic:.2f}%')

''' Save and Load model'''

# 1. Save the trained model from the basic LSTM
#torch.save(model.state_dict(), 'basic_lstm_model')


# 2. load the saved model to Prox_model, which is an instance of LSTMClassifier
Prox_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
Prox_model.load_state_dict(torch.load(saved_model_name), strict=False)


# 3. load the saved model to Adv_model, which is an instance of LSTMClassifier
Adv_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
Adv_model.load_state_dict(torch.load(saved_model_name), strict=False)



# ''' Training Prox_model'''
# for epoch in range(Adv_epoch):
#     optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Prox_model.parameters()), lr=1e-3, weight_decay=1e-3)
#     train_loss, train_acc = train_model(Prox_model, train_iter, mode = 'ProxLSTM')
#     val_loss, val_acc = eval_model(Prox_model, test_iter, mode ='ProxLSTM')
#     print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')



# ''' Training Adv_model'''
# '''In adv_training.py ''' #