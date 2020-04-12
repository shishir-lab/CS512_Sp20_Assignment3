import torch
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
from Classifier import LSTMClassifier
from matplotlib import pyplot as plt
import load_data

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


print("Loading Data")
train_iter, test_iter = load_data.load_data('JV_data.mat', batch_size)
loss_fn = F.cross_entropy
Adv_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size, epsilon=0.8)

Adv_model.load_state_dict(torch.load(saved_model_name), strict=False)

# ''' Training Adv_model'''
best_adv=None
val_accs_1 = []

for epoch in range(Adv_epoch):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Adv_model.parameters()), lr=5e-4, weight_decay=1e-4)
    train_loss, train_acc = train_model(Adv_model, train_iter, mode = 'AdvLSTM')
    val_loss, val_acc = eval_model(Adv_model, test_iter, mode ='AdvLSTM')
    val_accs_1.append(val_acc)
    if best_adv==None:
        best_adv=val_acc
        torch.save(Adv_model.state_dict(), 'best_adv_model')
    elif val_acc>best_adv:
        best_adv=val_acc
        torch.save(Adv_model.state_dict(), 'best_adv_model')
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%, Best: {best_adv:.2f}%')

best_adv=None
val_accs_2 = []
Adv_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
Adv_model.load_state_dict(torch.load(saved_model_name), strict=False)
Adv_model.eplison=0.01
for epoch in range(Adv_epoch):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Adv_model.parameters()), lr=5e-4, weight_decay=1e-4)
    train_loss, train_acc = train_model(Adv_model, train_iter, mode = 'AdvLSTM')
    val_loss, val_acc = eval_model(Adv_model, test_iter, mode ='AdvLSTM')
    val_accs_2.append(val_acc)
    if best_adv==None:
        best_adv=val_acc
        torch.save(Adv_model.state_dict(), 'best_adv_model')
    elif val_acc>best_adv:
        best_adv=val_acc
        torch.save(Adv_model.state_dict(), 'best_adv_model')
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%, Best: {best_adv:.2f}%')

best_adv=None
val_accs_3 = []
Adv_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
Adv_model.load_state_dict(torch.load(saved_model_name), strict=False)
Adv_model.eplison=0.1
for epoch in range(Adv_epoch):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Adv_model.parameters()), lr=5e-4, weight_decay=1e-4)
    train_loss, train_acc = train_model(Adv_model, train_iter, mode = 'AdvLSTM')
    val_loss, val_acc = eval_model(Adv_model, test_iter, mode ='AdvLSTM')
    val_accs_3.append(val_acc)
    if best_adv==None:
        best_adv=val_acc
        torch.save(Adv_model.state_dict(), 'best_adv_model')
    elif val_acc>best_adv:
        best_adv=val_acc
        torch.save(Adv_model.state_dict(), 'best_adv_model')
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%, Best: {best_adv:.2f}%')

best_adv=None
val_accs_4 = []
Adv_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
Adv_model.load_state_dict(torch.load(saved_model_name), strict=False)
Adv_model.eplison=1
for epoch in range(Adv_epoch):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Adv_model.parameters()), lr=5e-4, weight_decay=1e-4)
    train_loss, train_acc = train_model(Adv_model, train_iter, mode = 'AdvLSTM')
    val_loss, val_acc = eval_model(Adv_model, test_iter, mode ='AdvLSTM')
    val_accs_4.append(val_acc)
    if best_adv==None:
        best_adv=val_acc
        torch.save(Adv_model.state_dict(), 'best_adv_model')
    elif val_acc>best_adv:
        best_adv=val_acc
        torch.save(Adv_model.state_dict(), 'best_adv_model')
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%, Best: {best_adv:.2f}%')


x=range(50)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(x,val_accs_1,label='0.8')
plt.plot(x,val_accs_2,label='0.01')
plt.plot(x,val_accs_3,label='0.1')
plt.plot(x,val_accs_4,label='1')
plt.legend()
plt.show()