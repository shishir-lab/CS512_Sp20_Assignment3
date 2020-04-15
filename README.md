Assignment 3 LSTM and Adversarial training 

Classifier.py defines the forward pass for all the models.
training.py trains the basic LSTM model and stores the weights in a file. We have saved the weights for basicLSTM in "basic_lstm_model" file.
adv_training.py trains the adversarial network with initial weights set to the saved model in basic_lstm_model. Executing this code will generate a plot for different values of epsilon.
ProxLSTM.py calculates the Jacobian matrix and the backward pass.
Q6_Clasifier.py is for ProxLSTM with dropout and batch normalization and Q6_training,py trains that model.
