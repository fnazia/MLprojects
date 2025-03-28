
import numpy as np
import torch
import time
import copy

class ConvolutionalNeuralNetwork:
    
    def __init__(self, 
                 in_channels, 
                 width,
                 height,
                 n_units_in_conv_layers, 
                 kernel_size_and_stride, 
                 n_units_in_fc_layers, 
                 classes, 
                 use_gpu = False, 
                 dropout = None, 
                 bnorm = False):
        
        self.train_error_trace = []
        self.valid_error_trace = []
        self.best_validation_loss = None
        self.training_time = None
        self.device = self.select_device(use_gpu)
        
        self.width = width
        self.height = height
        self.classes = classes
        
        in_units = in_channels if n_units_in_conv_layers else in_channels*width*height
        out_units = len(classes)
        self.nnet = self.construct_network(in_units, 
                                           n_units_in_conv_layers, 
                                           kernel_size_and_stride, 
                                           n_units_in_fc_layers, 
                                           out_units, 
                                           dropout, 
                                           bnorm)
        
        # for the purpose of saving the architecture
        self.args = [in_channels, width, height, 
                     n_units_in_conv_layers, kernel_size_and_stride, n_units_in_fc_layers, 
                     classes, use_gpu, dropout, bnorm]
    
    def select_device(self, use_gpu):
        if torch.cuda.is_available() and use_gpu:
            return 'cuda'
        if torch.backends.mps.is_built() and use_gpu:
            return torch.device('mps')
        return 'cpu'
        
    def construct_network(self, 
                          in_units, 
                          n_units_in_conv_layers, 
                          kernel_size_and_stride, 
                          n_units_in_fc_layers, 
                          out_units, 
                          dropout, 
                          bnorm):
        nnet = torch.nn.Sequential()
        nnet, in_units = self.network_body(nnet, in_units, n_units_in_conv_layers, kernel_size_and_stride, n_units_in_fc_layers, dropout, bnorm)
        output_layer = self._final_output_layer(in_units, out_units)
        nnet.add_module('output_final', output_layer)
        return nnet
    
    def network_body(self, nnet, in_units, n_units_in_conv_layers, kernel_size_and_stride, n_units_in_fc_layers, dropout, bnorm):
        layer = 0
        if n_units_in_conv_layers:
            nnet, in_units, layer = self._convolution_layers(nnet, in_units, n_units_in_conv_layers, kernel_size_and_stride, layer, dropout, bnorm)
        nnet.add_module('flatten', torch.nn.Flatten())
        if n_units_in_fc_layers:
            nnet, in_units, layer = self._linear_layers(nnet, in_units, n_units_in_fc_layers, layer)
        return nnet, in_units
    
    def _convolution_layers(self, nnet, in_units, n_units_in_conv_layers, kernel_size_and_stride, layer, dropout, bnorm):
        for units, kernel_stride in zip(n_units_in_conv_layers, kernel_size_and_stride):
            kernel, stride = kernel_stride
            nnet.add_module(f'conv_{layer}', torch.nn.Conv2d(in_units, units, kernel, stride))
            if dropout is not None:
                nnet.add_module(f'dropout_{layer}', torch.nn.Dropout(dropout))
            nnet.add_module(f'activation_{layer}', torch.nn.Tanh()) 
            if bnorm:
                nnet.add_module(f'bnorm_{layer}', torch.nn.BatchNorm2d(units))
            in_units = units
            self.width = (self.width - kernel) // stride + 1
            self.height = (self.height - kernel) // stride + 1
            layer += 1
        in_units = in_units*self.width*self.height
        return nnet, in_units, layer
    
    def _linear_layers(self, nnet, in_units, n_units_in_fc_layers, layer):
        for units in n_units_in_fc_layers:
            nnet.add_module(f'linear_{layer}', torch.nn.Linear(in_units, units))
            nnet.add_module(f'activation_{layer}', torch.nn.Tanh())
            in_units = units
            layer += 1
        return nnet, in_units, layer
    
    def _final_output_layer(self, in_units, out_units):
        output_layer = torch.nn.Linear(in_units, out_units)
        return output_layer
    
    def setup_optimizer(self, learning_rate):
        return torch.optim.Adam(self.nnet.parameters(), lr = learning_rate)
    
    def setup_loss_function(self):
        return torch.nn.CrossEntropyLoss()
    
    def train(self, trainloader, validloader, learning_rate, n_epochs, best_vloss = None, states_path = './model_states.pth'):
        if best_vloss is None:
            best_vloss = 1_000_000
        
        optimizer = self.setup_optimizer(learning_rate)
        loss_F = self.setup_loss_function()
        self.nnet = self.nnet.to(self.device)
        start_time = time.time()
        
        for epoch in range(n_epochs):
            epoch_error = []
            for batch in trainloader:
                X = batch['features'].to(self.device)
                T = batch['targets'].to(self.device) #.numpy()
                #if T.ndim == 1:
                #    T = T.reshape((-1, 1))
                #_, T = np.where(T == self.classes)
                #T = torch.tensor(T.reshape(-1)).to(self.device)
                T = T.reshape(-1)
                self.nnet.train()
                optimizer.zero_grad()
                Y = self.nnet(X)
                error = loss_F(Y, T)
                epoch_error.append(error.item())
                error.backward()
                optimizer.step()
            self.train_error_trace.append(np.mean(epoch_error))
            
            # Validation
            valid_error = self.validate(validloader, loss_F)
            if valid_error < best_vloss:
                best_vloss = valid_error
                self.save_model_states(optimizer, epoch, loss_F, states_path)
                
        self.training_time = time.time() - start_time
        self.best_validation_loss = best_vloss
        
    def validate(self, validloader, loss_F):
        epoch_error = []
        self.nnet.eval()
        with torch.no_grad():
            for batch in validloader:
                X = batch['features'].to(self.device)
                T = batch['targets'].to(self.device) #.numpy()
                #if T.ndim == 1:
                #    T = T.reshape((-1, 1))
                #_, T = np.where(T == self.classes)
                #T = torch.tensor(T.reshape(-1)).to(self.device)
                T = T.reshape(-1)
                Y = self.nnet(X)
                error = loss_F(Y, T)
                epoch_error.append(error.item())
        valid_error = np.mean(epoch_error)
        self.valid_error_trace.append(valid_error)
        return valid_error
    
    def predict(self, testloader, test_device = 'cpu'): # , evaluate = False
        Y_classes=[]
        Y_confidence = []
        #T_classes = []
        if self.device != test_device:
            self.nnet = self.nnet.to(test_device)
        self.nnet.eval()
        with torch.no_grad():
            if isinstance(testloader, np.ndarray):
                X = torch.from_numpy(testloader).to(test_device)
                Y = self.nnet(X)
                classes, confidence = self.class_and_confidence(Y)
                if test_device in ['cuda', 'mps']:
                    classes, confidence = classes.cpu(), confidence.cpu().numpy()
                return classes, confidence
            for batch in testloader:
                X = batch['features'].to(test_device)
                Y = self.nnet(X)
                classes, confidence = self.class_and_confidence(Y)
                if test_device in ['cuda', 'mps']:
                    classes, confidence = classes.cpu(), confidence.cpu().numpy()
                #confidence = confidence.numpy()
                Y_classes.append(classes)
                Y_confidence.append(confidence)
                #if evaluate:
                #    T = self.classes[batch['targets'].numpy()]
                #    T_classes.append(T.reshape(-1))
        #if evaluate:                   
        #    return np.concatenate(Y_classes), np.concatenate(Y_confidence), np.concatenate(T_classes)
        return np.concatenate(Y_classes), np.concatenate(Y_confidence)           
                
    def class_and_confidence(self, Y):
        Y_softmax = torch.nn.functional.softmax(Y, dim = 1)
        probs_pred, classes_pred = torch.max(Y_softmax, dim = 1)
        classes = self.classes[classes_pred]
        return classes, probs_pred
    
    def get_error_trace(self):
        return self.train_error_trace, self.valid_error_trace
    
    def get_best_validation_loss(self):
        return self.best_validation_loss
    
    def get_training_time(self):
        return self.training_time
    
    def evaluate(self, testloader, device = 'cpu'):
        Y_preds, Y_confs = self.predict(testloader, device) #True
        T = np.concatenate([self.classes[batch['targets'].numpy()].reshape(-1) for batch in testloader])
        if Y_preds.ndim != T.ndim:
            Y_preds, T = Y_preds.reshape(-1, 1), T.reshape(-1, 1)
        accuracy = np.sum(Y_preds == T) / T.shape[0]
        return accuracy
    
    def save_model_states(self, optimizer, epoch, loss_F, states_path):
        checkpoint = {'model_state_dict': self.nnet.state_dict(), 
                      'optimizer_state_dict': optimizer.state_dict(), 
                      'model_args': self.args,
                      'epoch': epoch, 
                      'loss': loss_F}
        torch.save(checkpoint, states_path)

