import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, f1_score

import sys,os,time,random,pickle,ast 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils import _calculate_distance, _table_calculate_distance, _find_aoi_center_x_ratio, _find_aoi_center_y_ratio


class TransformerRegression(nn.Module):
    def __init__(self, input_size, num_output=4, d_model=64, nhead=8, num_encoder_layers=1, dropout=0.15):
        super(TransformerRegression, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)  # Convert 1D data to d_model size
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        self.decoder = nn.Linear(d_model, num_output)  # Output is a single regression value

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_length, d_model]
        x = x.permute(1, 0, 2)  # [seq_length, batch_size, d_model]
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Take the mean over the sequence
        x = self.decoder(x)
        return x

class TransformerClassification(nn.Module):
    def __init__(self, input_size, num_class, d_model=64, nhead=8, num_encoder_layers=1, dropout=0.15):
        super(TransformerClassification, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)  # Convert 1D data to d_model size
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        self.decoder = nn.Linear(d_model, num_class)  

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add batch dimension
        x = self.embedding(x)  # [batch_size, seq_length, d_model]
        x = x.permute(1, 0, 2)  # [seq_length, batch_size, d_model]
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Take the mean over the sequence
        x = self.decoder(x)
        return x


class RNNRegression(nn.Module):
    def __init__(self, input_size, num_output = 4, hidden_size=64, num_layers=1, dropout=0.15):
        super(RNNRegression, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN with multiple layers and dropout
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,  # Expecting input as (batch, seq, features)
        )

        # Fully connected layer for regression
        self.fc = nn.Linear(hidden_size, num_output)  # Output single regression value

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_size]
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state

        # rnn forward pass
        out, h_n = self.rnn(x, h_0)
        
        # Use the final hidden state for the regression
        out = self.fc(out[:, -1, :])  # Take only the last time step
        return out

class RNNClassification(nn.Module):
    def __init__(self, input_size, num_class, hidden_size=64, num_layers=1, dropout=0.15):
        super(RNNClassification, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_class)  # Output for multi-class classification

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add batch dimension
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, h_n = self.rnn(x, h_0)

        out = self.fc(out[:, -1, :])  # Take only the last time step
        return out





class Baseline_Simulation():
    def __init__(self,task_type,feature_type,slide_id,model_name,dataset_folder,log_folder):
        self.task_type = task_type
        self.feature_type = feature_type
        self.slide_id = slide_id
        self.model_name = model_name
        self.dataset_folder = dataset_folder
        self.log_folder = log_folder
        self.aoi_info_table = pd.read_csv(self.dataset_folder + '/' + 'aoi_material_ext_slide.csv',sep='\t')

        self.make_assertion()

        if task_type in ['gaze','cognitive']:
            self.train_file = self.dataset_folder + '/' + 'behavior' + '_' + feature_type + '_train_' + str(slide_id) + '.csv'
            self.test_file = self.dataset_folder + '/' + 'behavior' + '_' + feature_type + '_test_' + str(slide_id) + '.csv'
        elif task_type == 'question':
            self.train_file = self.dataset_folder + '/' + 'gkt' + '_' + feature_type + '_train_' + str(slide_id) + '.csv'
            self.test_file = self.dataset_folder + '/' + 'gkt' + '_' + feature_type + '_test_' + str(slide_id) + '.csv'
        else:
            raise ValueError('This task type is not supported!')

        self.model_root_folder = log_folder + '/' + task_type + '/' + feature_type + '/' + model_name + '/slide' + str(slide_id) + '/'
        os.makedirs(self.model_root_folder, exist_ok=True)
        

    def set_seed(self,seed_num):
        np.random.seed(seed_num)
        random.seed(seed_num)
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)  
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

    def make_assertion(self):
        assert self.task_type in ['gaze','question','cognitive']
        assert self.model_name in ['linear_regression','random_forest','decision_tree','rnn','transformer']

    def dataset_prepare(self,norm):
        train_raw = pd.read_csv(self.train_file,sep='\t')
        test_raw = pd.read_csv(self.test_file,sep='\t')

        print('len(train_raw),len(test_raw): ',len(train_raw),len(test_raw))

        header_encode_behavior = ['student_id','course_name','slide_id','transcript_id','gender','age','major','education','ML_familarity','ML_Experience','ML_bg_rate','course_content']
        header_encode_question = ['student_id','course_name','slide_id','question_id','gender','age','major','education','ML_familarity','ML_Experience','ML_bg_rate','question_content','choice_content','course_content']

        if self.task_type == 'gaze':
            header_x = [hx + '_' + self.feature_type for hx in header_encode_behavior] if self.feature_type in ['number'] else ['embedding_'+self.feature_type]
            header_y = ['gaze_aoi_id']
        elif self.task_type == 'cognitive':
            header_x = [hx + '_' + self.feature_type for hx in header_encode_behavior] if self.feature_type in ['number'] else ['embedding_'+self.feature_type]
            header_y = ['workload','curiosity','valid_focus','course_follow']
        elif self.task_type == 'question':
            header_x = [hx + '_' + self.feature_type for hx in header_encode_question] if self.feature_type in ['number'] else ['embedding_'+self.feature_type]
            header_y = ['correctness']
        else:
            raise ValueError('Task type not supported yet!')

        train_data = train_raw[header_x+header_y+['uid']].copy().dropna()
        test_data = test_raw[header_x+header_y+['uid']].copy().dropna()

        train_data = train_data[train_data['gaze_aoi_id']!=-1] if self.task_type == 'gaze' else train_data
        test_data = test_data[test_data['gaze_aoi_id']!=-1] if self.task_type == 'gaze' else test_data

        def string_to_list(s):
            return ast.literal_eval(s)

        if self.feature_type == 'number':
            x_train = np.array(train_data[header_x]) 
            x_test = np.array(test_data[header_x])
        
        elif self.feature_type in ['bert','llm']:
            train_data['embedding_'+self.feature_type] = train_data['embedding_'+self.feature_type].apply(string_to_list)
            test_data['embedding_'+self.feature_type] = test_data['embedding_'+self.feature_type].apply(string_to_list)

            x_train = np.array(train_data['embedding_'+self.feature_type].tolist())
            x_test = np.array(test_data['embedding_'+self.feature_type].tolist())
        else:
            raise ValueError('This feature type is not supported yet in dataset preparation!')
        
        y_train = np.array(train_data[header_y])
        y_test = np.array(test_data[header_y])

        print('x_train.shape: ',x_train.shape)

        self.header_y = header_y

        self.input_size, self.output_size = x_train.shape[1], y_train.shape[1]
        print('input_size: ',self.input_size,', output_size: ',self.output_size)

        if self.task_type != 'cognitive':
            self.num_class = max([int(np.max(y_train))+1,int(np.max(y_test))+1])
            print('num_class: ',self.num_class)

        self.uid_test = np.array(test_data[['uid']]).reshape((len(test_data),1))

        if norm == True:
            self.scaler = StandardScaler()
            self.scaler.fit(x_train)
            x_train = self.scaler.transform(x_train)
            x_test = self.scaler.transform(x_test)
        
        if self.model_name == 'linear_regression' and self.feature_type in ['bert','llm']:
            pca = PCA(n_components=0.95)  # Keep 95% of the variance
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)


        print('x_train.shape, y_train.shape, x_test.shape, y_test.shape, np.max(y_train), np.min(y_train): ', x_train.shape, y_train.shape, x_test.shape, y_test.shape, np.max(y_train), np.min(y_train))
        return x_train, y_train, x_test, y_test
    
    def dataset_train_valid_loader(self,x_train,y_train,batch_size=8,val_ratio=0.2):
        if self.task_type == 'cognitive':
            train_dataset = TensorDataset(torch.FloatTensor(x_train).unsqueeze(1), torch.FloatTensor(y_train))
        else:
            train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train.reshape((len(y_train),))))
        val_size = int(len(train_dataset) * val_ratio)
        train_size = len(train_dataset) - val_size

        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader


    def dataset_test_loader(self,x_test,y_test,batch_size=8):
        if self.task_type == 'cognitive':
            test_dataset = TensorDataset(torch.FloatTensor(x_test).unsqueeze(1), torch.FloatTensor(y_test))
        else:
            test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test.reshape((len(y_test),))))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader

    def model_init(self,load_model):
        if self.model_name == 'random_forest': 
            self.model = RandomForestRegressor() if self.task_type == 'cognitive' else RandomForestClassifier()
        elif self.model_name == 'linear_regression': 
            if self.task_type == 'cognitive':
                self.model = LinearRegression()
            elif self.task_type == 'gaze':
                self.model = LogisticRegression(multi_class='multinomial')
            elif self.task_type == 'question':
                self.model = LogisticRegression()
        elif self.model_name == 'decision_tree': 
            self.model = DecisionTreeRegressor() if self.task_type == 'cognitive' else DecisionTreeClassifier()
        elif self.model_name in ['transformer','rnn']:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Using device:", self.device)

            if self.model_name == 'transformer':
                self.model = TransformerRegression(input_size = self.input_size).to(self.device) if self.task_type == 'cognitive' else TransformerClassification(input_size = self.input_size, num_class = self.num_class).to(self.device)
            elif self.model_name == 'rnn':
                self.model = RNNRegression(input_size = self.input_size).to(self.device) if self.task_type == 'cognitive' else RNNClassification(input_size = self.input_size, num_class = self.num_class).to(self.device)
            
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.MSELoss().to(self.device) if self.task_type == 'cognitive' else nn.CrossEntropyLoss().to(self.device)

            if load_model == True:
                checkpoint_path = os.path.join(self.model_root_folder, f'{self.model_name}.pt')
                checkpoint = torch.load(checkpoint_path)
                self.epoch_exist = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.train_losses = checkpoint['train_loss']
                self.val_losses = checkpoint['val_loss']
            else:
                self.epoch_exist = 0
                self.train_losses = []
                self.val_losses = []

    def model_train(self,x_train,y_train,num_epochs = 1,vis = True,print_epoch_num = 1,batch_size = 8):
        if self.model_name in ['random_forest','linear_regression','decision_tree']:
            self.model.fit(x_train, y_train)
            self.save_ml_model(self.model, self.model_root_folder+'/'+f'{self.model_name}.pkl')
        elif self.model_name in ['transformer','rnn']:
            train_loader, val_loader = self.dataset_train_valid_loader(x_train,y_train,batch_size=batch_size)
            os.remove(self.model_root_folder+'/'+self.model_name+'_loss.csv') if os.path.exists(self.model_root_folder+'/'+self.model_name+'_loss.csv') else None
            with open(self.model_root_folder+'/'+self.model_name+'_loss.csv', "a+") as file1:
                file1.write('train_loss'+','+'val_loss'+'\n')

            for epoch in range(num_epochs):
                if epoch + 1 <= self.epoch_exist: continue

                self.model.train()
                epoch_train_loss = 0.0
                t1 = time.time()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()  # Clear the gradients
                    outputs = self.model(inputs)  # Forward pass
                    loss = self.criterion(outputs, targets)  # Compute the loss
                    loss.backward()  # Backpropagation
                    self.optimizer.step()  # Update weights
                    epoch_train_loss += loss.item()
                t2 = time.time()
                delta_time1 = t2-t1
                # Record the average training loss for this epoch
                epoch_train_loss_item = epoch_train_loss / len(train_loader)
                self.train_losses.append(epoch_train_loss_item)
                
                # Validation phase after training epoch
                self.model.eval()  # Set model to evaluation mode
                epoch_val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)  # Move to CUDA
                        outputs = self.model(inputs)  # Forward pass
                        loss = self.criterion(outputs, targets)  # Calculate validation loss
                        epoch_val_loss += loss.item()
                
                epoch_val_loss_item = epoch_val_loss / len(val_loader)
                self.val_losses.append(epoch_val_loss_item)
                t3 = time.time()
                delta_time2 = t3-t2
                
                with open(self.model_root_folder+'/'+self.model_name+'_loss.csv', "a+") as file1:
                    file1.write(str(epoch_train_loss_item)+','+str(epoch_val_loss_item)+'\n')

                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': self.train_losses,
                    'val_loss': self.val_losses
                }
                checkpoint_path = os.path.join(self.model_root_folder+'/', f'{self.model_name}.pt')
                checkpoint_epoch_path = os.path.join(self.model_root_folder+'/', f'{self.model_name}_epoch_{epoch + 1}.pt')
                if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
                if os.path.exists(checkpoint_epoch_path): os.remove(checkpoint_epoch_path)
                torch.save(checkpoint, checkpoint_path)
                torch.save(checkpoint, checkpoint_epoch_path)

                if (len(self.val_losses) == 0) or (len(self.val_losses) != 0 and epoch_val_loss_item == min(self.val_losses)):
                    checkpoint_best_path = os.path.join(self.model_root_folder+'/', f'{self.model_name}_best.pt')
                    if os.path.exists(checkpoint_best_path): os.remove(checkpoint_best_path)
                    torch.save(checkpoint, checkpoint_best_path)

                if epoch % print_epoch_num == 0:
                    print(f"Model saved! Epoch {epoch + 1}/{num_epochs}, Train Loss: {self.train_losses[-1]:.4f}, Validation Loss: {self.val_losses[-1]:.4f}, Train Time: {delta_time1:.4f}, Val Time: {delta_time2:.4f}")

            if vis == True:
                fig, ax = plt.subplots()
                plt.plot(self.train_losses, label='Train Loss')
                plt.plot(self.val_losses, label='Validation Loss')
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training and Validation Loss")
                plt.legend()
                plt.savefig(self.model_root_folder+'/'+self.model_name+'_train.png')

    def save_ml_model(self, model, filename):
        with open(filename, 'wb') as file1:
            pickle.dump(model, file1)
        print(f"Model saved to {filename}")

    def load_ml_model(self, filename):
        with open(filename, 'rb') as file1:
            model = pickle.load(file1)
        print(f"Model loaded from {filename}")
        return model


    def model_test(self,x_test,y_test,test_epoch = -1,vis = True,batch_size = 8):
        output_file_predict = self.model_root_folder+'/'+self.model_name+'_predict.csv'
        header_predict = ['uid','predict','label'] if self.task_type != 'cognitive' else ['uid'] + ['predict_'+h_y for h_y in self.header_y] + ['label_'+h_y for h_y in self.header_y]

        if self.model_name in ['random_forest','linear_regression','decision_tree']:
            self.model = self.load_ml_model(self.model_root_folder+'/'+f'{self.model_name}.pkl')

            y_pred = self.model.predict(x_test)
            y_pred = np.array(y_pred).reshape((len(y_pred),self.output_size))
            y_label = np.array(y_test).reshape((len(y_test),self.output_size))

            predict_data = pd.DataFrame(np.concatenate((self.uid_test,y_pred,y_label),axis=1),columns=header_predict)
            predict_data.to_csv(output_file_predict,index=False)
            
        elif self.model_name in ['transformer','rnn']:
            if test_epoch == -1:
                checkpoint_path = os.path.join(self.model_root_folder+'/', f'{self.model_name}_best.pt')
            else:
                checkpoint_path = os.path.join(self.model_root_folder+'/', f'{self.model_name}_epoch_{int(test_epoch)}.pt')
            checkpoint = torch.load(checkpoint_path)
            print('load model path for testing: ',checkpoint_path,' epoch: ',checkpoint['epoch'])

            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.model.eval()
            test_loader = self.dataset_test_loader(x_test,y_test,batch_size=batch_size)
            y_pred = np.empty((0,self.output_size))
            y_label = np.empty((0,self.output_size))

            os.remove(output_file_predict) if os.path.exists(output_file_predict) else None
            with open(output_file_predict, "a+") as file1:
                file1.write(','.join(header_predict)+'\n')

            with torch.no_grad():
                idx = 0
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    if self.task_type != 'cognitive':
                        _, outputs = torch.max(outputs, 1)

                    pred_item = outputs.cpu().numpy()
                    label_item = targets.cpu().numpy()

                    if self.task_type == 'cognitive':
                        pred_item = pred_item.reshape((pred_item.shape[0],pred_item.shape[-1]))
                    else:
                        pred_item = pred_item.reshape((len(pred_item),1))
                        label_item = label_item.reshape((len(label_item),1))
                    y_pred = np.concatenate((y_pred,pred_item),axis=0)
                    y_label = np.concatenate((y_label,label_item),axis=0)

                    for di in range(len(pred_item)):
                        with open(output_file_predict, "a+") as file1:
                            if self.task_type != 'cognitive':
                                file1.write(str(self.uid_test[idx][0])+','+str(pred_item[di][0])+','+str(label_item[di][0])+'\n')
                            else:
                                file1.write(','.join([str(self.uid_test[idx][0])]+[str(h_y) for h_y in pred_item[di]]+[str(h_y) for h_y in label_item[di]])+'\n')
                        idx += 1
                    
        if vis == True:
            fig, ax = plt.subplots(1,self.output_size,figsize=(3*self.output_size,3))
            if self.task_type == 'cognitive':
                for v in range(self.output_size):
                    ax[v].scatter(y_label[:,v:v+1], y_pred[:,v:v+1], color='red', label='Predicted')
                    plt.xlabel('y_label')
                    plt.ylabel('y_pred')
            else:
                ax.scatter(y_label, y_pred, color='red', label='Predicted')
            plt.title(self.model_name)
            plt.legend()
            plt.savefig(self.model_root_folder+'/'+self.model_name+'_test.png')


        if self.task_type == 'cognitive':
            metric_value = mean_squared_error(y_label, y_pred)  
        elif self.task_type == 'gaze':
            test_set_table = pd.read_csv(self.test_file,sep='\t')
            predict_table = pd.read_csv(output_file_predict)
            predict_table_merged = pd.merge(predict_table, test_set_table, on='uid')
            predict_table_merged['user_gaze_aoi_center_tuple_x'] = predict_table_merged['gaze_aoi_center_x_ratio']
            predict_table_merged['user_gaze_aoi_center_tuple_y'] = predict_table_merged['gaze_aoi_center_y_ratio']

            predict_table_merged['agent_gaze_aoi_center_tuple_x'] = predict_table_merged.apply(_find_aoi_center_x_ratio, axis=1, args=(self.aoi_info_table,'predict',))
            predict_table_merged['agent_gaze_aoi_center_tuple_y'] = predict_table_merged.apply(_find_aoi_center_y_ratio, axis=1, args=(self.aoi_info_table,'predict',))

            predict_table_merged['label_gaze_aoi_center_tuple_x'] = predict_table_merged.apply(_find_aoi_center_x_ratio, axis=1, args=(self.aoi_info_table,'label',))
            predict_table_merged['label_gaze_aoi_center_tuple_y'] = predict_table_merged.apply(_find_aoi_center_y_ratio, axis=1, args=(self.aoi_info_table,'label',))

            assert predict_table_merged['label_gaze_aoi_center_tuple_x'].equals(predict_table_merged['user_gaze_aoi_center_tuple_x'])
            assert predict_table_merged['label_gaze_aoi_center_tuple_y'].equals(predict_table_merged['user_gaze_aoi_center_tuple_y'])

            predict_table_merged = predict_table_merged[(predict_table_merged['agent_gaze_aoi_center_tuple_x']!=None)&(predict_table_merged['agent_gaze_aoi_center_tuple_y']!=None)]

            predict_table_merged['gaze_aoi_distance'] = predict_table_merged.apply(_table_calculate_distance, axis=1)
            metric_value = predict_table_merged['gaze_aoi_distance'].mean()
        else:
            metric_value = f1_score(y_label, y_pred, average='weighted')
        print(f'\n test set result: {metric_value}')
        
        return metric_value

    def model_loop(self,train_model_flag,load_model,test_epoch,num_epochs,batch_size,vis,norm):
        self.set_seed(4)

        x_train, y_train, x_test, y_test = self.dataset_prepare(norm)
        self.model_init(load_model)
        if train_model_flag == True:
            self.model_train(x_train,y_train,num_epochs = num_epochs,vis = vis,print_epoch_num = 1,batch_size = batch_size)
        metric_value = self.model_test(x_test,y_test,test_epoch=test_epoch,vis = vis,batch_size = batch_size)
        return metric_value


def run_exp(exp_config):
    with open(exp_config['output_file'], "a+") as file1:
        file1.write('task_type,feature_type,slide_id,model_name,metric_value\n')
    for task_type in exp_config['task_type_list']:
        for feature_type in exp_config['feature_type_list']:
            for slide_id in exp_config['slide_id_list']:
                for model_name in exp_config['model_name_list']:
                    print('\n'+'*'*40+f' Running experiment in task: {task_type}, feature: {feature_type}, slide: {slide_id}, model: {model_name}\n')
                    base_pipeline = Baseline_Simulation(task_type = task_type, feature_type = feature_type, slide_id = slide_id, model_name = model_name, dataset_folder = exp_config['dataset_folder'], log_folder = exp_config['log_folder'])
                    metric_value = base_pipeline.model_loop(exp_config['train_model_flag'],exp_config['load_model'],exp_config['test_epoch'],exp_config['num_epochs'],exp_config['batch_size'],exp_config['vis'],exp_config['norm'])

                    with open(exp_config['output_file'], "a+") as file1:
                        file1.write(task_type+','+feature_type+','+str(slide_id)+','+model_name+','+str(metric_value)+'\n')


# Task 1: question performance simulation
# model input: persona category, question contents, course contents
# model output: question accuracy

# Task 2,3: behavior simulation (gaze, cognitive states)
# model input: persona category, course contents
# model output: gaze AOI ID, cognitive states

# train set: past slide data, test set: current slide data

# encode type: 
# 1. unique number, 
# 2. Bert embedding for ALL concatenated textual input of LLMs together as a WHOLE, 
# 3. OpenAI LLM embedding for ALL concatenated textual input of LLMs together as a WHOLE.

exp_config = {
    'model_name_list': ['linear_regression','random_forest','decision_tree','rnn','transformer'],
    # 'model_name_list': ['linear_regression'],
    'feature_type_list': ['number','bert','llm'],
    # 'feature_type_list': ['number'],
    'task_type_list': ['cognitive','gaze','question'],
    'slide_id_list': [0,1,2],
    'num_epochs': 200,
    'batch_size': 32,
    'vis': True,
    'norm': True,
    'load_model': False,  # if True, the old model is used. otherwise, the old model is overlapped.
    'train_model_flag': True,
    'test_epoch': -1,
    'dataset_folder': '/dataset/dataset_dl_4',
    'log_folder': '/log/',
    'output_file': 'baseline_simulation.csv',
}

run_exp(exp_config)





