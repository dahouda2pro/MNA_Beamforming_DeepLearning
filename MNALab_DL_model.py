import os 
import numpy as np
import keras
from keras.layers import Dense,Dropout
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

# Reading input and output sets generated from MATLAB
In_set_file=loadmat('MNALab_DLBeam_dataset/DLCB_input.mat')
Out_set_file=loadmat('MNALab_DLBeam_dataset/DLCB_output.mat')

In_set=In_set_file['DL_input']
Out_set=Out_set_file['DL_output']

# Parameter initialization
num_user_tot=In_set.shape[0]
n_DL_size=[.001,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7]
count=0
num_tot_TX=2
num_beams=512

# Model training function
def train(X_train, y_train, X_test, y_test, epochs, batch_size,dr, num_hidden_layers, nodes_per_layer, loss_fn,n_BS,n_beams):
    
    in_shp = list(X_train.shape[1:])

    AP_models = []
    plot = 0
    for bs_idx in range(0, n_BS):
        idx = bs_idx*n_beams
        idx_str = 'BS%i'%bs_idx
        
        model = keras.Sequential()
        model.add(Dense(nodes_per_layer, activation='relu', kernel_initializer='he_normal', input_shape=in_shp))
        model.add(Dropout(dr))
        for h in range(num_hidden_layers):
            model.add(Dense(nodes_per_layer, activation='relu', kernel_initializer='he_normal'))
            model.add(Dropout(dr))
        
        model.add(Dense(n_beams, activation='relu', kernel_initializer='he_normal'))
        model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])
        model.summary()
        
        filepath = r'C:/Users/MWAMBA/MNA_Beamforming_DeepLearning/MNALab_DLBeam_code_output/Results_mmWave_ML_BS' + idx_str
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        history = model.fit(X_train,
                            y_train[:, idx:idx + n_beams],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            validation_data=(X_test, y_test[:,idx:idx + n_beams]),
                            callbacks = [
                                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
                            ])
    
        model.load_weights(filepath)
        
        AP_models.append(model)
        
         # Check how loss & mse went down
        epoch_loss = history.history['loss']
        epoch_val_loss = history.history['val_loss']
        epoch_mae = history.history['accuracy']
        epoch_val_mae = history.history['val_accuracy']

        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(range(0,len(epoch_loss)), epoch_loss, 'b-', linewidth=2, label='Train Loss')
        plt.plot(range(0,len(epoch_val_loss)), epoch_val_loss, 'r-', linewidth=2, label='Test Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title('Evolution of loss on train & validation sets over epochs')
        plt.legend(loc='best')

        plt.subplot(1,2,2)
        plt.plot(range(0,len(epoch_mae)), epoch_mae, 'b-', linewidth=2, label='Train Acc')
        plt.plot(range(0,len(epoch_val_mae)), epoch_val_mae, 'r-', linewidth=2,label='Test Acc')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title('Evolution of Accuracy on Train & Validation sets over epochs')
        plt.legend(loc='best')

        #plt.show()
        plt.savefig('Figure_LossAcc/Loss_Acc_'+str(plot)+'.png')
        plot = plot + 1
        
    return AP_models

for DL_size_ratio in n_DL_size:
    
    print (DL_size_ratio)
    count=count+1
    DL_size=int(num_user_tot*DL_size_ratio)
    
    np.random.seed(2022)
    n_examples = DL_size
    num_train  = int(DL_size * 0.8)
    num_test   = int(num_user_tot*.2)
    
    train_index = np.random.choice(range(0,num_user_tot), size=num_train, replace=False)
    rem_index = set(range(0,num_user_tot))-set(train_index)
    test_index= list(set(np.random.choice(list(rem_index), size=num_test, replace=False)))
    
    X_train = In_set[train_index]
    X_test =  In_set[test_index] 
        
    y_train = Out_set[train_index]
    y_test = Out_set[test_index]
    
    # Learning model parameters
    epochs = 25
    batch_size = 100  
    dr = 0.05                  # dropout rate  
    num_hidden_layers=4
    nodes_per_layer=X_train.shape[1]
    loss_fn='mean_squared_error'
    
    # Model training
    AP_models = train(X_train, y_train, X_test, y_test, epochs, batch_size,dr, num_hidden_layers, nodes_per_layer, loss_fn,num_tot_TX,num_beams)
    
    # Model running/testing
    DL_Result={}
    for idx in range(0,num_tot_TX,1): 
        beams_predicted=AP_models[idx].predict( X_test, batch_size=10, verbose=0)
    
        DL_Result['TX'+str(idx+1)+'Pred_Beams']=beams_predicted
        DL_Result['TX'+str(idx+1)+'Opt_Beams']=y_test[:,idx*num_beams:(idx+1)*num_beams]

    DL_Result['user_index']=test_index
    savemat('MNALab_DLBeam_code_output/DL_Result' + str(count) + '.mat', DL_Result)