import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing

eeg = loadmat(r"./temp/Bonn_eeg_E_100.mat")
Y1 = eeg['eeg_data']

min_max_scaler = preprocessing.MinMaxScaler()
Y = min_max_scaler.fit_transform(Y1)
# Y = mylorenz(30)
X1 = Y.copy()
X=X1
delta=0.5

m = X.shape[0]
n = X.shape[1]

for i in range(1,m):
    X[i,]=X1[i,:]+delta*(X[i-1,:])

Accurate_predictions = 0
ii = 0
all=[]
real=[]

# while ii < 2000:
while ii < 1000:
    ii = ii + 1
    print(f'Case number: {ii/1}')
    
    INPUT_trainlength = 4
    selected_variables_idx = list(range(90))
    
    xx = X[3000 + ii:, selected_variables_idx].T
    
    noisestrength = 0
    xx_noise = xx + noisestrength * np.random.rand(*xx.shape)
    
    predict_len = 2
    
    start_idx = max(0, INPUT_trainlength - 3 * predict_len)
    traindata = xx_noise[:, start_idx:INPUT_trainlength]
    trainlength = traindata.shape[1]
    k = 60
    
    jd = 2
    
    D = xx_noise.shape[0]
    origin_real_y = xx[jd, :]
    real_y = xx[jd, start_idx:]
    real_y_noise = real_y + noisestrength * np.random.rand(*real_y.shape)
    traindata_y = real_y_noise[:trainlength]
    
    traindata_x_NN = traindata.copy()
    
    w_flag = np.zeros((traindata_x_NN.shape[0],))
    A = np.zeros((predict_len, traindata_x_NN.shape[0]))
    B = np.zeros((traindata_x_NN.shape[0], predict_len))
    
    predict_pred = np.zeros((predict_len - 1,))
    
    for iter_num in range(1000):
        other_idx = list(set(range(traindata_x_NN.shape[0])) - {jd})
        random_sample = np.random.choice(other_idx, k - 1, replace=False)
        random_idx = sorted([jd] + list(random_sample))
        
        traindata_x = traindata_x_NN[random_idx, :trainlength]
        
        for i in range(len(random_idx)):
            b = traindata_x[i, :trainlength - predict_len + 1]
            
            B_w = np.zeros((trainlength - predict_len + 1, predict_len))
            for j in range(trainlength - predict_len + 1):
                B_w[j, :] = traindata_y[j:j + predict_len]
            
            B_para = np.linalg.lstsq(B_w, b, rcond=None)[0]
            
            B[random_idx[i], :] = (B[random_idx[i], :] + B_para + 
                                   B_para * (1 - w_flag[random_idx[i]])) / 2
            w_flag[random_idx[i]] = 1
        
        super_bb = []
        super_AA = []
        
        for i in range(traindata_x_NN.shape[0]):
            kt = 0
            bb = []
            AA = np.zeros((predict_len - 1, predict_len - 1))
            
            for j in range(trainlength - (predict_len - 1), trainlength):
                bb_val = traindata_x_NN[i, j]
                col_known_y_num = trainlength - j
                
                for r in range(col_known_y_num):
                    bb_val = bb_val - B[i, r] * traindata_y[trainlength - col_known_y_num + r]
                
                AA[kt, :predict_len - col_known_y_num] = B[i, col_known_y_num:predict_len]
                bb.append(bb_val)
                kt += 1
            
            super_bb.extend(bb)
            super_AA.append(AA)
        
        super_bb = np.array(super_bb)
        super_AA = np.vstack(super_AA)
        
        pred_y_tmp = np.linalg.lstsq(super_AA, super_bb, rcond=None)[0]
        
        tmp_y = np.concatenate([real_y[:trainlength], pred_y_tmp])
        Ym = np.zeros((predict_len, trainlength))
        for j in range(predict_len):
            Ym[j, :] = tmp_y[j:j + trainlength]
        
        BX = np.hstack([B, traindata_x_NN])
        IY = np.hstack([np.eye(predict_len), Ym])
        A = IY @ np.linalg.pinv(BX)
        
        union_predict_y = []
        for j1 in range(predict_len - 1):
            tmp_y_list = []
            for j2 in range(j1, predict_len - 1):
                row = j2 + 1
                col = trainlength - j2 + j1 - 1
                tmp_y_list.append(A[row, :] @ traindata_x_NN[:, col])
            union_predict_y.append(np.mean(tmp_y_list))
        
        union_predict_y = np.array(union_predict_y)
        
        eof_error = np.sqrt(np.mean((union_predict_y - predict_pred) ** 2))
        if eof_error < 0.0001:
            break
        
        predict_pred = union_predict_y.copy()
    all=np.append(all,union_predict_y)
    myreal = real_y[trainlength:trainlength + predict_len - 1]
    real=np.append(real,myreal)
    RMSE = np.sqrt(np.mean((union_predict_y - myreal) ** 2))
    
    std_val = np.std(real_y[trainlength - 2 * predict_len:trainlength + predict_len - 1])
    RMSE = RMSE / (std_val + 0.001)
    
    if RMSE < 0.5:
        Accurate_predictions += 1
    
    Accurate_prediction_rate = Accurate_predictions / (ii / 2)
    print(f'Accurate_prediction_rate: {Accurate_prediction_rate}')
    print()
    
    refx = X[3000 + ii - 100:, :].T
    
    plt.figure(1, figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(refx[jd, :150], 'c-*', linewidth=2, markersize=4)
    plt.plot(range(100, 100 + INPUT_trainlength), 
            origin_real_y[:INPUT_trainlength], 'b-*', linewidth=2, markersize=4)
    plt.title(f'Original attractor. Init: {ii}, Noise strength: {noisestrength}', 
             fontsize=18)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(INPUT_trainlength), 
            origin_real_y[:INPUT_trainlength], 'b-*', linewidth=2, markersize=4, 
            label='Training data')
    plt.plot(range(INPUT_trainlength, INPUT_trainlength + predict_len - 1),
            origin_real_y[INPUT_trainlength:INPUT_trainlength + predict_len - 1], 
            'c-p', markersize=4, linewidth=2, label='True values')
    plt.plot(range(INPUT_trainlength, INPUT_trainlength + predict_len - 1),
            union_predict_y, 'ro', markersize=5, linewidth=2,
            label='predictions')
    plt.title(f'Pred: KnownLen={trainlength}, PredLen={predict_len-1}, RMSE={RMSE:.4f}',
             fontsize=18)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.pause(0.1)
    plt.savefig(f'./fig/result_case_{ii//2}.png', dpi=100, bbox_inches='tight')
    plt.close()
