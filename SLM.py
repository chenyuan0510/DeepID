import csv, sys, os, math
import numpy as np
import pandas as pd
import random
import time
#from sklearn.decomposition import PCA
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable

def data_loader(x, y, batch_size=50, mode='train'):
    def reader():
        if mode == 'train':
            train=np.reshape(x,[482,1,1,-1])
            #train=train[cvind[:,0]!=cv_num,:,:,:]
            trainy=y.astype('int64')
            #trainy=trainy[cvind[:,0]!=cv_num,:]
            rand_index = np.arange(0, trainy.shape[0])
            random.shuffle(rand_index)
            #batch_samples = []
            #batch_labels = []
            lsize=math.floor((trainy.shape[0]/round(trainy.shape[0]/batch_size)))
            loopr=round(trainy.shape[0]/batch_size)
            start_pos=0
            add_pos=list(range(loopr))
            random.shuffle(add_pos)
            add_pos=add_pos[0:trainy.shape[0]-lsize*loopr]
            for num_loop in range(loopr):
                if num_loop in add_pos:
                    train_array=train[rand_index[start_pos:start_pos+lsize+1],:,:,:]
                    labels_array=trainy[rand_index[start_pos:start_pos+lsize+1],0]
                    start_pos=start_pos+lsize+1
                else:
                    train_array=train[rand_index[start_pos:start_pos+lsize],:,:,:]
                    labels_array=trainy[rand_index[start_pos:start_pos+lsize],0]
                    start_pos=start_pos+lsize
                labels_array=labels_array.reshape(-1, 1)
                yield train_array, labels_array
        else:
            test=np.reshape(x,[x.shape[0],1,1,-1])
            testy=y.astype('int64')
            testy=testy.reshape(-1, 1)
            yield test, testy
    return reader

class MCLS2(fluid.dygraph.Layer):
    def __init__(self,c0):
        super(MCLS2, self).__init__()
        self.p1_1 = Conv2D(num_channels=1, num_filters=512, filter_size=(1,c0), stride=(1,c0), act='relu')
        self.pool1 = Pool2D(pool_size=(1,2), pool_stride=(1,2), pool_padding=0, pool_type='avg')#80
        self.p1_2 = Conv2D(num_channels=512, num_filters=256, filter_size=(1,3), padding=(0,1), stride=(1,1), act='relu')#80
        self.pool2 = Pool2D(pool_size=(1,2), pool_stride=(1,2), pool_padding=0, pool_type='avg')#40
        self.p1_3 = Conv2D(num_channels=256, num_filters=128, filter_size=(1,3), padding=(0,1), stride=1, act='relu')#40
        self.pool3 = Pool2D(pool_size=(1,2), pool_stride=(1,2), pool_padding=0, pool_type='avg')#20
        self.fc1 = Linear(input_dim=2560, output_dim=1000, act='relu')
        self.drop_ratio1 = 0.5
        self.fc2 = Linear(input_dim=1000, output_dim=64, act='relu')
        self.drop_ratio2 = 0.5
        self.fc3 = Linear(input_dim=64, output_dim=2, act='softmax')
    def forward(self, inputs):
        x = self.p1_1(inputs)
        x = self.pool1(x)
        x = self.p1_2(x)
        x = self.pool2(x)
        x = self.p1_3(x)
        x = self.pool3(x)
        x = fluid.layers.reshape(x, [x.shape[0], -1])
        outputs1 = self.fc1(x)
        outputs1= fluid.layers.dropout(outputs1, self.drop_ratio1)
        outputs2 = self.fc2(outputs1)
        outputs3= fluid.layers.dropout(outputs2, self.drop_ratio2)
        outputs_final = self.fc3(outputs3)
        return outputs_final

def evaluation(model,x,y,output_file):
    with fluid.dygraph.guard():
        #print('start evaluation .......')
        model_state_dict, _ = fluid.load_dygraph('SLM')
        model.load_dict(model_state_dict)
        model.eval()
        train_loader = data_loader(x, y, batch_size=50, mode='test')
        accuracies = []
        rlt=np.zeros([1,5])
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            pred = model(img)
            acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
            accuracies.append(acc.numpy())
            pred_lab = np.argsort(pred.numpy())
            rlt=np.vstack((rlt,np.hstack((pred.numpy(),pred_lab,y_data))))
        print("[validation] accuracy: {}".format(np.mean(accuracies)))
        np.savetxt(output_file, rlt, fmt='%f', delimiter=',')
        #return np.mean(accuracies)

if __name__ == '__main__':
    with fluid.dygraph.guard():
        x_file_in = sys.argv[1]
        y_file_in = sys.argv[2]
        output_file = sys.argv[3]
        test0=np.load('%s.npy'%x_file_in)
        test0y=np.load('%s.npy'%y_file_in)
        saved_feat=[158,159,114,58]
        tmp_train = train0[:,:,saved_feat]
        tmp_test = test0[:,:,saved_feat]
        model = MCLS2(len(saved_feat))
        #train(model, tmp_train, train0y)
        evaluation(model, tmp_test, test0y, output_file)