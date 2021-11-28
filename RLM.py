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

def data_loader(xfile, yfile, batch_size=50, mode='train'):
    def reader():
        if mode == 'train':
            train=xfile.astype('float32')
            trainy=yfile.astype('int64')
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
                    train_array=train[rand_index[start_pos:start_pos+lsize+1],:]
                    labels_array=trainy[rand_index[start_pos:start_pos+lsize+1],0]
                    start_pos=start_pos+lsize+1
                else:
                    train_array=train[rand_index[start_pos:start_pos+lsize],:]
                    labels_array=trainy[rand_index[start_pos:start_pos+lsize],0]
                    start_pos=start_pos+lsize
                labels_array=labels_array.reshape(-1, 1)
                #imgs_array = np.array(batch_samples).astype('float32')
                #labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield train_array, labels_array
        else:
            testx=xfile.astype('float32')
            testy=yfile.astype('int64')
            testy=testy.reshape(-1, 1)
            yield testx, testy
    return reader

class myfcm(fluid.dygraph.Layer):
    def __init__(self,c0):
        super(myfcm, self).__init__()
        self.fc1 = Linear(input_dim=c0, output_dim=256, act='relu') 
        self.drop_ratio1 = 0.3
        self.fc2 = Linear(input_dim=256, output_dim=64, act='relu')
        self.drop_ratio2 = 0.3
        self.fc3 = Linear(input_dim=64, output_dim=2, act='softmax')
    def forward(self, inputs):
        #inputs = fluid.layers.reshape(inputs, [inputs.shape[0], -1])
        outputs1 = self.fc1(inputs)
        outputs1= fluid.layers.dropout(outputs1, self.drop_ratio1)
        outputs2 = self.fc2(outputs1)
        outputs2= fluid.layers.dropout(outputs2, self.drop_ratio2)
        outputs_final = self.fc3(outputs2)
        return outputs_final

def evaluation(model, xfile, yfile):
    #with fluid.dygraph.guard():
    with fluid.dygraph.guard(place=fluid.CPUPlace()):
        print('start evaluation .......')
        model_state_dict, _ = fluid.load_dygraph('RLM')
        #model_state_dict, _ = fluid.load_dygraph('model_pc')
        model.load_dict(model_state_dict)
        model.eval()
        train_loader = data_loader(xfile, yfile, batch_size=50, mode='test')
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
        np.savetxt('RLM_test_rlt.csv', rlt, fmt='%f', delimiter=',')
        return np.mean(accuracies)

if __name__ == '__main__':
    with fluid.dygraph.guard():
        epoch_num=500
        testfile=np.load('test_repertoire_level_features.npy')
        #testyfile=np.zeros([testfile.shape[0],1])
        testyfile=np.load('y_test.npy')
        model = myfcm(testfile.shape[1])
        #train(model, trainfile, trainyfile, featnum)
        acc = evaluation(model, testfile, testyfile)
        print("RLM_test acc is: {}".format(acc))