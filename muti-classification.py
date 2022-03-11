import csv, sys, os, math
import numpy as np
import pandas as pd
import random
import time
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable

def data_loader(x,y,batch_size=50,mode='train'):
    def reader():
        if mode == 'train':
            train=xfile.astype('float32')
            trainy=yfile.astype('int64')
            rand_index = np.arange(0, trainy.shape[0])
            random.shuffle(rand_index)
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
                yield train_array, labels_array
        else:
            testx=xfile.astype('float32')
            testy=yfile.astype('int64')
            testy=testy.reshape(-1, 1)
            yield test, testy
    return reader

class MCF(fluid.dygraph.Layer):
    def __init__(self,c0):
        super(MCF, self).__init__()
        self.fc1 = Linear(input_dim=c0, output_dim=256, act='relu') 
        self.drop_ratio1 = 0.3
        self.fc2 = Linear(input_dim=256, output_dim=64, act='relu')
        self.drop_ratio2 = 0.3
        self.fc3 = Linear(input_dim=64, output_dim=4, act='softmax')
    def forward(self, inputs):
        #inputs = fluid.layers.reshape(inputs, [inputs.shape[0], -1])
        outputs1 = self.fc1(inputs)
        outputs1= fluid.layers.dropout(outputs1, self.drop_ratio1)
        outputs2 = self.fc2(outputs1)
        outputs2= fluid.layers.dropout(outputs2, self.drop_ratio2)
        outputs_final = self.fc3(outputs2)
        return outputs_final

def evaluation(model, xfile, yfile,output_file):
    with fluid.dygraph.guard():
        #print('start evaluation .......')
        model_state_dict, _ = fluid.load_dygraph('muti-classification')
        model.load_dict(model_state_dict)
        model.eval()
        train_loader = data_loader(xfile, yfile, batch_size=100, mode='test')
        accuracies = []
        #rlt=np.zeros([1,5])
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            pred = model(img)
            acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
            accuracies.append(acc.numpy())
            pred_lab = np.argsort(pred.numpy())
            rlt=np.hstack((pred.numpy(),pred_lab,y_data))
        print("[validation] accuracy: {}".format(np.mean(accuracies)))
        np.savetxt(output_file, rlt, fmt='%f', delimiter=',')
        return np.mean(accuracies)
if __name__ == '__main__':
    with fluid.dygraph.guard():
        x_file_in = sys.argv[1]
        y_file_in = sys.argv[2]
        output_file = sys.argv[3]
        testfile=np.load('%s.npy'%x_file_in)
        testyfile=np.load('%s.npy'%y_file_in)
        model = MCF(testfile.shape[1])
        acc = evaluation(model, testfile, testyfile, output_file)
        print("test acc is: {}".format(acc))
