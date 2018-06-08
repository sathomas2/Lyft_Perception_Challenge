from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess
from sklearn.metrics import recall_score, precision_score, f1_score

import helper 

sys.path.append("models")
from DeepLabV3_plus_2ASPP import build_deeplabv3_plus

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="data", 
                    help='Directory containing training data')

parser.add_argument('--model', type=str, default="DeepLabV3_plus-Res152", 
                    help='Which model to use as backbone')

parser.add_argument('--load_dir', type=str, default=None, 
                    help='Directory within checkpoints directory to load checks')

parser.add_argument('--save_dir', type=str, default="data", 
                    help='Directory within checkpoints directory to save checks')

parser.add_argument('--batch_size', type=int, default=15, 
                    help='Batch size to train')

parser.add_argument('--num_epochs', type=int, default=100, 
                    help='Number of epochs to train')

parser.add_argument('--save_every', type=int, default=1, 
                    help='How often to save during training and run on validation set')

parser.add_argument('--is_reduced', type=bool, default=False, 
                    help='Train on only road, vehicle, and void or on all classes')

parser.add_argument('--use_OLDsave', type=bool, default=False, 
                    help='If is_reduced and previous checkpoint used all 14 classes then True, else False')
args = parser.parse_args()

dataset = args.dataset #'data'
model = args.model #"DeepLabV3_plus-Res152"
load_dir = args.load_dir #None
save_dir = args.save_dir #'fri50_14C_1'
batch_size = args.batch_size #15
num_epochs = args.num_epochs #50
save_every = args.save_every #1
is_reduced = args.is_reduced #False
use_OLDsave = args.use_OLDsave #False

_, label_values = helper.get_label_info(os.path.join(dataset, "class_dict.csv"))
class_names_string = ""
num_classes = len(label_values)
input_names, output_names, val_input_names, val_output_names = helper.prepare_data()
#print(len(input_names), len(output_names), len(val_input_names), len(val_output_names))

tf.reset_default_graph()

if is_reduced:
    num_classes = 3
    inverse_class_weights = np.array([1.59310932, 2.87312183, 41.24863708])
else:
    inverse_class_weights = np.array([6.39387166, 12.84423503, 38.18780216, 116.79734864, 997.09103688, 
                                      120.63503461, 82.94638985, 2.97621278, 7.08734984, 5.64887416, 
                                      41.24863708, 79.2489474, 626.04653671, 58.58800969])

if "Res152" in model and not os.path.isfile("models/resnet_v2_152.ckpt"):
    helper.download_checkpoints("Res152")

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3], name='net_input')
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes], name='net_output')
loss_weights = tf.placeholder(tf.float32,shape=[None,None,None,num_classes], name='loss_weights')
do_rate = tf.placeholder(tf.float32, name='final_drop_rate')
train_mode = tf.placeholder(tf.bool, name='final_train_mode')

network, init_fn = build_deeplabv3_plus(net_input, num_classes, preset_model=model, is_training=train_mode, drop_rate=do_rate)
    
softmax = tf.nn.softmax(network, name='final_softmax')
labels = tf.argmax(net_output, axis=-1)
pred_labels = tf.argmax(network, axis=-1)
reg_constant = 0.05

losses = tf.losses.log_loss(net_output, softmax, weights=loss_weights, #softmax_like=False, offset_weight=1.5,
                            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

loss = losses + reg_constant*tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) 

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    opt = tf.train.AdamOptimizer(0.0001).minimize(loss, var_list=[var for var in tf.trainable_variables()])

if use_OLDsave:
    OLDsaver = tf.train.Saver([var for var in tf.trainable_variables()[:-2]])

saver = tf.train.Saver(max_to_keep=1)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    init_fn(sess)

    if load_dir is not None:
        print('Loaded latest model checkpoint')
        if use_OLDsave:
            OLDsaver.restore(sess, tf.train.latest_checkpoint('checkpoints/'+load_dir))
        else:
            saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'+load_dir))

    num_images = len(input_names)
    steps_per_epoch = num_images//batch_size + min(num_images%batch_size, 1)
   
    for epoch in range(num_epochs):
        current_losses = []
        car_fs_B = []
        road_fs_B = []
        cnt = 0

        st = time.time()
        print('Running on Train Set...')
        for train_x, train_y in helper.get_batches(input_names, output_names, batch_size, label_values,
                                                   mode='train', reduce=is_reduced):
            cnt+=1

            weights = np.ones_like(train_y)
            for i in range(num_classes):
                weights[:,:,:,i] = inverse_class_weights[i]
                    
            lls, p, _, batch_loss = sess.run([labels, pred_labels, opt, loss], feed_dict={net_input: train_x,
                                                                                          net_output: train_y,
                                                                                          loss_weights: weights,
                                                                                          train_mode: True, 
                                                                                          do_rate:0.1})
            current_losses.append(batch_loss)
            _, _, car_f, _, _, road_f = helper.get_rpf_scores(p, lls, reduce=is_reduced)
            car_fs_B.append(car_f)
            road_fs_B.append(road_f)
            if cnt%20==0:
                et = time.time()
                dt = et-st
                st = et
                print('Epoch: {}/{}...Steps: {}/{}...Loss: {:.4}...RoadF: {:.4}...CarF: {:.4}...DT: {:.4}'.format(
                    epoch+1,num_epochs, cnt, steps_per_epoch, np.mean(current_losses), 
                    np.mean(road_fs_B), np.mean(car_fs_B), dt))
        print('')                                                            
        if (epoch+1)%save_every==0:
            
            print('Running on Val Set...')
            test_losses = []
            test_ious = []
            car_fs_T = []
            road_fs_T = []
            for test_x, test_y in helper.get_batches(val_input_names, val_output_names, batch_size, 
                                                     label_values, mode='val', reduce=is_reduced):
                
                weights = np.ones_like(test_y)
                for i in range(num_classes):
                    weights[:,:,:,i] = inverse_class_weights[i]
                
                lls, p, test_loss = sess.run([labels, pred_labels, loss], 
                                             feed_dict={net_input: test_x,
                                                        net_output: test_y,
                                                        loss_weights: weights,
                                                        train_mode: False, 
                                                        do_rate: 0.0})
                test_losses.append(test_loss)
                _, _, car_f, _, _, road_f = helper.get_rpf_scores(p, lls, reduce=is_reduced)
                car_fs_T.append(car_f)
                road_fs_T.append(road_f)
                
            print('Val #s:')
            print('Epoch: {}/{}...Loss: {:.4}...RoadF: {:.4}...CarF: {:.4}'.format(epoch+1, num_epochs,
                                                                                    np.mean(test_losses), 
                                                                                    np.mean(road_fs_T), 
                                                                                    np.mean(car_fs_T)))
            
            if not os.path.isdir('checkpoints/'+save_dir):
                os.makedirs('checkpoints/'+save_dir)
            saver.save(sess, 'checkpoints/'+save_dir+'/Res152', global_step=epoch+1)
            print('Model Saved!')
            print('')
print('')
print('Model finished training :-)')