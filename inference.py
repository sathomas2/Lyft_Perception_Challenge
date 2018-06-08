import tensorflow as tf
import cv2
import numpy as np
import sys, json, base64
from io import BytesIO, StringIO
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

fn = sys.argv[-1]

def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def pad_frame(img):
    bottom_padding = np.zeros((img.shape[0], 176, 800))
    top_padding = np.zeros((img.shape[0], 88, 800))
    return np.concatenate([bottom_padding, img, top_padding], axis=1)

cap = cv2.VideoCapture(fn)
ret, frame = cap.read()
ret = True
frames = []

with tf.gfile.GFile('frozen_checkpoints/2ASPP_3C_sat_3v1.pb', 'rb') as f:
    graph_def_optimized = tf.GraphDef()
    graph_def_optimized.ParseFromString(f.read())

G = tf.Graph()

answer_key = {}
cnt = 1

with tf.Session(graph=G) as sess:
    preds = tf.import_graph_def(graph_def_optimized, return_elements=['final_softmax:0'])
    preds_reshaped = tf.image.resize_bilinear(preds[0], [336, 800])
    x = G.get_tensor_by_name('import/net_input:0')
    train_mode = G.get_tensor_by_name("import/final_train_mode:0")
    drop_rate = G.get_tensor_by_name("import/final_drop_rate:0")
    
    while ret:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[176:512, :, :]
        image = cv2.resize(image, (400, 168))
        ret,frame = cap.read()
        
        feed_dict ={x: np.expand_dims(image,axis=0),
                    train_mode: False, 
                    drop_rate: 0.}
        out = sess.run(preds_reshaped, feed_dict)

        car_out = np.squeeze(pad_frame(out[:,:,:,2]), axis=0)
        road_out = np.squeeze(pad_frame(out[:,:,:,1]), axis=0)
        
        road_out = np.where(road_out>=0.41, 1, 0)
        
        car_out = np.where(car_out>=0.085, 1, 0).astype(np.float32)
        car_out = cv2.GaussianBlur(car_out, (5, 5), 0)
        car_out[car_out>0] = 1
        
        road_out[car_out==1] = 0
        
        car_out = car_out.astype('uint8')
        road_out = road_out.astype('uint8')
        
        answer_key[cnt] = [encode(car_out), encode(road_out)]
        cnt+=1
       
print(json.dumps(answer_key))