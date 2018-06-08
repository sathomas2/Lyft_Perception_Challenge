### Overview
For the Lyft Perception Challenge, I used a variation of Google's DeepLabV3 with ResNet-152 pretrained on ImageNet as the backbone. Below I will discuss my tactics. Unfortunately, I didn't have access to the CARLA simulator (mainly, I didn't want to rent out more GPU space for costs), so I coudln't create additional data. This was a big mistake I made early on. I should have collected more data. I believe that with extra data, I could have squeezed out the extra points to give me an average weighted F-score in the low 90s. Oh well. Live and learn. Next time...

The contents of my submission include:
```
workspace
|   README.md
|   train.py
|   inference.py
|   helper.py
|   get_pretrained_checkpoints.py
|   preinstall_script.sh
|   resnet.yml
|
|___data
|   |   ...
|
|___models
|   |   DeepLabV3_plus_2ASPP.py
|   |   resnet_utils.py
|   |   resnet_v2.py
|   |   resnet_v2_152.ckpt
|
|___checkpoints
|   |   ...
|
|___frozen_checkpoints
|   |   ...
```

To train a model on the included dataset or your own, run:
```
# From /home/workspace
python train.py \
    --dataset ${PATH_TO_DATASET} \
    --model DeepLabV3_plus-Res152 \
    --load_dir ${PATH_TO_PREVIOUS_CHKPT}  \
    --save_dir ${PATH_TO_SAVE_CHKPT}  \
    --batch_size 15 \
    --num_epochs 100 \
    --save_every 5 \
    --is_reduced True \
    --use_OLDsave False \
```

'is_reduced' flag is whether you will train on all 14 CARLA classes or only on road, vehicle, and void. 'use_OLDsave' flag is whether you will load a checkpoint trained on all 14 classes, but now only train on 3, as I find this pretraining helps the network learn, much like how using a backbone trained on ImageNet helps.

To save the checkpoints of a model into a frozen graph, run:
```
# From /home/workspace
python freeze_graph.py \
    --checkpoint ${PATH_TO_CHKPT} \
    --output_node final_softmax \
    --output_graph ${PATH_TO_SAVED_GRAPH}  \
```
'checkpoint' flag assumes that the checkpoint directory is within the checkpoints directory. 'output_graph' flag only takes the name of the file and will save it in the frozen_checkpoints directory.

To run inference on the grader, run:
```
# From /home/workspace
grader 'python inference.py'
```

### Model
After building ResNet-152 and loading the pretrained weights, I extract the second-to-last, third-to-last, and fourth-to-last pooling layers. The fourth-to-last pooling layer is run through 1x1 convolutional layer, with L2-weight reularization, followed by batch normalization, then a leaky ReLU activation and finally dropout. The third-to-last and second-to-last pooling layers are sent to Atrous Spatial Pyramid Pooling modules with dilation rates of 2,4,8, and 4,8,12 respectively. See models/DeepLabV3_plus_2ASPP.py lines 13-56. I used smaller dilation rates than in the paper because otherwise the dilation rates would have exceeded the boundaries of the input features due to image cropping. The two pooling layers are upsampled to the size of the fourth-to-last pooling layer and finally the three layers are concatenated. After which there is a series of 3x3 convolutions with L2-weight Regularization, batch normalization, leaky ReLUs and dropout, followed by a final upsampling to the input image size then final convolution layers to make binary, per-class predictions.

### Loss Function
To run the model precisely as I ran it, you must alter tf.losses.log_loss in your installation of tensorflow. The file can be found at tensorflow/python/ops/losses/losses_impl.py. The new log_loss function allows one to choose an offset_weight that is less than 1.0 to apply to negative or 0 labels in the ground truth so that False Negatives contribute more to the loss than False Positives. The altered log_loss function should look like:
```
def log_loss(labels, predictions, weights=1.0, epsilon=1e-7, 
             scope=None,softmax_like=False, offset_weight=1.,
             loss_collection=ops.GraphKeys.LOSSES,
             reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    if labels is None:      
        raise ValueError("labels must not be None.")
                                                                                        
    if predictions is None:
        raise ValueError("predictions must not be None.")
    with ops.name_scope(scope, "log_loss", (predictions, labels, weights)) as scope:
       predictions = math_ops.to_float(predictions)
       labels = math_ops.to_float(labels)
       predictions.get_shape().assert_is_compatible_with(labels.get_shape())
       
       if softmax_like:
            losses = -math_ops.multiply(labels, math_ops.log(predictions + epsilon))
       else:
            losses = -math_ops.multiply(labels, math_ops.log(predictions + epsilon)) - \
                                        math_ops.multiply(offset_weight, (math_ops.multiply( \
                                        (1 - labels), math_ops.log(1 - predictions + epsilon))))
                                                      
       return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)
```
In addition to the altered loss function, I also weighted each class such that loss per class before averaging was multiplied by 1/ClassFrequency, where the sum of all ClassFrequencies equaled one. Class frequencies were computed using the ground truth data. The weights along with offset_weight helped the network learn underrepresented classes like vehicles. The reason I chose log_loss over softmax_cross_entropy was precisely because of this fine-tuned controlling of the weights. Tensorflow's implementation of cross_entropy doesn't allow one to weight by class. I actually altered the class weights and the offset_weight as I finetuned to squeeze every last bit of performance out of my model, but generally an offset_weight of 0.75 works well.

### Data Augmentation
Data was randomly brightened, randomly cropped, randomly rotated, and randomly flipped. See helper.py for details. Also I searched every training image to find the max and min heights in which road or car appeared and only trained on those parts of the image to narrow the field and give more class balance to cars and less to void.

### Inference
During inference, rather than take the argmax from the output of the softmax layer to determine predictions, I use my own thresholding to control precision and recall more precisely. Also, I use a Gaussian Blur kernel on the cars because often the model has trouble locating the entire car and it seems to improve recall without a great expense to precision.
