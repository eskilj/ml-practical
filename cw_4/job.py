from cw_4.layer import Conv2dLayer, PoolLayer, AffineLayer
from cw_4.model import Model
import cw_4.train as train_model
import tensorflow as tf

wd = 0.005

# ROUND 2
lrs = [0.002, 0.004]
epochs = 10

for lr in lrs:
    layers = [
        Conv2dLayer([3, 3, 3, 24], [24], 'conv_1', True),
        PoolLayer('pool_1'),
        Conv2dLayer([3, 3, 24, 24], [24], 'conv_2', True),
        PoolLayer('pool_2'),
        AffineLayer('fc_1', True, wd),
        AffineLayer('fc_2', True, wd),
        AffineLayer('output', final_layer=True)
    ]

    _mo = Model(
        'lr/conv2,fc2,bn,fs=3,nf=24,wd={},lr={}'.format(wd, lr),
        layers=layers,
        activation=tf.nn.elu,
        train_epochs=epochs,
        initial_lr=lr
    )
    train_model.train_graph(_mo)