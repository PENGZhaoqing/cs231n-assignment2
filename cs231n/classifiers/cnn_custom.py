import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, dropout=0, use_batchnorm=True, seed=None):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0

        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(64, 3, 3, 3)
        self.params['b1'] = np.zeros((1, 64))

        # self.params['gamma1'] = np.ones((100 * 31 * 31, 64))
        # self.params['beta1'] = np.zeros((100 * 31 * 31, 64))

        self.params['W2'] = weight_scale * np.random.randn(64, 64, 3, 3)
        self.params['b2'] = np.zeros((1, 64))

        self.params['W3'] = weight_scale * np.random.randn(32, 64, 3, 3)
        self.params['b3'] = np.zeros((1, 32))

        self.params['W4'] = weight_scale * np.random.randn(32, 32, 3, 3)
        self.params['b4'] = np.zeros((1, 32))
        #
        self.params['W5'] = weight_scale * np.random.randn(32 * 8 * 8, 512)
        self.params['b5'] = np.zeros((1, 512))
        #
        self.params['W6'] = weight_scale * np.random.randn(512, 64)
        self.params['b6'] = np.zeros((1, 64))

        self.params['W7'] = weight_scale * np.random.randn(64, num_classes)
        self.params['b7'] = np.zeros((1, num_classes))

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'}]

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode

        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        W6, b6 = self.params['W6'], self.params['b6']
        W7, b7 = self.params['W7'], self.params['b7']
        # gamma, beta = self.params['gamma1'], self.params['beta1']

        # conv1-relu1
        conv_out1, conv_cache1 = conv_forward_fast(X, W1, b1, {'stride': 1, 'pad': 1})
        # batchnorm_out1, batchnorm_cache1 = spatial_batchnorm_forward(conv_out1, gamma, beta, self.bn_params[0])
        relu_out1, relu_cache1 = relu_forward(conv_out1)

        # conv4-relu4-max_pool2
        conv_out2, conv_cache2 = conv_forward_fast(relu_out1, W2, b2, {'stride': 1, 'pad': 1})
        relu_out2, relu_cache2 = relu_forward(conv_out2)
        pool_out2, pool_cache2 = max_pool_forward_fast(relu_out2, {'pool_height': 2, 'pool_width': 2, 'stride': 2})

        # conv3-relu3
        conv_out3, conv_cache3 = conv_forward_fast(pool_out2, W3, b3, {'stride': 1, 'pad': 1})
        relu_out3, relu_cache3 = relu_forward(conv_out3)

        # conv4-relu4-max_pool4
        conv_out4, conv_cache4 = conv_forward_fast(relu_out3, W4, b4, {'stride': 1, 'pad': 1})
        relu_out4, relu_cache4 = relu_forward(conv_out4)
        pool_out4, pool_cache4 = max_pool_forward_fast(relu_out4, {'pool_height': 2, 'pool_width': 2, 'stride': 2})

        # fc5-relu5
        fc_out5, fc_cache5 = affine_forward(pool_out4, W5, b5)
        relu_out5, relu_cache5 = relu_forward(fc_out5)

        # fc5-relu5
        drop_out, drop_cache = dropout_forward(relu_out5, self.dropout_param)

        # fc6-relu6
        fc_out6, fc_cache6 = affine_forward(drop_out, W6, b6)
        relu_out6, relu_cache6 = relu_forward(fc_out6)

        # fc7
        scores, fc_cache7 = affine_forward(relu_out6, W7, b7)

        if y is None:
            return scores

        data_loss, d_scores = softmax_loss(scores, y)

        # fc7
        dx7, dW7, db7 = affine_backward(d_scores, fc_cache7)

        # fc6-relu6
        d_relu_out6 = relu_backward(dx7, relu_cache6)
        dx6, dW6, db6 = affine_backward(d_relu_out6, fc_cache6)

        # dropout
        dx_drop = dropout_backward(dx6, drop_cache)

        # fc5-relu5
        d_relu_out5 = relu_backward(dx_drop, relu_cache5)
        dx5, dW5, db5 = affine_backward(d_relu_out5, fc_cache5)

        # conv4-relu4-max_pool4
        d_pool_out4 = max_pool_backward_fast(dx5, pool_cache4)
        d_relu_out4 = relu_backward(d_pool_out4, relu_cache4)
        dx4, dW4, db4 = conv_backward_fast(d_relu_out4, conv_cache4)

        # conv3-relu3
        d_relu_out3 = relu_backward(dx4, relu_cache3)
        dx3, dW3, db3 = conv_backward_fast(d_relu_out3, conv_cache3)

        # conv2-relu2-pool2
        d_pool_out2 = max_pool_backward_fast(dx3, pool_cache2)
        d_relu_out2 = relu_backward(d_pool_out2, relu_cache2)
        dx2, dW2, db2 = conv_backward_fast(d_relu_out2, conv_cache2)

        # conv1-relu1
        d_relu_out1 = relu_backward(dx2, relu_cache1)
        # d_batchnorm_out1, d_gamma, d_beta = spatial_batchnorm_backward(d_relu_out1, batchnorm_cache1)
        dx1, dW1, db1 = conv_backward_fast(d_relu_out1, conv_cache1)

        # Add regularization
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        dW4 += self.reg * W4
        dW5 += self.reg * W5
        dW6 += self.reg * W6
        dW7 += self.reg * W7

        reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4, W5, W6, W7])

        loss = data_loss + reg_loss
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'W5': dW5,
                 'b5': db5, 'W6': dW6, 'b6': db6, 'W7': dW7, 'b7': db7}

        # , 'gamma1': d_gamma, 'beta1': d_beta

        return loss, grads
