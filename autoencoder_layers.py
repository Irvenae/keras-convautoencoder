import theano
from keras import backend as K
from keras.layers import UpSampling2D
from keras.layers.core import Dense, Layer
from theano import tensor as T
import tensorflow as tf

class DePool2D(UpSampling2D):
    '''Simplar to UpSample, yet traverse only maxpooled elements

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.

    # Arguments
        size: tuple of 2 integers. The upsampling factors for rows and columns.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, pool2d_layer, *args, **kwargs):
        self._pool2d_layer = pool2d_layer
        super().__init__(*args, **kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.dim_ordering == 'th':
            output = K.repeat_elements(X, self.size[0], axis=2)
            output = K.repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = K.repeat_elements(X, self.size[0], axis=1)
            output = K.repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        f = T.grad(T.sum(self._pool2d_layer.get_output(train)), wrt=self._pool2d_layer.get_input(train)) * output

        return f

class DependentDense(Dense):
    def __init__(self, output_dim, master_layer, init='glorot_uniform', activation='linear', weights=None,
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, input_dim=None, **kwargs):
        self.master_layer = master_layer
        self.output_dim = output_dim
        super().__init__(output_dim, **kwargs)

    def build(self, args):
        self.kernel = tf.transpose(self.master_layer.kernel)
        self.bias = K.zeros((self.output_dim,))
        self.params = [self.bias]
        self.regularizers = []
        if self.kernel_regularizer:
            self.kernel_regularizer.set_param(self.kernel)
            self.regularizers.append(self.kernel_regularizer)

        if self.bias_regularizer:
            self.bias_regularizer.set_param(self.bias)
            self.regularizers.append(self.bias_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self._initial_weights is not None:
            self.set_weights(self._initial_weights)
            del self._initial_weights
