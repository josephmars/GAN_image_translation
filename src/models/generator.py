from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.activations import elu

def resnet_block(n_filters, input_layer):
    """
    ResNet block with two convolutional layers.
    """
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation(elu)(g)
    
    g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    
    g = Concatenate()([g, input_layer])
    return g

def downsampling(input_layer, k, size=(3,3), strides=(2,2)):
    """
    Downsampling block using Conv2D.
    """
    init = RandomNormal(stddev=0.02)
    g = Conv2D(k, size, strides=strides, padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation(elu)(g)
    return g

def upsampling(input_layer, k, size=(3,3), strides=(2,2)):
    """
    Upsampling block using Conv2DTranspose.
    """
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(k, size, strides=strides, padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation(elu)(g)
    return g

def generator(img_shape=(256,256,3), n_resnet=9):
    """
    Define the CycleGAN generator model.
    """
    init = RandomNormal(stddev=0.02)
    in_img = Input(shape=img_shape)
    
    # C7S1-64
    g = downsampling(in_img, 64, size=(7,7), strides=(1,1))
    # D128
    g = downsampling(g, 128)
    # D256
    g = downsampling(g, 256)
    
    # R256
    for _ in range(n_resnet):
        g = resnet_block(256, g)
    
    # U128
    g = upsampling(g, 128)
    # U64
    g = upsampling(g, 64)
    
    # C7S1-3
    g = Conv2DTranspose(3, (7,7), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_img = Activation('tanh')(g)
    
    model = Model(in_img, out_img)
    return model 