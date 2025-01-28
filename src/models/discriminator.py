from tensorflow.keras.layers import Conv2D, Activation, Input
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.activations import elu

def CBR_block(d_in, k, instance_norm=True):
    """
    Convolution-BatchNorm-ReLU block.
    """
    init = RandomNormal(stddev=0.02)
    d = Conv2D(k, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d_in)
    if instance_norm:
        d = InstanceNormalization(axis=-1)(d)
    d = Activation(elu)(d)
    return d

def discriminator(img_shape=(256,256,3)):
    """
    Define the PatchGAN discriminator model.
    """
    init = RandomNormal(stddev=0.02)
    in_img = Input(shape=img_shape)

    # C64
    d = CBR_block(in_img, 64, instance_norm=False)
    # C128
    d = CBR_block(d, 128, instance_norm=True)
    # C256
    d = CBR_block(d, 256, instance_norm=True)
    # C512
    d = CBR_block(d, 512, instance_norm=True)
    # Layer before output
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = Activation(elu)(d)
    # Output layer
    patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)

    model = Model(in_img, patch_out)
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        loss_weights=[0.5]
    )
    return model 