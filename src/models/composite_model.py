from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def composite_model(g_1, d, g_2, img_shape=(256,256,3)):
    """
    Define the composite model for CycleGAN generator updates.
    """
    # Make sure only the generator is trainable
    g_1.trainable = True
    g_2.trainable = False
    d.trainable = False
    
    # Discriminator process
    input_gen = Input(shape=img_shape)
    gen1_out = g_1(input_gen)
    d_output = d(gen1_out)
    
    # Identity process
    input_id = Input(shape=img_shape)
    output_id = g_1(input_id)
    
    # Forward cycle
    output_f = g_2(gen1_out)
    
    # Backward cycle
    gen2_out = g_2(input_id)
    output_b = g_1(gen2_out)
    
    # Define model
    model = Model([input_gen, input_id], [d_output, output_id, output_f, output_b])
    
    # Compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(
        loss=['mse', 'mae', 'mae', 'mae'],
        loss_weights=[1, 5, 10, 10],
        optimizer=opt
    )
    
    return model 