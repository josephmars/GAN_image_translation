import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from src.utils import (
    generate_real_samples, 
    generate_fake_samples, 
    fake_pool, 
    save_models
)
import os

def train_model(d_M, d_PH, g_MtoPH, g_PHtoM, c_MtoPH, c_PHtoM, trainM, trainPH, epochs=2, batch_n=1, save_interval=300):
    """
    Train the CycleGAN models.
    """
    # Initialize loss history
    losses = pd.DataFrame(columns=[
        "step", 
        "g_lossPHtoM", 
        "dM_loss1", 
        "dM_loss2", 
        "g_lossMtoPH", 
        "dPH_loss1", 
        "dPH_loss2"
    ])
    
    patch_shape = d_M.output_shape[1]
    poolM, poolPH = [], []
    bat_per_epo = int(len(trainM) / batch_n)
    n_steps = bat_per_epo * epochs
    
    for step in range(n_steps):
        # Real samples
        X_realM, y_realM = generate_real_samples(trainM, batch_n, patch_shape)
        X_realPH, y_realPH = generate_real_samples(trainPH, batch_n, patch_shape)
        
        # Fake samples
        X_fakeM, y_fakeM = generate_fake_samples(g_PHtoM, X_realM, patch_shape)
        X_fakePH, y_fakePH = generate_fake_samples(g_MtoPH, X_realPH, patch_shape)
        
        # Update fake pools
        X_fakeM = fake_pool(poolM, X_fakeM, max_size=100)
        X_fakePH = fake_pool(poolPH, X_fakePH, max_size=100)
        
        # Update generators
        g_lossPHtoM, _, _, _, _ = c_PHtoM.train_on_batch([X_realPH, X_realM], [y_realM, X_realM, X_realPH, X_realM])
        g_lossMtoPH, _, _, _, _ = c_MtoPH.train_on_batch([X_realM, X_realPH], [y_realPH, X_realPH, X_realM, X_realPH])
        
        # Update discriminators
        dM_loss1 = d_M.train_on_batch(X_realM, y_realM)
        dM_loss2 = d_M.train_on_batch(X_fakeM, y_fakeM)
        
        dPH_loss1 = d_PH.train_on_batch(X_realPH, y_realPH)
        dPH_loss2 = d_PH.train_on_batch(X_fakePH, y_fakePH)
        
        # Record losses
        losses.loc[len(losses)] = [
            step, 
            g_lossPHtoM, 
            dM_loss1, 
            dM_loss2, 
            g_lossMtoPH, 
            dPH_loss1, 
            dPH_loss2
        ]
        
        # Print progress
        if (step + 1) % 50 == 0:
            print(f"Step {step+1}/{n_steps} | g_lossPHtoM: {g_lossPHtoM:.4f} | "
                  f"dM_loss: {dM_loss1 + dM_loss2:.4f} | "
                  f"g_lossMtoPH: {g_lossMtoPH:.4f} | dPH_loss: {dPH_loss1 + dPH_loss2:.4f}")
        
        # Save models at intervals
        if (step + 1) % save_interval == 0:
            save_models(step, g_MtoPH, g_PHtoM)
    
    # Save loss history
    losses.to_excel("src/losses_elu.xlsx", index=False)
    
    return losses 