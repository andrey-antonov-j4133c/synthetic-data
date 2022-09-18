"""A wrapper module for TabFairGan repository"""

import pandas as pd
import torch

from methods.TabFairGAN.TabFairGAN_nofair import get_original_data, train_plot


def tab_fair_gan_no_fairness(
        df: pd.DataFrame,
        output_size=100
):
    """
    TabFairGan wrapper function
    Args:
        df: target dataframe
        output_size: size of the output df

    Returns:
        Fake data df of specified size
    """
    n_epochs = 20
    batch_size = 64

    device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")

    generator, critic, ohe, scaler, data_train, data_test, input_dim = train_plot(
        df,
        n_epochs,
        batch_size,
        0, 0
    )
    fake_numpy_array = generator(
        torch.randn(
            size=(output_size, input_dim),
            device=device)
    ).cpu().detach().numpy()
    fake_df = get_original_data(fake_numpy_array, df, ohe, scaler)
    fake_df = fake_df[df.columns]

    return fake_df
