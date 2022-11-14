from typing import List

import pandas
import torch

from ctgan import CTGANSynthesizer
from TabFairGAN_consistency.TabFairGAN_consistency import train, get_original_data


class Method:
    def __init__(
            self,
            df: pandas.DataFrame,
            cat_cols: List[str],
            target_col: str = None,
            epochs: int = 10
    ):
        self.name = ''
        self.target_col = target_col
        self.fit(df, epochs, cat_cols)

    def fit(self, df, epochs, cat_cols, **kwargs):
        pass

    def sample(self, num_samples: int) -> pandas.DataFrame:
        pass


class CTGAN(Method):
    def __init__(self, df: pandas.DataFrame, cat_cols: List[str], target_col: str = None, epochs: int = 10):
        self.ctgan = None
        super().__init__(df, cat_cols, target_col, epochs)
        self.name = 'CTGAN'

    def fit(self, df, epochs, cat_cols, **kwargs):
        self.ctgan = CTGANSynthesizer(epochs=epochs)
        self.ctgan.fit(df, cat_cols)

    def sample(self, num_samples: int) -> pandas.DataFrame:
        return self.ctgan.sample(num_samples)


class TabFiarGAN(Method):
    def __init__(self, df: pandas.DataFrame, cat_cols: List[str], target_col: str = None, epochs: int = 10):
        self.generator = None
        self.encoder = None
        self.scaler = None
        self.input_dim = None
        self.batch_size = 64
        self.columns = df.columns
        self.num_cols = df.select_dtypes(['float', 'integer']).columns
        self.num_cols_shape = df.select_dtypes(['float', 'integer']).shape
        self.obj_cols = df.select_dtypes('object').columns
        super().__init__(df, cat_cols, target_col, epochs)
        self.name = 'TabFairGan'

    def fit(self, df, epochs, cat_cols, **kwargs):
        self.generator, critic, self.encoder, self.scaler, _, _, self.input_dim = train(
            df,
            epochs,
            self.batch_size,
            fair_epochs=0,
            lamda=0.5,
        )

    def sample(self, num_samples: int) -> pandas.DataFrame:
        device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")
        fake_numpy_array = self.generator(
            torch.randn(
                size=(num_samples, self.input_dim),
                device=device)
        ).cpu().detach().numpy()
        fake_df = get_original_data(
            fake_numpy_array, self.num_cols, self.obj_cols, self.num_cols_shape, self.encoder, self.scaler
        )
        fake_df = fake_df[self.columns]
        return fake_df


class TabFiarGANConsistent(Method):
    def __init__(self, df: pandas.DataFrame, cat_cols: List[str], target_col: str = None, epochs: int = 10):
        self.generator = None
        self.encoder = None
        self.scaler = None
        self.input_dim = None
        self.batch_size = 64
        self.columns = df.columns
        self.num_cols = df.select_dtypes(['float', 'integer']).columns
        self.num_cols_shape = df.select_dtypes(['float', 'integer']).shape
        self.obj_cols = df.select_dtypes('object').columns
        super().__init__(df, cat_cols, target_col, epochs)
        self.name = 'TabFairGanConsistent'

    def fit(self, df, epochs, cat_cols, **kwargs):
        self.generator, critic, self.encoder, self.scaler, _, _, self.input_dim = train(
            df,
            epochs,
            self.batch_size,
            fair_epochs=0,
            lamda=0.5,
            target_col=self.target_col
        )

    def sample(self, num_samples: int) -> pandas.DataFrame:
        device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")
        fake_numpy_array = self.generator(
            torch.randn(
                size=(num_samples, self.input_dim),
                device=device)
        ).cpu().detach().numpy()
        fake_df = get_original_data(
            fake_numpy_array, self.num_cols, self.obj_cols, self.num_cols_shape, self.encoder, self.scaler
        )
        fake_df = fake_df[self.columns]
        return fake_df