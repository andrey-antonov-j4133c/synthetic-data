"""
Modified version of TabFairGAN method
original can be viewed at:
https://github.com/amirarsalan90/TabFairGAN
"""
from collections import OrderedDict

import pandas as pd
import numpy as np
import torch

import sk2torch
from sklearn.linear_model import SGDClassifier

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as f
from torch import nn
from torchmetrics import F1Score, Accuracy, Recall

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier

DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")
DISPLAY_STEP = 50

CLASSIFICATION_METHODS = [
    (RandomForestClassifier, 'Random Forest', {'max_depth': 3})
]

REGRESSION_METHODS = [
    (RandomForestRegressor, 'Random Forest', {'max_depth': 3})
]


def get_ohe_data(df):
    df_int = df.select_dtypes(['float', 'integer']).values
    continuous_columns_list = list(df.select_dtypes(['float', 'integer']).columns)
    ##############################################################
    scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
    df_int = scaler.fit_transform(df_int)

    df_cat = df.select_dtypes('object')
    df_cat_names = list(df.select_dtypes('object').columns)
    numerical_array = df_int
    ohe = OneHotEncoder()
    ohe_array = ohe.fit_transform(df_cat)

    cat_lens = [i.shape[0] for i in ohe.categories_]
    discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))

    final_array = np.hstack((numerical_array, ohe_array.toarray()))
    return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array


def get_original_data(df_transformed, df_num_cols, df_obj_cols, df_num_cols_shape, ohe, scaler):
    df_ohe_int = df_transformed[:, :df_num_cols_shape[1]]
    df_ohe_int = scaler.inverse_transform(df_ohe_int)
    df_ohe_cats = df_transformed[:, df_num_cols_shape[1]:]
    if df_ohe_cats is not None:
        df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
    df_int = pd.DataFrame(df_ohe_int, columns=df_num_cols)
    if df_ohe_cats is not None:
        df_cat = pd.DataFrame(df_ohe_cats, columns=df_obj_cols)
        return pd.concat([df_int, df_cat], axis=1)
    else:
        return df_int


def prepare_data(df, batch_size):
    ohe, scaler, discrete_columns, continuous_columns, df_transformed = get_ohe_data(df)
    input_dim = df_transformed.shape[1]
    X_train, X_test = train_test_split(df_transformed, test_size=0.1, shuffle=True)

    data_train = X_train.copy()
    data_test = X_test.copy()

    data = torch.from_numpy(data_train).float()

    train_ds = TensorDataset(data)
    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    return ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test


class Generator(nn.Module):
    def __init__(self, input_dim, continuous_columns, discrete_columns):
        super(Generator, self).__init__()
        self._input_dim = input_dim
        self._discrete_columns = discrete_columns
        self._num_continuous_columns = len(continuous_columns)

        self.lin1 = nn.Linear(self._input_dim, self._input_dim)
        self.lin_numerical = nn.Linear(self._input_dim, self._num_continuous_columns)

        self.lin_cat = nn.ModuleDict()
        for key, value in self._discrete_columns.items():
            self.lin_cat[key] = nn.Linear(self._input_dim, value)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        # x = f.leaky_relu(self.lin1(x))
        # x_numerical = f.leaky_relu(self.lin_numerical(x))
        x_numerical = f.relu(self.lin_numerical(x))
        x_cat = []
        for key in self.lin_cat:
            x_cat.append(f.gumbel_softmax(self.lin_cat[key](x), tau=0.2))
        x_final = torch.cat((x_numerical, *x_cat), 1)
        return x_final


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self._input_dim = input_dim
        # self.dense1 = nn.Linear(109, 256)
        self.dense1 = nn.Linear(self._input_dim, self._input_dim)
        self.dense2 = nn.Linear(self._input_dim, self._input_dim)
        # self.dense3 = nn.Linear(256, 1)
        # self.drop = nn.Dropout(p=0.2)
        # self.activation = nn.Sigmoid()

    def forward(self, x):
        x = f.leaky_relu(self.dense1(x))
        # x = self.drop(x)
        # x = f.leaky_relu(self.dense2(x))
        x = f.leaky_relu(self.dense2(x))
        # x = self.drop(x)
        return x


class FairLossFunc(nn.Module):
    def __init__(self, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index):
        super(FairLossFunc, self).__init__()
        self._S_start_index = S_start_index
        self._Y_start_index = Y_start_index
        self._underpriv_index = underpriv_index
        self._priv_index = priv_index
        self._undesire_index = undesire_index
        self._desire_index = desire_index

    def forward(self, x, crit_fake_pred, lamda):
        G = x[:, self._S_start_index:self._S_start_index + 2]
        # print(x[0,64])
        I = x[:, self._Y_start_index:self._Y_start_index + 2]
        # disp = (torch.mean(G[:,1]*I[:,1])/(x[:,65].sum())) - (torch.mean(G[:,0]*I[:,0])/(x[:,64].sum()))
        # disp = -1.0 * torch.tanh(torch.mean(G[:,0]*I[:,1])/(x[:,64].sum()) - torch.mean(G[:,1]*I[:,1])/(x[:,65].sum()))
        # gen_loss = -1.0 * torch.mean(crit_fake_pred)
        disp = -1.0 * lamda * (torch.mean(G[:, self._underpriv_index] * I[:, self._desire_index]) / (
            x[:, self._S_start_index + self._underpriv_index].sum()) - torch.mean(
            G[:, self._priv_index] * I[:, self._desire_index]) / (
                                   x[:, self._S_start_index + self._priv_index].sum())) - 1.0 * torch.mean(
            crit_fake_pred)
        # print(disp)
        return disp


def get_gradient(crit, real, fake, epsilon):
    mixed_data = real * epsilon + fake * (1 - epsilon)

    mixed_scores = crit(mixed_data)

    gradient = torch.autograd.grad(
        inputs=mixed_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)

    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp

    return crit_loss


def get_consistency_model(train_data, target_index):
    x_train = np.delete(train_data, target_index, 1)
    y_train = train_data[:, target_index]
    consistency_model = GradientBoostingClassifier(
        loss='exponential',
        n_estimators=100,
        learning_rate=1.0,
        max_depth=1
    )
    consistency_model.fit(x_train, y_train)
    consistency_model = sk2torch.wrap(consistency_model)
    return consistency_model


def get_consistency_penalty(data, generated, consistency_metrics, model, target_index, input_dim):
    non_target = [i for i in range(input_dim) if i != target_index]

    x_true = data[:, non_target]
    y_true = data[:, target_index]
    x_fake = generated[:, non_target]
    y_fake = generated[:, target_index]

    pred_true = model.predict(x_true.double())
    pred_fake = model.predict(x_fake.double())

    result = torch.tensor(0)
    for metric in consistency_metrics:
        res = metric(pred_true.int(), y_true.int()) - metric(pred_fake.int(), y_fake.int())
        torch.add(result, torch.abs(res))
    return result


def train(df, epochs=500, batch_size=64, fair_epochs=10, lamda=0.5, target_col=None):
    ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test \
        = prepare_data(df, batch_size)

    # if target_col exists, count consistency
    consistency_model = None
    target_index = None
    consistency_metrics = None
    if target_col:
        target_index = df.columns.get_loc(target_col)
        num_classes = len(np.unique(data_train[:, target_index]))
        consistency_metrics = [
            F1Score(num_classes=num_classes),
        ]
        consistency_model = get_consistency_model(data_train, target_index)

    generator = Generator(input_dim, continuous_columns, discrete_columns).to(DEVICE)
    critic = Critic(input_dim).to(DEVICE)
    # second_critic = FairLossFunc(S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index).to(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    gen_optimizer_fair = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    crit_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # loss = nn.BCELoss()
    critic_losses = []
    generator_losses = []
    cur_step = 0
    for i in range(epochs):
        # j = 0
        print("epoch {}".format(i + 1))
        ############################
        if i + 1 <= (epochs - fair_epochs):
            print("training for accuracy")
        if i + 1 > (epochs - fair_epochs):
            print("training for fairness")
        for data in train_dl:
            data[0] = data[0].to(DEVICE)
            # j += 1
            loss_of_epoch_G = 0
            loss_of_epoch_D = 0
            crit_repeat = 4
            mean_iteration_critic_loss = 0
            for k in range(crit_repeat):
                # training the critic
                crit_optimizer.zero_grad()
                fake_noise = torch.randn(size=(batch_size, input_dim), device=DEVICE).float()
                fake = generator(fake_noise)

                crit_fake_pred = critic(fake.detach())
                crit_real_pred = critic(data[0])

                epsilon = torch.rand(batch_size, input_dim, device=DEVICE, requires_grad=True)
                gradient = get_gradient(critic, data[0], fake.detach(), epsilon)
                gp = gradient_penalty(gradient)

                if target_index:
                    consistency_penalty = get_consistency_penalty(
                        data[0],
                        fake,
                        consistency_metrics,
                        consistency_model,
                        target_index,
                        input_dim
                    )
                    torch.add(gp, consistency_penalty*2)

                crit_loss = get_crit_loss(
                    crit_fake_pred,
                    crit_real_pred,
                    gp,
                    c_lambda=10
                )

                mean_iteration_critic_loss += crit_loss.item() / crit_repeat
                crit_loss.backward(retain_graph=True)
                crit_optimizer.step()
            #############################
            if cur_step > 50:
                critic_losses += [mean_iteration_critic_loss]

            #############################
            if i + 1 <= (epochs - fair_epochs):
                # training the generator for accuracy
                gen_optimizer.zero_grad()
                fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=DEVICE).float()
                fake_2 = generator(fake_noise_2)
                crit_fake_pred = critic(fake_2)

                gen_loss = get_gen_loss(crit_fake_pred)
                gen_loss.backward()

                # Update the weights
                gen_optimizer.step()

            ###############################
            if i + 1 > (epochs - fair_epochs):
                # training the generator for fairness
                gen_optimizer_fair.zero_grad()
                fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=DEVICE).float()
                fake_2 = generator(fake_noise_2)

                crit_fake_pred = critic(fake_2)

                gen_fair_loss = second_critic(fake_2, crit_fake_pred, lamda)
                gen_fair_loss.backward()
                gen_optimizer_fair.step()
            """
            # Keep track of the average generator loss
            #################################
            if cur_step > 50:
                if i + 1 <= (epochs - fair_epochs):
                    generator_losses += [gen_loss.item()]
                if i + 1 > (epochs - fair_epochs):
                    generator_losses += [gen_fair_loss.item()]

                    # print("cr step: {}".format(cur_step))
            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                crit_mean = sum(critic_losses[-display_step:]) / display_step
                print("Step {}: Generator loss: {}, critic loss: {}".format(cur_step, gen_mean, crit_mean))
                step_bins = 20
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss"
                )
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Critic Loss"
                )
                plt.legend()
                plt.show()
                """
            cur_step += 1

    return generator, critic, ohe, scaler, data_train, data_test, input_dim
