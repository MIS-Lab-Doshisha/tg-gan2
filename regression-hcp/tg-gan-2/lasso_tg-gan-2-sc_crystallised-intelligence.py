import os
import sys
import pickle
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from nilearn.connectome import sym_matrix_to_vec

sys.path.append('.')
from utils.load_dataset import load_score, load_structural_connectivity_hcp

# ------ Load connectivity/target ------ #
# Setting
dropout = 0.1
alpha = 0.2
n_splits = 2

# Real
hcp_crystallised_intelligence_train_csv = '../../data_hcp/hcp-cognition/hcp_crystallised-intelligence/hcp_crystallised-intelligence_prediction-model_train_baseline.csv'
X_real = load_structural_connectivity_hcp(participants_csv=hcp_crystallised_intelligence_train_csv,
                                     data_dir='../../data_hcp/structural_connectivity_360')
X_real = sym_matrix_to_vec(X_real, discard_diagonal=True)
y_real = load_score(participants_csv=hcp_crystallised_intelligence_train_csv, target='crystallised intelligence')


# Fake
synthesized_connectivity_pkl = f'../../data_hcp/tg-gan-2-sc_hcp_crystallised-intelligence/dr_{dropout}_alpha_{alpha}/tg-gan-2-sc_hcp_crystallised-intelligence_n-splits_{n_splits}_matrices.pkl'
synthesized_target_pkl = f'../../data_hcp/tg-gan-2-sc_hcp_crystallised-intelligence/dr_{dropout}_alpha_{alpha}/tg-gan-2-sc_hcp_crystallised-intelligence_n-splits_{n_splits}_targets.pkl'

with open(synthesized_connectivity_pkl, mode='rb') as f:
    X_fake = pickle.load(f)

with open(synthesized_target_pkl, mode='rb') as f:
    y_fake = pickle.load(f)

# Concatenate
X_fake = sym_matrix_to_vec(X_fake, discard_diagonal=True)
X = np.concatenate([X_real, X_fake], axis=0)
y = np.concatenate([y_real, y_fake])

print('X: ', X.shape)
print('y: ', y.shape)

# ------- Regression ------ #
# Hyperparameter setting
params = [2 ** i for i in range(-10, 6)]

# Nested cross validation
inner_mae_inv_list = []
inner_corr_list = []
inner_mae_inv_mean_list = []
inner_corr_mean_list = []
inner_eval_list = []

opt_params_list = []
outer_y_true = []
outer_y_pred = []
outer_y_train = []
outer_eval = []
outer_model = []

# Configure the cross-validation procedure for repeated cv
n_outer_splits = 5
n_inner_splits = 5
n_repetitions = 20
rkf = RepeatedKFold(n_splits=n_outer_splits, n_repeats=n_repetitions, random_state=42)
for i, (train_idx, valid_idx) in enumerate(rkf.split(X=X, y=y)):

    print('outer-fold: ', i % n_outer_splits + 1)
    print('repetition: ', i // n_outer_splits + 1)
    # print('train_idx:', train_idx)
    # print('valid_idx:', valid_idx)

    # Split data for outer cv
    X_train, X_valid = X[train_idx, :], X[valid_idx, :]
    y_train, y_valid = y[train_idx], y[valid_idx]

    print('X_train: ', X_train.shape)
    print('X_valid: ', X_valid.shape)
    print('y_train: ', y_train.shape)
    print('y_valid: ', y_valid.shape)

    # Configure the cross-validation procedure for inner cv
    cv_inner = KFold(n_splits=n_inner_splits, shuffle=True, random_state=42)
    inner_MAE_inv = np.zeros((n_inner_splits, len(params)))
    inner_Corr = np.zeros((n_inner_splits, len(params)))
    for j, param in enumerate(params):
        print('Hyperparamter: ', j + 1)
        for k, (in_train_ix, in_valid_ix) in enumerate(cv_inner.split(X=X_train, y=y_train)):
            print('Inner: ', k + 1)

            # Split data for inner cv
            in_X_train, in_X_valid = X_train[in_train_ix, :], X_train[in_valid_ix, :]
            in_y_train, in_y_valid = y_train[in_train_ix], y_train[in_valid_ix]

            print('in_X_train: ', in_X_train.shape)
            print('in_X_valid: ', in_X_valid.shape)
            print('in_y_train: ', in_y_train.shape)
            print('in_y_valid: ', in_y_valid.shape)

            # Training
            model = Lasso(alpha=param, fit_intercept=True)
            model.fit(X=in_X_train, y=in_y_train)

            # Validation
            in_y_pred = model.predict(in_X_valid)
            mae = mean_absolute_error(in_y_valid, in_y_pred)
            mae_inv = 1 / mae
            rval, _ = pearsonr(in_y_valid, in_y_pred)

            print('MAE: ', mae)
            print('Corr: ', rval)

            inner_MAE_inv[k, j] = mae_inv
            inner_Corr[k, j] = rval

            print('inner_MAE_inv: ', inner_MAE_inv)
            print('inner_Corr: ', inner_Corr)

    # Calculate inner evaluation score
    # Mean squared error
    inner_MAE_inv_mean = np.mean(inner_MAE_inv, axis=0)
    inner_MAE_inv_mean = (inner_MAE_inv_mean - np.mean(inner_MAE_inv_mean)) / np.std(inner_MAE_inv_mean)

    # Pearson correlation
    inner_Corr_mean = np.mean(inner_Corr, axis=0)
    inner_Corr_mean = (inner_Corr_mean - np.mean(inner_Corr_mean)) / np.std(inner_Corr_mean)

    # Inner score
    inner_score = inner_MAE_inv_mean + inner_Corr_mean

    # Hyperparameter selection
    opt_param_idx = np.argmax(inner_score)
    opt_param = params[opt_param_idx]
    opt_params_list.append(opt_param)
    print('opt_param_idx: ', opt_param_idx)
    print('opt_param: ', opt_param)

    # Get the best performing model and fit it on the whole training set
    best_model = Lasso(alpha=opt_param, fit_intercept=True)
    best_model.fit(X=X_train, y=y_train)

    y_train_pred = best_model.predict(X=X_train)
    y_valid_pred = best_model.predict(X=X_valid)

    # Evaluate the model on the validation dataset
    outer_rval, outer_pval = pearsonr(y_valid, y_valid_pred)
    outer_eval.append([outer_rval, outer_pval])

    # Save
    inner_mae_inv_list.append(inner_MAE_inv)
    inner_corr_list.append(inner_Corr)
    inner_mae_inv_mean_list.append(inner_MAE_inv_mean)
    inner_corr_mean_list.append(inner_Corr_mean)
    inner_eval_list.append(inner_score)

    outer_y_true.append(y_valid)
    outer_y_pred.append(y_valid_pred)
    outer_y_train.append([y_train, y_train_pred])
    outer_model.append(best_model)

outer_data = {
    'opt_params': opt_params_list,
    'outer_eval': outer_eval,
    'outer_y_true': outer_y_true,
    'outer_y_pred': outer_y_pred,
    'outer_y_train': outer_y_train,
    'outer_model': outer_model
}

inner_data = {
    'inner_mae_inv': inner_mae_inv_list,
    'inner_corr': inner_corr_list,
    'inner_mae_inv_mean': inner_mae_inv_mean_list,
    'inner_corr_mean': inner_corr_mean_list,
    'inner_eval': inner_eval_list
}

# Save as pickle
result_dir = f'./data/tg-gan-2-sc_hcp_crystallised-intelligence/lasso/n_splits_{n_splits}/outer_{n_outer_splits}_inner_{n_inner_splits}'
os.makedirs(result_dir, exist_ok=True)

inner_data_pkl = os.path.join(result_dir, 'inner.pkl')
outer_data_pkl = os.path.join(result_dir, 'outer.pkl')

with open(inner_data_pkl, mode='wb') as f:
    pickle.dump(inner_data, f)

with open(outer_data_pkl, mode='wb') as f:
    pickle.dump(outer_data, f)

# Save as text
outer_data_txt = os.path.join(result_dir, 'outer_score.txt')

with open(outer_data_txt, mode='w') as f:
    for k, (evaluation, params) in enumerate(zip(outer_eval, opt_params_list)):
        f.writelines(f'Outer {k + 1}-fold\n')
        f.writelines(f'selected hyperparameter [alpha]: {params}\n')
        f.writelines(f'pearson correlation: {evaluation}\n\n')
