import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

sys.path.append('.')
from utils.outlier_detection import remove_outliers_3sigma


# ------ Matplotlib ------- #
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ------ Setting ------ #
n_outer_splits = 5
n_inner_splits = 5
algorithm = 'ridge'

# ------ Load regression results ------ #
# baseline
baseline_score_pkl = f'../baseline/data/baseline-sc_crystallised-intelligence/{algorithm}/outer_{n_outer_splits}_inner_{n_inner_splits}/score.pkl'
with open(baseline_score_pkl, mode='rb') as f:
    baseline_score = pickle.load(f)

baseline_disc_mse = baseline_score['disc_mse']
baseline_test_mse = baseline_score['test_mse']
baseline_test_r = baseline_score['test_pearson_r']

baseline_disc_mse_cleaned, baseline_disc_mse_removed = remove_outliers_3sigma(baseline_disc_mse)
baseline_test_mse_cleaned, baseline_test_mse_removed = remove_outliers_3sigma(baseline_test_mse)
baseline_test_r_cleaned, baseline_test_r_removed = remove_outliers_3sigma(baseline_test_r)

print('[test mse] average: ', np.mean(baseline_test_mse))
print('[test r] average: ', np.mean(baseline_test_r))
print('[test mse] std: ', np.std(baseline_test_mse))
print('[test r] std: ', np.std(baseline_test_r))

# enhanced sample size
n_splits = [2, 3, 4, 5, 6]
b_train_disc_mse = []
b_train_test_mse = []
b_train_test_r = []
b_train_disc_mse_cleaned = []
b_train_test_mse_cleaned = []
b_train_test_r_cleaned = []
b_train_disc_mse_removed = []
b_train_test_mse_removed = []
b_train_test_r_removed = []
for i in n_splits:
    b_train_score_pkl = f'./data/baseline-sc_train_{i}_crystallised-intelligence/{algorithm}/outer_{n_outer_splits}_inner_{n_inner_splits}/score.pkl'
    with open(b_train_score_pkl, mode='rb') as f:
        b_train_score = pickle.load(f)

    b_train_disc_mse.append(b_train_score['disc_mse'])
    b_train_test_mse.append(b_train_score['test_mse'])
    b_train_test_r.append(b_train_score['test_pearson_r'])

    _b_train_disc_mse_cleaned, _b_train_disc_mse_removed = remove_outliers_3sigma(b_train_score['disc_mse'])
    _b_train_test_mse_cleaned, _b_train_test_mse_removed = remove_outliers_3sigma(b_train_score['test_mse'])
    _b_train_test_r_cleaned, _b_train_test_r_removed = remove_outliers_3sigma(b_train_score['test_pearson_r'])

    b_train_disc_mse_cleaned.append(_b_train_disc_mse_cleaned), b_train_disc_mse_removed.append(_b_train_test_mse_removed)
    b_train_test_mse_cleaned.append(_b_train_test_mse_cleaned), b_train_test_mse_removed.append(_b_train_test_mse_removed)
    b_train_test_r_cleaned.append(_b_train_test_r_cleaned), b_train_test_r_removed.append(_b_train_test_r_removed)

    print('split: ', i)
    print('[test mse] average: ', np.mean(b_train_test_mse))
    print('[test r] average: ', np.mean(b_train_test_r))
    print('[test mse] std: ', np.std(b_train_test_mse))
    print('[test r] std: ', np.std(b_train_test_r))

# ------ Score plot ------ #
# baseline_sc_crystallised-intelligence
_nan_baseline_disc_mse_cleaned = [None] * (100 - len(baseline_disc_mse_cleaned))
_nan_baseline_disc_mse_removed = [None] * (100 - len(baseline_disc_mse_removed))
_baseline_disc_mse_cleaned = np.concatenate([baseline_disc_mse_cleaned, _nan_baseline_disc_mse_cleaned])
_baseline_disc_mse_removed = np.concatenate([baseline_disc_mse_removed, _nan_baseline_disc_mse_removed])

dict_disc_mse = {'baseline': baseline_disc_mse}
dict_disc_mse_cleaned = {'baseline': _baseline_disc_mse_cleaned}
dict_disc_mse_removed = {'baseline': _baseline_disc_mse_removed}
for i, j in enumerate(n_splits):
    _nan_disc_mse_cleaned = [None] * (100 - len(b_train_disc_mse_cleaned[i]))
    _nan_disc_mse_removed = [None] * (100 - len(b_train_disc_mse_removed[i]))
    _b_train_disc_mse_cleaned = np.concatenate([b_train_disc_mse_cleaned[i], _nan_disc_mse_cleaned])
    _b_train_disc_mse_removed = np.concatenate([b_train_disc_mse_removed[i], _nan_disc_mse_removed])

    dict_disc_mse[f'{j}'] = b_train_disc_mse[i]
    dict_disc_mse_cleaned[f'{j}'] = _b_train_disc_mse_cleaned
    dict_disc_mse_removed[f'{j}'] = _b_train_disc_mse_removed
disc_mse = pd.DataFrame(dict_disc_mse)
disc_mse_cleaned = pd.DataFrame(dict_disc_mse_cleaned)
disc_mse_removed = pd.DataFrame(dict_disc_mse_removed)

_nan_baseline_test_mse_cleaned = [None] * (100 - len(baseline_test_mse_cleaned))
_nan_baseline_test_mse_removed = [None] * (100 - len(baseline_test_mse_removed))
_baseline_test_mse_cleaned = np.concatenate([baseline_test_mse_cleaned, _nan_baseline_test_mse_cleaned])
_baseline_test_mse_removed = np.concatenate([baseline_test_mse_removed, _nan_baseline_test_mse_removed])
dict_test_mse = {'baseline': baseline_test_mse}
dict_test_mse_cleaned = {'baseline': _baseline_test_mse_cleaned}
dict_test_mse_removed = {'baseline': _baseline_test_mse_removed}
for i, j in enumerate(n_splits):
    _nan_test_mse_cleaned = [None] * (100 - len(b_train_test_mse_cleaned[i]))
    _nan_test_mse_removed = [None] * (100 - len(b_train_test_mse_removed[i]))
    _b_train_test_mse_cleaned = np.concatenate([b_train_test_mse_cleaned[i], _nan_test_mse_cleaned])
    _b_train_test_mse_removed = np.concatenate([b_train_test_mse_removed[i], _nan_test_mse_removed])

    dict_test_mse[f'{j}'] = b_train_test_mse[i]
    dict_test_mse_cleaned[f'{j}'] = _b_train_test_mse_cleaned
    dict_test_mse_removed[f'{j}'] = _b_train_test_mse_removed
test_mse = pd.DataFrame(dict_test_mse)
test_mse_cleaned = pd.DataFrame(dict_test_mse_cleaned)
test_mse_removed = pd.DataFrame(dict_test_mse_removed)

_nan_baseline_test_r_cleaned = [None] * (100 - len(baseline_test_r_cleaned))
_nan_baseline_test_r_removed = [None] * (100 - len(baseline_test_r_removed))
_baseline_test_r_cleaned = np.concatenate([baseline_test_r_cleaned, _nan_baseline_test_r_cleaned])
_baseline_test_r_removed = np.concatenate([baseline_test_r_removed, _nan_baseline_test_r_removed])
dict_test_r = {'baseline': baseline_test_r}
dict_test_r_cleaned = {'baseline': _baseline_test_r_cleaned}
dict_test_r_removed = {'baseline': _baseline_test_r_removed}
for i, j in enumerate(n_splits):
    _nan_test_r_cleaned = [None] * (100 - len(b_train_test_r_cleaned[i]))
    _nan_test_r_removed = [None] * (100 - len(b_train_test_r_removed[i]))
    _b_train_test_r_cleaned = np.concatenate([b_train_test_r_cleaned[i], _nan_test_r_cleaned])
    _b_train_test_r_removed = np.concatenate([b_train_test_r_removed[i], _nan_test_r_removed])

    dict_test_r[f'{j}'] = b_train_test_r[i]
    dict_test_r_cleaned[f'{j}'] = _b_train_test_r_cleaned
    dict_test_r_removed[f'{j}'] = _b_train_test_r_removed
test_r = pd.DataFrame(dict_test_r)
test_r_cleaned = pd.DataFrame(dict_test_r_cleaned)
test_r_removed = pd.DataFrame(dict_test_r_removed)

test_mse.to_csv('./baseline_sc_crystallised-intelligence_test_mse.csv')
test_r.to_csv('./baseline_sc_crystallised-intelligence_test_pearson-r.csv')

# Box plot
fig = plt.figure()
ax = fig.add_subplot()

color_pallete = ['#d0dad0', '#a2b48f', '#518f6b', '#53685b', '#23583d', '#001307']
with sns.color_palette(color_pallete):
    ax = sns.boxplot(data=test_mse_cleaned, linewidth=1.5, whis=(0, 100))
    ax = sns.stripplot(data=test_mse_cleaned, edgecolor='#000000', linewidth=1.0, size=3)
    ax = sns.stripplot(data=test_mse_removed, edgecolor='#000000', linewidth=1.0, size=3, marker='o', color='#ED1A3D')
ax.set_xlabel('Sample size', fontsize=14)
ax.set_ylabel('RMSE', fontsize=14)
ax.set_ylim(10, 40)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

spines = 2.0
ax.spines["top"].set_linewidth(spines)
ax.spines["left"].set_linewidth(spines)
ax.spines["bottom"].set_linewidth(spines)
ax.spines["right"].set_linewidth(spines)

fig.savefig('./fig/baseline_sc_crystallised-intelligence_ridge_test_rmse.pdf', transparent=True)
fig.savefig('./fig/baseline_sc_crystallised-intelligence_ridge_test_rmse.png', dpi=700)

fig = plt.figure()
ax = fig.add_subplot()

color_pallete = ['#d0dad0', '#a2b48f', '#518f6b', '#53685b', '#23583d', '#001307']
with sns.color_palette(color_pallete):
    ax = sns.boxplot(data=test_r_cleaned, linewidth=1.5, whis=(0, 100))
    ax = sns.stripplot(data=test_r_cleaned, edgecolor='#000000', linewidth=1.0, size=3)
    ax = sns.stripplot(data=test_r_removed, edgecolor='#000000', linewidth=1.0, size=3, marker='o', color='#ED1A3D')
ax.set_ylim(-0.4, 0.7)
ax.set_xlabel('Sample size', fontsize=14)
ax.set_ylabel("Pearson's correlation", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

spines = 2.0
ax.spines["top"].set_linewidth(spines)
ax.spines["left"].set_linewidth(spines)
ax.spines["bottom"].set_linewidth(spines)
ax.spines["right"].set_linewidth(spines)

fig.savefig('./fig/baseline_sc_crystallised-intelligence_ridge_test_r.pdf', transparent=True)
fig.savefig('./fig/baseline_sc_crystallised-intelligence_ridge_test_r.png', dpi=700)

# T-test [baseline < baseline_sc_crystallised-intelligence]
print('Pearson r')
p_val = []
for i, j in enumerate(n_splits):
    t, p = stats.ttest_ind(b_train_test_r_cleaned[i], baseline_test_r_cleaned, equal_var=False, alternative="greater")
    p_val.append(p)
    print('n_splits: ', j)
    print('t: ', t)
    print('p: ', p)

# T-test [baseline > baseline_sc_crystallised-intelligence]
print('RMSE')
p_val = []
for i, j in enumerate(n_splits):
    t, p = stats.ttest_ind(baseline_test_mse_cleaned, b_train_test_mse_cleaned[i], equal_var=False, alternative="less")
    p_val.append(p)
    print('n_splits: ', j)
    print('t: ', t)
    print('p: ', p)

plt.show()
