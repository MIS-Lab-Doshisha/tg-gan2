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
algorithm = 'enet'

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

# tg-gan-2-sc
n_splits = [2, 3, 4, 5, 6]
tggan_disc_mse = []
tggan_test_mse = []
tggan_test_r = []
tggan_disc_mse_cleaned = []
tggan_test_mse_cleaned = []
tggan_test_r_cleaned = []
tggan_disc_mse_removed = []
tggan_test_mse_removed = []
tggan_test_r_removed = []
for i in n_splits:
    tggan_score_pkl = f'./data/tg-gan-2-sc_hcp_crystallised-intelligence/{algorithm}/n_splits_{i}/outer_{n_outer_splits}_inner_{n_inner_splits}/score.pkl'
    with open(tggan_score_pkl, mode='rb') as f:
        tggan_score = pickle.load(f)

    tggan_disc_mse.append(tggan_score['disc_mse'])
    tggan_test_mse.append(tggan_score['test_mse'])
    tggan_test_r.append(tggan_score['test_pearson_r'])

    _tggan_disc_mse_cleaned, _tggan_disc_mse_removed = remove_outliers_3sigma(tggan_score['disc_mse'])
    _tggan_test_mse_cleaned, _tggan_test_mse_removed = remove_outliers_3sigma(tggan_score['test_mse'])
    _tggan_test_r_cleaned, _tggan_test_r_removed = remove_outliers_3sigma(tggan_score['test_pearson_r'])

    tggan_disc_mse_cleaned.append(_tggan_disc_mse_cleaned), tggan_disc_mse_removed.append(_tggan_test_mse_removed)
    tggan_test_mse_cleaned.append(_tggan_test_mse_cleaned), tggan_test_mse_removed.append(_tggan_test_mse_removed)
    tggan_test_r_cleaned.append(_tggan_test_r_cleaned), tggan_test_r_removed.append(_tggan_test_r_removed)

    print('split: ', i)
    print('[test mse] average: ', np.mean(tggan_test_mse))
    print('[test r] average: ', np.mean(tggan_test_r))
    print('[test mse] std: ', np.std(tggan_test_mse))
    print('[test r] std: ', np.std(tggan_test_r))

# ------ Score plot ------ #
# tg-gan-2-sc
_nan_baseline_disc_mse_cleaned = [None] * (100 - len(baseline_disc_mse_cleaned))
_nan_baseline_disc_mse_removed = [None] * (100 - len(baseline_disc_mse_removed))
_baseline_disc_mse_cleaned = np.concatenate([baseline_disc_mse_cleaned, _nan_baseline_disc_mse_cleaned])
_baseline_disc_mse_removed = np.concatenate([baseline_disc_mse_removed, _nan_baseline_disc_mse_removed])

dict_disc_mse = {'baseline': baseline_disc_mse}
dict_disc_mse_cleaned = {'baseline': _baseline_disc_mse_cleaned}
dict_disc_mse_removed = {'baseline': _baseline_disc_mse_removed}
for i, j in enumerate(n_splits):
    _nan_disc_mse_cleaned = [None] * (100 - len(tggan_disc_mse_cleaned[i]))
    _nan_disc_mse_removed = [None] * (100 - len(tggan_disc_mse_removed[i]))
    _tggan_disc_mse_cleaned = np.concatenate([tggan_disc_mse_cleaned[i], _nan_disc_mse_cleaned])
    _tggan_disc_mse_removed = np.concatenate([tggan_disc_mse_removed[i], _nan_disc_mse_removed])

    dict_disc_mse[f'{j}'] = tggan_disc_mse[i]
    dict_disc_mse_cleaned[f'{j}'] = _tggan_disc_mse_cleaned
    dict_disc_mse_removed[f'{j}'] = _tggan_disc_mse_removed
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
    _nan_test_mse_cleaned = [None] * (100 - len(tggan_test_mse_cleaned[i]))
    _nan_test_mse_removed = [None] * (100 - len(tggan_test_mse_removed[i]))
    _tggan_test_mse_cleaned = np.concatenate([tggan_test_mse_cleaned[i], _nan_test_mse_cleaned])
    _tggan_test_mse_removed = np.concatenate([tggan_test_mse_removed[i], _nan_test_mse_removed])

    dict_test_mse[f'{j}'] = tggan_test_mse[i]
    dict_test_mse_cleaned[f'{j}'] = _tggan_test_mse_cleaned
    dict_test_mse_removed[f'{j}'] = _tggan_test_mse_removed
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
    _nan_test_r_cleaned = [None] * (100 - len(tggan_test_r_cleaned[i]))
    _nan_test_r_removed = [None] * (100 - len(tggan_test_r_removed[i]))
    _tggan_test_r_cleaned = np.concatenate([tggan_test_r_cleaned[i], _nan_test_r_cleaned])
    _tggan_test_r_removed = np.concatenate([tggan_test_r_removed[i], _nan_test_r_removed])

    dict_test_r[f'{j}'] = tggan_test_r[i]
    dict_test_r_cleaned[f'{j}'] = _tggan_test_r_cleaned
    dict_test_r_removed[f'{j}'] = _tggan_test_r_removed
test_r = pd.DataFrame(dict_test_r)
test_r_cleaned = pd.DataFrame(dict_test_r_cleaned)
test_r_removed = pd.DataFrame(dict_test_r_removed)

test_mse.to_csv('./tg-gan-2-sc_hcp_crystallised-intelligence_enet_test_mse.csv')
test_r.to_csv('./tg-gan-2-sc_hcp_crystallised-intelligence_enet_test_pearson-r.csv')

# Box plot
fig = plt.figure()
ax = fig.add_subplot()

color_pallete = ['#f2f0f7', '#cbc9e2', '#9e9ac8', '#756bb1', '#54278f', '#301934']
with sns.color_palette(color_pallete):
    ax = sns.boxplot(data=test_mse_cleaned, linewidth=1.5, whis=(0, 100))
    ax = sns.stripplot(data=test_mse_cleaned, edgecolor='#000000', linewidth=1.0, size=3)
    ax = sns.stripplot(data=test_mse_removed, edgecolor='#000000', linewidth=1.0, size=3, marker='o', color='#ED1A3D')
ax.set_xlabel('Sample size', fontsize=14)
ax.set_ylabel('RMSE', fontsize=14)
ax.set_ylim(25, 80)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

spines = 2.0
ax.spines["top"].set_linewidth(spines)
ax.spines["left"].set_linewidth(spines)
ax.spines["bottom"].set_linewidth(spines)
ax.spines["right"].set_linewidth(spines)

fig.savefig('./fig/tg-gan-2-sc_hcp_crystallised-intelligence_enet_test_rmse.pdf', transparent=True)
fig.savefig('./fig/tg-gan-2-sc_hcp_crystallised-intelligence_enet_test_rmse.png', dpi=700)

fig = plt.figure()
ax = fig.add_subplot()

color_pallete = ['#f2f0f7', '#cbc9e2', '#9e9ac8', '#756bb1', '#54278f', '#301934']
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

fig.savefig('./fig/tg-gan-2-sc_hcp_crystallised-intelligence_enet_test_r.pdf', transparent=True)
fig.savefig('./fig/tg-gan-2-sc_hco_crystallised-intelligence_enet_test_r.png', dpi=700)

# T-test [baseline < tg-gan-2-sc]
print('Pearson r')
p_val = []
for i, j in enumerate(n_splits):
    t, p = stats.ttest_ind(tggan_test_r_cleaned[i], baseline_test_r_cleaned, equal_var=False, alternative="greater")
    p_val.append(p)
    print('n_splits: ', j)
    print('t: ', t)
    print('p: ', p)

# T-test [baseline > tg-gan-2-sc]
print('RMSE')
p_val = []
for i, j in enumerate(n_splits):
    #t, p = stats.ttest_ind(baseline_test_mse_cleaned, tggan_test_mse_cleaned[i], equal_var=False, alternative="less")
    t, p = stats.ttest_ind(tggan_test_mse_cleaned[i], baseline_test_mse_cleaned, equal_var=False, alternative="less")

    p_val.append(p)
    print('n_splits: ', j)
    print('t: ', t)
    print('p: ', p)

plt.show()
