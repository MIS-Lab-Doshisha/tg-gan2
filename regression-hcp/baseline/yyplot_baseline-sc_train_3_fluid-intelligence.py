import os
import sys
import pickle
import matplotlib.pyplot as plt
from nilearn.connectome import sym_matrix_to_vec
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

sys.path.append('.')
from utils.load_dataset import load_structural_connectivity_hcp, load_score

# ------ Matplotlib ------- #
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ------ Setting ------- #
algorithm = 'enet'
n_outer_splits = 5
n_inner_splits = 5
n_repetitions = 20


fig_dir = f'./fig/baseline-sc_train_3_fluid-intelligence/{algorithm}/'
os.makedirs(fig_dir, exist_ok=True)

# ------- Load dataset ------ #
hcp_fluid_intelligence_train_csv = '../../data_hcp/hcp-cognition/hcp_fluid-intelligence/hcp_fluid-intelligence_prediction-model_test.csv'
X = load_structural_connectivity_hcp(participants_csv=hcp_fluid_intelligence_train_csv,
                                     data_dir='../../data_hcp/structural_connectivity_360')
X = sym_matrix_to_vec(X, discard_diagonal=True)
y = load_score(participants_csv=hcp_fluid_intelligence_train_csv, target='fluid intelligence')

print('X: ', X.shape)
print('y: ', y.shape)

# ------ Load result ------ #
outer_data_pkl = f'./data/baseline-sc_train_3_fluid-intelligence/{algorithm}/outer_{n_outer_splits}_inner_{n_inner_splits}/outer.pkl'
with open(outer_data_pkl, mode='rb') as f:
    outer_data = pickle.load(f)

outer_y_true = outer_data['outer_y_true']
outer_y_pred = outer_data['outer_y_pred']
outer_y_train = outer_data['outer_y_train']
outer_reg_model = outer_data['outer_model']

# ------ [Discovery] Plotting ------ #
disc_pearson_r = []
disc_mse = []
for i in range(n_repetitions):
    fig = plt.figure(figsize=(15, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.9, bottom=0.1, top=0.9)
    for j in range(n_outer_splits):
        # Index
        idx = i * n_outer_splits + j

        # Metrics
        r, p = pearsonr(outer_y_true[idx], outer_y_pred[idx])
        mse = mean_squared_error(outer_y_true[idx], outer_y_pred[idx], squared=False)
        disc_pearson_r.append(r)
        disc_mse.append(mse)

        # Plotting
        ax = fig.add_subplot(2, 3, j + 1)
        # Scatter plot
        ax.scatter(outer_y_train[idx][0], outer_y_train[idx][0], c='#ffffff', edgecolors='#000000', s=20, label='train')
        ax.scatter(outer_y_true[idx], outer_y_pred[idx], marker='x', c='#ff0000', s=20, label='valid')

        # Diagonal Plot
        ax.plot(range(350, 650), range(350, 650), c='#a9a9a9', linestyle='dotted', label='obs = pred')

        # Setting
        ax.set_xlim(350, 650)
        ax.set_ylim(350, 650)
        ax.set_aspect('equal')
        ax.set_xlabel('Observed score')
        ax.set_ylabel('Predicted score')
        ax.grid(linestyle='dotted')
        ax.legend(loc='lower right')

    fig.savefig(f'{fig_dir}/disc_yyplot_reps_{i}.pdf', transparent=True)

# ------ [Test] Plotting ------ #
test_pearson_r = []
test_mse = []
for i in range(n_repetitions):
    fig = plt.figure(figsize=(15, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.9, bottom=0.1, top=0.9)
    for j in range(n_outer_splits):
        # Index
        idx = i * n_outer_splits + j

        # Regression
        model = outer_reg_model[idx]
        y_pred = model.predict(X)

        # Metrics
        r, p = pearsonr(y, y_pred)
        mse = mean_squared_error(y, y_pred, squared=False)
        test_pearson_r.append(r)
        test_mse.append(mse)

        # Plotting
        ax = fig.add_subplot(2, 3, j + 1)
        # Scatter plot
        ax.scatter(y, y_pred, marker='x', c='#ff0000', s=20)

        # Text plot
        ax.text(0.60, 0.05, f'r = {r:.2f}\np = {p:.2f}\nRMSE = {mse:.2f}', transform=ax.transAxes)

        # Diagonal Plot
        ax.plot(range(350, 650), range(350, 650), c='#a9a9a9', linestyle='dotted', label='obs = pred')

        # Setting
        ax.set_xlim(350, 650)
        ax.set_ylim(350, 650)
        ax.set_aspect('equal')
        ax.set_xlabel('Observed score')
        ax.set_ylabel('Predicted score')
        ax.grid(linestyle='dotted')

    fig.savefig(f'{fig_dir}/test_yyplot_reps_{i}.pdf', transparent=True)

# ------ Save ------ #
score = {
    'disc_pearson_r': disc_pearson_r,
    'disc_mse': disc_mse,
    'test_pearson_r': test_pearson_r,
    'test_mse': test_mse,
}

score_pkl = f'./data/baseline-sc_train_3_fluid-intelligence/{algorithm}/outer_{n_outer_splits}_inner_{n_inner_splits}/score.pkl'
with open(score_pkl, mode='wb') as f:
    pickle.dump(score, f)

# plt.show()
