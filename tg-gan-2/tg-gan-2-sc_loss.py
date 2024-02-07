import sys
import glob
import pickle
import matplotlib.pyplot as plt

sys.path.append('.')
from utils.utils import natural_keys


# Visualize loss history of tg-gan-2
# ------ Matplotlib ------ #
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ------ Setting ------ #
dropout = [0.5, 0.4, 0.3, 0.2, 0.1]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for i, a in enumerate(alphas):
    for j, dr in enumerate(dropout):

        loss_path = f'./data/tg-gan-sc/dr_{dr}_alpha_{a}/loss*'
        loss_pkls = glob.glob(loss_path)
        loss_pkls = sorted(loss_pkls, key=natural_keys)

        # ------- Load loss ------ #
        loss_pkl = loss_pkls[-1]
        print('loss_pkl: ', loss_pkl)
        with open(loss_pkl, mode='rb') as f:
            loss = pickle.load(f)

        generator_loss = loss['generator']
        critics_loss = loss['critics']
        wasserstein_loss = loss['wasserstein']
        regressor_train_loss = loss['regressor']
        regressor_valid_loss = loss['regressor_val']
        # print('generator: ', generator_loss[-1])
        # print('critics: ', critics_loss[-1])
        # print('wasserstein: ', wasserstein_loss[-1])
        # print('regressor [train]: ', regressor_train_loss[-1])
        # print('regressor [valid]: ', min(regressor_valid_loss))

        loss_txt = f'./data/tg-gan-sc/loss.txt'
        with open(loss_txt, mode='a') as f:
            f.write(f'Dropout: {dr}, alpha: {a}\n')
            f.write(f'generator: {generator_loss[-51]}\n')
            f.write(f'critics: {critics_loss[-51]}\n')
            f.write(f'wasserstein: {wasserstein_loss[-51]}\n')
            f.write(f'regressor [train]: {regressor_train_loss[-51]}\n')
            f.write(f'regressor [valid]: {regressor_valid_loss[-51]}\n')
            f.write(f'regressor [total]: {regressor_train_loss[-51]+regressor_valid_loss[-51]}\n\n')

        # ------ Plotting ------ #
        fig = plt.figure(figsize=(18, 12))
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=0.5)

        ax1 = fig.add_subplot(4, 1, 1)
        ax1.plot(generator_loss, c='blue', linewidth=2.0)
        ax1.set_ylim(-100, 150)
        ax1.set_xlabel('#Epoch', fontsize=18)
        ax1.set_ylabel('Generator', fontsize=18)
        ax1.grid(linestyle='dotted', linewidth=1.0)

        ax1.spines["top"].set_linewidth(2.0)
        ax1.spines["bottom"].set_linewidth(2.0)
        ax1.spines["right"].set_linewidth(2.0)
        ax1.spines["left"].set_linewidth(2.0)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        ax2 = fig.add_subplot(4, 1, 2)
        ax2.plot(critics_loss, c='red', linewidth=2.0)
        ax2.set_ylim(-100, 150)
        ax2.set_xlabel('#Epoch', fontsize=18)
        ax2.set_ylabel('Critics', fontsize=18)
        ax2.grid(linestyle='dotted', linewidth=1.0)

        ax2.spines["top"].set_linewidth(2.0)
        ax2.spines["bottom"].set_linewidth(2.0)
        ax2.spines["right"].set_linewidth(2.0)
        ax2.spines["left"].set_linewidth(2.0)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        ax3 = fig.add_subplot(4, 1, 3)
        ax3.plot(wasserstein_loss, c='green', linewidth=2.0)
        ax3.set_ylim(-100, 150)
        ax3.set_xlabel('#Epoch', fontsize=18)
        ax3.set_ylabel('Wasserstein loss', fontsize=18)
        ax3.grid(linestyle='dotted', linewidth=1.0)

        ax3.spines["top"].set_linewidth(2.0)
        ax3.spines["bottom"].set_linewidth(2.0)
        ax3.spines["right"].set_linewidth(2.0)
        ax3.spines["left"].set_linewidth(2.0)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        ax4 = fig.add_subplot(4, 1, 4)
        ax4.plot(regressor_train_loss, c='orange', label='training', linewidth=2.0)
        ax4.plot(regressor_valid_loss, c='purple', label='validation', linewidth=2.0)
        ax4.set_ylim(-10, 150)
        ax4.set_xlabel('#Epoch', fontsize=18)
        ax4.set_ylabel('Regressor loss', fontsize=18)
        ax4.grid(linestyle='dotted')
        ax4.legend(loc='upper right', fontsize=16)

        ax4.spines["top"].set_linewidth(2.0)
        ax4.spines["bottom"].set_linewidth(2.0)
        ax4.spines["right"].set_linewidth(2.0)
        ax4.spines["left"].set_linewidth(2.0)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        fig.savefig(f'./fig/tg-gan-2-sc/tg-gan-2-sc_training_loss_dr_{dr}_alpha_{a}.pdf', transparent=True)
        fig.savefig(f'./fig/tg-gan-2-sc/tg-gan-2-sc_training_loss_dr_{dr}_alpha_{a}.png', dpi=700)

# plt.show()
