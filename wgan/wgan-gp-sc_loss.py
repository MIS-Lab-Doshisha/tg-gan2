import sys
import glob
import pickle
import matplotlib.pyplot as plt

sys.path.append('.')
from utils.utils import natural_keys


# Visualize loss history of wgan-gp
# ------ Matplotlib ------ #
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ------ Setting ------ #
dropout= [0.5, 0.4, 0.3, 0.2, 0.1]

for j, dr in enumerate(dropout):

    loss_path = f'./data/wgan-gp-sc/dr_{dr}/loss*'
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
    # print('generator: ', generator_loss[-1])
    # print('critics: ', critics_loss[-1])
    # print('wasserstein: ', wasserstein_loss[-1])

    loss_txt = f'./data/wgan-gp-sc/loss.txt'
    with open(loss_txt, mode='a') as f:
        f.write(f'Dropout: {dr}\n')
        f.write(f'generator: {generator_loss[-51]}\n')
        f.write(f'critics: {critics_loss[-51]}\n')
        f.write(f'wasserstein: {wasserstein_loss[-51]}\n')

    # ------ Plotting ------ #
    fig = plt.figure(figsize=(18, 12))
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=0.5)

    ax1 = fig.add_subplot(3, 1, 1)
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

    ax2 = fig.add_subplot(3, 1, 2)
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

    ax3 = fig.add_subplot(3, 1, 3)
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

    fig.savefig(f'./fig/wgan-gp-sc/wgan-gp-sc_training_loss_dr_{dr}.pdf', transparent=True)
    fig.savefig(f'./fig/wgan-gp-sc/wgan-gp-2-sc_training_loss_dr_{dr}.png', dpi=700)

# plt.show()
