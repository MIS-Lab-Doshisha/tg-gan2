# Task-guided GAN II (TG GAN II)

This repository contains the code for our paper:
**Task-guided Generative Adversarial Networks for Synthesizing and Augmenting Structural Connectivity Matrices for Connectivity-Based Prediction**

📝 Paper: [bioRxiv link](https://www.biorxiv.org/content/10.1101/2024.02.13.580039v1)

**Task-guided GAN II (TG GAN II)** is an advanced data augmentation method that utilizes generative adversarial networks (GANs) to enhance sample sizes in limited datasets for connectome-based prediction tasks.

This code has been tested with **Python 3.9.13**.

---

## How to Train the Synthesis Model (TG GAN II / WGAN-GP)

### 1. Install Dependencies

Install the required Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset for Augmentation

#### 2.1 Structural Connectivity Matrices

- Prepare the structural connectivity matrices that will be augmented and store them in the **`data/`** directory.
- This project uses `.mat` files to store structural connectivity matrices. Example data can be found in the **`data/`** directory.

#### 2.2 Objective Variables

- Prepare the objective variables (targets) to be predicted and store them in the **`data/`** directory as well.
- The objective variables should be formatted as shown below:

    | Index | Participant ID | NIH Toolbox Score |
    |---|---|---|
    | 1 | sub-ON41606 | 332 |
    | 2 | sub-ON89475 | 340 |
    | ... | ... | ... |

---

### 3. Train TG GAN II / WGAN-GP

#### 3.1 Create a PyTorch-Compatible Dataset

- The synthesis models in this project are implemented using **PyTorch**, so the dataset (structural connectivity matrices and objective variables) must be converted into a **PyTorch-compatible format**.
- The functions for creating a PyTorch-compatible dataset are defined in **`utils/load_dataset.py`**, based on the data structure in the **`data/`** directory.

📌 Reference: [PyTorch Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

#### 3.2 Define Training Parameters

- TG GAN II and WGAN-GP require several parameters for training. These should be set **before starting the training process**.
- The main training scripts are:
  - **TG GAN II**: `tg-gan-2/tg-gan-2_sc.py`
  - **WGAN-GP**: `wgan-gp_sc.py`

##### Common Parameters

| Parameter | Description |
|---|---|
| `data_dir` | Path to the dataset directory containing structural connectivity matrices and objective variables. |
| `n_epochs` | Maximum number of training epochs. |
| `n_critics` | Number of critic training steps per generator update. |
| `dropout_rate` | List of dropout rates for critics, generator, and regressor (only in TG GAN II). |
| `n_features` | Number of latent space dimensions. |
| `n_regions` | Number of brain regions (i.e., rows/columns in structural connectivity matrices). |

Additionally, the following optional parameters can be tuned:

| Parameter | Description |
|---|---|
| `alpha` | Learning rate for the critics, generator, and regressor. |
| `betas` | Beta parameters for the Adam optimizer. |
| `start` | Minimum number of epochs to be trained. |
| `patience` | Number of epochs to wait before stopping training if the loss does not improve. |

##### TG GAN II-Specific Parameters

| Parameter | Description |
|---|---|
| `alphas` | List of alpha values used for balancing the WGAN-GP loss and regressor loss. |

#### 3.3 Run the Training Script

Once the parameters are correctly defined, run the training script:

```bash
python tg-gan-2/tg-gan-2_sc.py
python wgan/wgan-gp_sc.py
```

The training process will generate the following output files:

| Output File | Description |
|---|---|
| `loss_{n_epoch}.pkl` | Loss values recorded up to *n_epoch*. |
| `checkpoint_encoder_epoch_{n_epoch}.pth` | Encoder model checkpoint at *n_epoch*. |
| `checkpoint_decoder_epoch_{n_epoch}.pth` | Decoder model checkpoint at *n_epoch*. |
| `checkpoint_critics_epoch_{n_epoch}.pth` | Critics model checkpoint at *n_epoch*. |
| `checkpoint_regressor_epoch_{n_epoch}.pth` | Regressor model checkpoint at *n_epoch*. |
| `loss.pkl` | Loss values for all epochs. |
| `checkpoint_encoder.pth` | Final trained encoder model. |
| `checkpoint_decoder.pth` | Final trained decoder model. |
| `checkpoint_critics.pth` | Final trained critics model. |
| `checkpoint_regressor.pth` | Final trained regressor model. |

---

## How to Evaluate Synthesis Model Performance

After the synthesis models have been trained, their performance should be evaluated. Evaluation scripts can be found in the **`tg-gan-2`**, **`wgan`**, and **`graph`** directories.

### 1. Evaluate the Latent Space

Use the scripts **`tg-gan-2/tg-gan-2-sc_latent-space.py`** and **`wgan/wgan-gp-sc_latent-space.py`** for latent space analysis:

- **Latent Space Visualization**: The latent space generated by TG GAN II and WGAN-GP encoders can be visualized using Principal Component Analysis (PCA).
- **Correlation Analysis Between Latent Space and Objective Scores**: The correlation between the principal component scores of the latent space and objective scores can be analyzed using Pearson’s correlation metric.

### 2. Evaluate Graph Theory Metrics

The structural connectivity matrices synthesized by TG GAN II and WGAN-GP can be analyzed using graph theory metrics. Scripts for this evaluation can be found in the **`graph`** directory.

#### 2.1 Similarity Between Acquired and Synthesized Matrices on the Average Matrix
- Compute the average acquired and synthesized matrices.
- Binarize the average matrices based on specified edge densities.
- Calculate graph metrics (edge strength, betweenness centrality, and clustering coefficient).
- Compute the similarity between acquired and synthesized matrices using KL divergence.
- Script: **`graph/graph-metrics_tg-gan-2-sc/wgan-gp-sc.py`**

#### 2.2 Similarity Between Acquired and Synthesized Matrices on Each Matrix
- Compute graph metrics (betweenness centrality, clustering coefficient, modularity, local efficiency, and global efficiency) for each matrix.
- Average graph metrics on a per-matrix basis.
- Compute similarity between acquired and synthesized matrices using Cohen’s d.
- Script: **`graph/graph-metrics_tg-gan-2-sc/wgan-gp-sc_for_each_matrix.py`**

For more details, refer to the scripts and [bctpy documentation](https://github.com/aestrivex/bctpy).