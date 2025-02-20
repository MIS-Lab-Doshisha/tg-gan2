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
- The objective variables should be formatted in a table as shown below:  

    | Index | Participant ID | NIH Toolbox Score |
    |---|---|---|
    | 1 | sub-ON41606 | 332 |
    | 2 | sub-ON89475 | 340 |
    | ... | ... | ... |

---

### 3. Train TG GAN II / WGAN-GP  

#### 3.1 Create a PyTorch-Compatible Dataset  

- The synthesis models in this project are implemented using **PyTorch**, so the dataset (structural connectivity matrices and their objective variables) must be converted into a **PyTorch-compatible format**.  
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
| `start` | (Description needed) |
| `patience` | (Description needed) |

##### TG GAN II-Specific Parameters  

| Parameter | Description |
|---|---|
| `alphas` | List of alpha values used for (XXX - description needed). |

#### 3.3 Run the Training Script  

- Once the parameters are correctly defined, you can run the training script:  

```bash
python tg-gan-2/tg-gan-2_sc.py
python wgan/wgan-gp_sc.py
```

- The training process will generate the following output files:  

| Output File | Description |
|---|---|
| `loss_{n_epoch}.pkl` | Loss values recorded up to *n_epoch*. |
| `checkpoint_encoder_epoch_{n_epoch}.pth` | Encoder model checkpoint at *n_epoch*. |
| `checkpoint_decoder_epoch_{n_epoch}.pth` | Decoder model checkpoint at *n_epoch*. |
| `checkpoint_critics_epoch_{n_epoch}.pth` | Critics model checkpoint at *n_epoch*. |
| `checkpoint_regressor_epoch_{n_epoch}.pth` | Regressor model checkpoint at *n_epoch*. |
|  |  |
| `loss.pkl` | Loss values for all epochs. |
| `checkpoint_encoder.pth` | Final trained encoder model. |
| `checkpoint_decoder.pth` | Final trained decoder model. |
| `checkpoint_critics.pth` | Final trained critics model. |
| `checkpoint_regressor.pth` | Final trained regressor model. |

---

## How to Generate New Samples Using the Trained Model  

Once the synthesis models have been trained, they can be used to generate new samples.  

To do this, use the script **`utils/synthesizer.py`**.  
The function **`matrices_targets_synthesizer`** in `synthesizer.py` is the main function for synthesizing new structural connectivity matrices and objective variables.  

Example usage can be found in:  
- `regressor/synthesizer_tg-gan-2-sc.py`  
- `synthesizer_wgan-gp-sc.py`  

### Usage: `matrices_targets_synthesizer` (in `synthesizer.py`)  

#### **Inputs:**  

| Parameter | Description |
|---|---|
| `encoder` | Trained encoder model. |
| `decoder` | Trained decoder model. |
| `matrices` | Structural connectivity matrices to augment (`n_samples × n_regions × n_regions`). |
| `targets` | Objective variables to augment (`n_samples,`). |
| `n_splits` | Number of interpolations. |

️ **Warning:** Ensure that the structural connectivity matrices and objective variables are **sorted in the same order** based on the objective variables.  

#### **Outputs:**  

| Output | Description |
|---|---|
| `synthesized_matrices` | Synthesized structural connectivity matrices. |
| `synthesized_targets` | Synthesized objective variables. |

---


