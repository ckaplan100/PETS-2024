# Artifact Appendix

Paper title: A Cautionary Tale: On the Role of Reference Data in Empirical Privacy Defenses

Artifacts HotCRP Id: 77

Requested Badge: **Reproducible**

## Description
The artifact consists of a series of Jupyter notebooks that accompany our paper. These notebooks contain the code and experiments presented in the paper and demonstrate the empirical evaluations performed. Through these artifacts, readers can reproduce the findings, plots, and data analyses discussed in the paper.

### Security/Privacy Issues and Ethical Concerns
There are no ethical concerns regarding our artifacts.

## Basic Requirements

### Hardware Requirements
These experiments can be executed on any machine with NVIDIA GPUs that support CUDA 11.0 or later.

### Software Requirements
The experiments were run on a Linux-based system and the Python packages are all contained in the requirements.txt. 

### Estimated Time and Storage Consumption
Depending on the hardware used, each of the three experiment notebooks may take between 2 to 8 hours to complete and approximately 10GB of disk space is required for data and additional files.

## Environment

### Accessibility
Our artifacts are hosted in GitHub.

### Set up the environment
```bash
# Clone the repository
git clone https://github.com/ckaplan100/PETS-2024.git

# Navigate into the repository
cd PETS-2024

# Build the Docker image
docker build -t artifact_image .

# Run the Docker container with volume mount to ensure GitHub repo access
docker run -it --gpus all -v $(pwd):/workspace artifact_image

# Once inside the Docker container, navigate to the mounted workspace
cd /workspace

# Start Jupyter Notebook
jupyter notebook --ip 0.0.0.0 --allow-root
```

### Testing the Environment

## Artifact Evaluation

### Main Results and Claims

#### Main Result 1: Utility-Privacy Curves and Associated Values
Our paper presents utility-privacy curves which can be seen in Figure 3 and an associated table showing utility-privacy values under various training-reference data privacy settings in Table 2. The experiments supporting this claim involve running each of the algorithms 10 times and averaging the resulting utility and privacy scores.

#### Main Result 2: Pearson Correlation Coefficients
Our paper shows the Pearson correlation coefficients between theoretical and empirical relative privacy for each algorithm.

#### Main Result 3: Per Epoch Training Time
Our paper shows the per epoch training time for each algorithm.

### Experiments

#### Experiment 1: Utility-Privacy Curves and Values
How to Execute: Navigate to each Jupyter notebook: werm.ipynb, mmd.ipynb, and adv_reg.ipynb. Run the code under the sections titled "setup" and "run and save experiments". After each notebook finishes, execute all the code in evaluation.ipynb up to and including the section titled "Figure 3 Plot (including WERM-ES) / Table 2 Results".

Expected Result: You should see a plot corresponding to Figure 3 in the paper, which also includes results for WERM-ES as shown only in Table 2 and the appendix of the paper.

Time and Space: It may take between 2 to 8 hours to complete this experiment, consuming around 10GB of disk space.

Supported Claims: This experiment supports Main Result 1.

#### Experiment 2: Pearson Correlation Coefficients
How to Execute: After all experiments are run and saved, go back to evaluation.ipynb and run the code under the section titled "Pearson Correlation Coefficients".

Expected Result: You will get values corresponding to Table 6 in the paper's appendix.

Time and Space: This will take a few minutes and consume negligible additional disk space.

Supported Claims: This experiment supports Main Result 2.

#### Experiment 3: Per Epoch Training Time

How to Execute: Navigate back to each of the Jupyter notebooks: werm.ipynb, mmd.ipynb, and adv_reg.ipynb. Run the setup code and then execute the code under the section titled "evaluate per epoch training time".

Expected Result: The per-epoch training time will be printed as output and should corroborate the paper's claim regarding the relative speed of WERM compared to the other algorithms.

Time and Space: This experiment will take around 30 minutes to an hour and consume negligible additional disk space.

Supported Claims: This experiment supports Main Result 3.

## Limitations
The artifact primarily supports the reproduction of key results and figures in the paper: utility-privacy curves, Pearson correlation coefficients, and per-epoch training times. However, it is important to note that the time estimations provided are based on specific hardware configurations. Results may vary when using different hardware.

## Notes on Reusability
Our artifact can serve as a framework for benchmarking the utility-privacy tradeoff of new empirical privacy defenses that use reference data.
