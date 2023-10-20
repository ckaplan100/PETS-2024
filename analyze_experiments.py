from models import *
from train_utils import *
from eval_utils import *
from train import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from werm import runtime_werm
from adv_reg import runtime_adv_reg
from mmd import runtime_mmd


def get_experiment_data():
    # global
    overwrite_files = False
    datasets = ["cifar", "purchase", "texas"]
    random_seeds = list(np.arange(5, 15))
    best_valid_acc = False
    # denotes best batch size per config
    selected_epoch = {
        "cifar": {
            "weighted_erm": 24, # 128
            "mmd": {"warmup": 7, "no-warmup": 14}, # 512
            "adv_reg": {"no-ref_term": 24, "ref_term": 24} # 128
        },
        "purchase": {
            "weighted_erm": 19, # 512
            "mmd": {"warmup": 19, "no-warmup": 24}, # 512
            "adv_reg": {"no-ref_term": 9, "ref_term": 34} # 128
        },
        "texas": {
            "weighted_erm": 3, # 128
            "mmd": {"warmup": 7, "no-warmup": 7}, # 512
            "adv_reg": {"no-ref_term": 9, "ref_term": 19} # 128 , 10 / 18
        },
    }
    
    # weighted erm
    weight_props = {
        "cifar": [0.5, 0.7, 0.9, 0.97, 0.995, 1.],
        "purchase": [0.5, 0.7, 0.9, 0.97, 0.98, 1.],
        "texas": [0.5, 0.7, 0.9, 0.97, 0.999, 1.],
    }
    batch_sizes = [128, 512]
    
    # mmd regularization
    mmd_weights = [0.1, 0.2, 0.35, 0.7, 1.5]
    mmd_versions = ["no-warmup", "warmup"]
    
    # AdvReg
    alphas = {
        "cifar": [1e-6, 1e-3, 1e-1, 1.],
        "purchase": [1., 2., 3., 6., 10., 20.],
        "texas": [1., 2., 3., 6., 10., 20.]
    }
    adv_reg_versions = ["no-ref_term", "ref_term"]

    # load WERM data
    weighted_erm_metrics = []
    for dataset in datasets:
        for weight_prop in weight_props[dataset]:
            for batch_size in batch_sizes:
                for rttr in [1.0]:
                    for validation_metric in ["valid_acc"]:
                        for random_seed in random_seeds:
                            run_name = f"weight{weight_prop}-rttr{rttr}"
                            if batch_size == 512:
                                run_name += "-bs512"
                            folder = f"{dataset}_checkpoints/seed{random_seed}/weighted-erm/train-{run_name}"
                            if os.path.isdir(folder) is False:
                                print(f"not found: {folder}")
                                continue
                            else:
                                print(f"found: {folder}")                        
                            
                            if dataset == "texas":
                                epochs = 8
                            elif dataset == "purchase":
                                epochs = 20
                            elif dataset == "cifar":
                                epochs = 25
                            else:
                                raise ValueError("unhandled dataset")
                            run_name_to_save = "weighted_erm-bs512" if batch_size == 512 else "weighted_erm"
                            for epoch in range(epochs):
                                filename = f"{folder}/Depoch{epoch}"
                                if os.path.isfile(filename) is False:
                                    continue 
                                run_metrics = torch.load(open(filename, "rb"))
                                run_metrics = {i:(v.item() if isinstance(run_metrics[i], torch.Tensor) else v) for i, v in run_metrics.items()}
                                run_metrics["run_name"] = run_name_to_save
                                run_metrics["seed"] = random_seed
                                run_metrics["dataset"] = dataset
                                run_metrics["alpha"] = weight_prop
                                run_metrics["validation_metric"] = validation_metric
                                run_metrics["selected_epoch"] = selected_epoch[dataset]["weighted_erm"]
                                weighted_erm_metrics.append(run_metrics)
    w_erm_df = pd.DataFrame(weighted_erm_metrics)
    w_erm_by_epoch_df = w_erm_df.groupby(["dataset", "run_name", "epoch", "alpha", "selected_epoch"])[[
        "test_acc", "train_acc", "valid_acc", "conf_acc_train", "conf_acc_ref"]].mean().reset_index()
    if best_valid_acc:
        w_erm_by_run_df = w_erm_by_epoch_df.groupby(["dataset", "run_name", "alpha"]).apply(
            lambda x: x.loc[x["valid_acc"].idxmax()]).reset_index(drop=True)
    else:
        w_erm_by_run_df = w_erm_by_epoch_df.groupby(["dataset", "run_name", "alpha"]).apply(
            lambda x: x[x["epoch"] == x["selected_epoch"]]).reset_index(drop=True)

    # get NN attack results
    if best_valid_acc:
        nn_attack_results = []
        for dataset in datasets:
            for weight_prop in weight_props[dataset]:
                for batch_size in batch_sizes:
                    for rttr in [1.0]:
                        for random_seed in random_seeds:
                            run_name = f"weight{weight_prop}-rttr{rttr}"
                            if batch_size == 512:
                                run_name += "-bs512"
                            folder = f"{dataset}_checkpoints/seed{random_seed}/weighted-erm/train-{run_name}"
                            filename = f"{folder}/best_valid_acc_nn_attack"
                            if os.path.isfile(filename) is False:
                                continue   
                            nn_attack_metrics = torch.load(open(filename, "rb"))
                            
                            # ensure old runs aren't collected
                            if "epoch_train" in nn_attack_metrics:
                                continue
                            
                            run_name_to_save = "weighted_erm-bs512" if batch_size == 512 else "weighted_erm"
                            nn_attack_metrics["dataset"] = dataset
                            nn_attack_metrics["seed"] = random_seed
                            nn_attack_metrics["run_name"] = run_name_to_save
                            nn_attack_metrics["alpha"] = weight_prop
                            nn_attack_metrics
                            nn_attack_results.append(nn_attack_metrics)
    
        nn_attack_df = pd.DataFrame(nn_attack_results).rename(
            columns={
                "best_acc_train": "nn_acc_train", 
                "best_acc_ref": "nn_acc_ref"}
        )
        nn_attack_df = nn_attack_df.groupby(
            ["dataset", "run_name", "alpha"])[["nn_acc_train", "nn_acc_ref"]].mean().reset_index()    
        w_erm_by_run_df = w_erm_by_run_df.merge(nn_attack_df, how="left", on=["dataset", "run_name", "alpha"])

    # get WERM-ES df
    w_erm_es_df = pd.concat([
        w_erm_by_epoch_df[
            (w_erm_by_epoch_df["dataset"] == "purchase") &
            (w_erm_by_epoch_df["run_name"] == "weighted_erm-bs512") & 
            (w_erm_by_epoch_df["epoch"] == 6)
        ],
        w_erm_by_epoch_df[
            (w_erm_by_epoch_df["dataset"] == "texas") &
            (w_erm_by_epoch_df["run_name"] == "weighted_erm") & 
            (w_erm_by_epoch_df["epoch"] == 0)
        ],
        w_erm_by_epoch_df[
            (w_erm_by_epoch_df["dataset"] == "cifar") &
            (w_erm_by_epoch_df["run_name"] == "weighted_erm") & 
            (w_erm_by_epoch_df["epoch"] == 5)
        ]
    ])
    w_erm_es_df["run_name"] = "weighted_erm-es"

    # final WERM df
    w_erm_by_run_df = pd.concat([w_erm_by_run_df, w_erm_es_df])


    # load MMD data
    mmd_metrics = []
    for dataset in datasets:
        for mmd_weight in mmd_weights:
            for mmd_version in mmd_versions:
                for rttr in [1.0]:
                    for validation_metric in ["valid_acc"]:
                        for random_seed in random_seeds:
                            if dataset != "cifar" and mmd_weight == 1e-4:
                                continue
                            
                            run_name = f"weight{mmd_weight}-rttr{rttr}"
                            if mmd_version == "no-warmup":
                                run_name += "-no-warmup"
                            folder = f"{dataset}_checkpoints/seed{random_seed}/mmd-regularization/train-{run_name}"
                            if os.path.isdir(folder) is False:
                                print(f"not found: {folder}")
                                continue
                            else:
                                print(f"found: {folder}") 
                        
                            if dataset == "texas":
                                if mmd_version == "no-warmup":
                                    epochs = 16
                                else:
                                    epochs = 8
                            elif dataset == "purchase":
                                if mmd_version == "no-warmup":
                                    epochs = 40
                                else:
                                    epochs = 20
                            elif dataset == "cifar":
                                if mmd_version == "no-warmup":
                                    epochs = 16
                                else:
                                    epochs = 8
                            else:
                                raise ValueError("unhandled dataset")
                            
                            run_name_to_save = "mmd-no-warmup" if mmd_version == "no-warmup" else "mmd"
                            for epoch in range(epochs):
                                run_metrics = torch.load(open(f"{folder}/Depoch{epoch}", "rb"))
                                run_metrics = {i:(v.item() if isinstance(run_metrics[i], torch.Tensor) else v) for i, v in run_metrics.items()}
                                run_metrics["run_name"] = run_name_to_save
                                run_metrics["seed"] = random_seed
                                run_metrics["dataset"] = dataset
                                run_metrics["alpha"] = mmd_weight
                                run_metrics["validation_metric"] = validation_metric
                                run_metrics["selected_epoch"] = selected_epoch[dataset]["mmd"][mmd_version]
                                mmd_metrics.append(run_metrics)
    mmd_df = pd.DataFrame(mmd_metrics)
    mmd_by_epoch_df = mmd_df.groupby(["dataset", "run_name", "epoch", "alpha", "selected_epoch"])[[
        "test_acc", "valid_acc", "conf_acc_train", "conf_acc_ref"]].mean().reset_index()
    if best_valid_acc:
        mmd_by_run_df = mmd_by_epoch_df.groupby(["dataset", "run_name", "alpha"]).apply(
            lambda x: x.loc[x["valid_acc"].idxmax()]).reset_index(drop=True)
    else:
        mmd_by_run_df = mmd_by_epoch_df.groupby(["dataset", "run_name", "alpha"]).apply(
            lambda x: x[x["epoch"] == x["selected_epoch"]]).reset_index(drop=True)
    # get NN attack results
    if best_valid_acc:
        nn_attack_results = []
        for dataset in datasets:
            for mmd_weight in mmd_weights:
                for mmd_version in mmd_versions:
                    for rttr in [1.0]:
                        for random_seed in random_seeds:
                            run_name = f"weight{mmd_weight}-rttr{rttr}"
                            if mmd_version == "no-warmup":
                                run_name += "-no-warmup"
                            folder = f"{dataset}_checkpoints/seed{random_seed}/mmd-regularization/train-{run_name}"
                            filename = f"{folder}/best_valid_acc_nn_attack"
                            if os.path.isfile(filename) is False:
                                continue 
                            nn_attack_metrics = torch.load(open(filename, "rb"))
                            
                            # ensure old runs aren't collected
                            if "epoch_train" in nn_attack_metrics:
                                continue
                            
                            run_name_to_save = "mmd-no-warmup" if mmd_version == "no-warmup" else "mmd"
                            nn_attack_metrics["dataset"] = dataset
                            nn_attack_metrics["seed"] = random_seed
                            nn_attack_metrics["run_name"] = run_name_to_save
                            nn_attack_metrics["alpha"] = mmd_weight
                            nn_attack_metrics
                            nn_attack_results.append(nn_attack_metrics)
    
        nn_attack_df = pd.DataFrame(nn_attack_results).rename(
            columns={
                "best_acc_train": "nn_acc_train", 
                "best_acc_ref": "nn_acc_ref"}
        )
        nn_attack_df = nn_attack_df.groupby(
            ["dataset", "run_name", "alpha"])[["nn_acc_train", "nn_acc_ref"]].mean().reset_index()     
        mmd_by_run_df = mmd_by_run_df.merge(nn_attack_df, how="left", on=["dataset", "run_name", "alpha"])

    # load AdvReg data
    adv_reg_metrics = []
    for dataset in datasets:
        for alpha in alphas[dataset]:
            for adv_reg_version in adv_reg_versions:
                for rttr in [1.0]:
                    for validation_metric in ["valid_acc"]:
                        for random_seed in random_seeds:
                            if adv_reg_version == "ref_term":
                                version_tag = "nf"
                            elif adv_reg_version == "no-ref_term":
                                version_tag = "of"
                            else:
                                raise ValueError("unhandled adv_reg_version")
                                
                            if dataset == "texas":
                                epochs = 20
                            elif dataset == "purchase":
                                epochs = 40
                            elif dataset == "cifar":
                                epochs = 30
                            else:
                                raise ValueError("unhandled dataset")
                                
                            run_name = f"coin_flip-mse-{version_tag}-{rttr}-{epochs}-20-1"
                            
                            folder = f"{dataset}_checkpoints/seed{random_seed}/alpha{alpha}/train-{run_name}"
                            if os.path.isdir(folder) is False:
                                print(f"not found: {folder}")
                                continue
                            else:
                                print(f"found: {folder}") 
                        
                            run_name_to_save = "adv_reg-ref_term" if adv_reg_version == "ref_term" else "adv_reg"
                            for epoch in range(epochs):
                                filename = f"{folder}/Depoch{epoch}"
                                if os.path.isfile(filename) is False:
                                    continue 
                                run_metrics = torch.load(open(filename, "rb"))
                                run_metrics = {i:(v.item() if isinstance(run_metrics[i], torch.Tensor) else v) for i, v in run_metrics.items()}
                                run_metrics["run_name"] = run_name_to_save
                                run_metrics["seed"] = random_seed
                                run_metrics["dataset"] = dataset
                                run_metrics["alpha"] = alpha
                                run_metrics["validation_metric"] = validation_metric
                                run_metrics["selected_epoch"] = selected_epoch[dataset]["adv_reg"][adv_reg_version]
                                adv_reg_metrics.append(run_metrics)
    adv_reg_df = pd.DataFrame(adv_reg_metrics)
    adv_reg_by_epoch_df = adv_reg_df.groupby(["dataset", "run_name", "epoch", "alpha", "selected_epoch"])[[
        "test_acc", "valid_acc", "conf_acc_train", "conf_acc_ref"]].mean().reset_index()
    if best_valid_acc:
        adv_reg_by_run_df = adv_reg_by_epoch_df.groupby(["dataset", "run_name", "alpha"]).apply(
            lambda x: x.loc[x["valid_acc"].idxmax()]).reset_index(drop=True)
    else:
        adv_reg_by_run_df = adv_reg_by_epoch_df.groupby(["dataset", "run_name", "alpha"]).apply(
            lambda x: x[x["epoch"] == x["selected_epoch"]]).reset_index(drop=True)
    # get NN attack results
    if best_valid_acc:
        nn_attack_results = []   
        for dataset in datasets:
            for alpha in alphas[dataset]:
                for adv_reg_version in adv_reg_versions:
                    for rttr in [1.0]:
                        for random_seed in random_seeds:
                            if adv_reg_version == "ref_term":
                                version_tag = "nf"
                            elif adv_reg_version == "no-ref_term":
                                version_tag = "of"
                            else:
                                raise ValueError("unhandled adv_reg_version")
                                
                            if dataset == "texas":
                                epochs = 20
                            elif dataset == "purchase":
                                epochs = 40
                            elif dataset == "cifar":
                                epochs = 30
                            else:
                                raise ValueError("unhandled dataset")
    
                            run_name = f"coin_flip-mse-{version_tag}-{rttr}-{epochs}-20-1"
                            folder = f"{dataset}_checkpoints/seed{random_seed}/alpha{alpha}/train-{run_name}"
                            filename = f"{folder}/best_valid_acc_nn_attack"
                            if os.path.isfile(filename) is False:
                                continue 
                            nn_attack_metrics = torch.load(open(filename, "rb"))
                            
                            # ensure old runs aren't collected
                            if "epoch_train" in nn_attack_metrics:
                                continue
                            
                            run_name_to_save = "adv_reg-ref_term" if adv_reg_version == "ref_term" else "adv_reg"
                            nn_attack_metrics["dataset"] = dataset
                            nn_attack_metrics["seed"] = random_seed
                            nn_attack_metrics["run_name"] = run_name_to_save
                            nn_attack_metrics["alpha"] = alpha
                            nn_attack_metrics
                            nn_attack_results.append(nn_attack_metrics)
    
        nn_attack_df = pd.DataFrame(nn_attack_results).rename(
            columns={
                "best_acc_train": "nn_acc_train", 
                "best_acc_ref": "nn_acc_ref"}
        )
        nn_attack_df = nn_attack_df.groupby(
            ["dataset", "run_name", "alpha"])[["nn_acc_train", "nn_acc_ref"]].mean().reset_index() 
        adv_reg_by_run_df = adv_reg_by_run_df.merge(nn_attack_df, how="left", on=["dataset", "run_name", "alpha"])

    by_run_df = pd.concat([w_erm_by_run_df, mmd_by_run_df, adv_reg_by_run_df])
    return by_run_df
    

def utility_privacy_evaluation(by_run_df, best_valid_acc):
    print("Running Utility-Privacy evaluation...")
    conf_acc_train_df = by_run_df[["dataset", "run_name", "alpha", "test_acc", "conf_acc_train"]].rename(columns={"conf_acc_train": "conf_acc"})
    conf_acc_train_df["target_dataset"] = ["training"] * len(conf_acc_train_df)
    conf_acc_ref_df = by_run_df[["dataset", "run_name", "alpha", "test_acc", "conf_acc_ref"]].rename(columns={"conf_acc_ref": "conf_acc"})
    conf_acc_ref_df["target_dataset"] = ["reference"] * len(conf_acc_ref_df)
    twin_plot_df = pd.concat([conf_acc_train_df, conf_acc_ref_df]).reset_index(drop=True)
    twin_plot_df["test_acc"] = twin_plot_df["test_acc"] / 100

    if best_valid_acc:
        nn_acc_train_df = by_run_df[["dataset", "run_name", "alpha", "test_acc", "nn_acc_train"]].rename(columns={"nn_acc_train": "nn_acc"})
        nn_acc_train_df["target_dataset"] = ["training"] * len(nn_acc_train_df)
        nn_acc_ref_df = by_run_df[["dataset", "run_name", "alpha", "test_acc", "nn_acc_ref"]].rename(columns={"nn_acc_ref": "nn_acc"})
        nn_acc_ref_df["target_dataset"] = ["reference"] * len(nn_acc_ref_df)
        twin_plot_nn_df = pd.concat([nn_acc_train_df, nn_acc_ref_df]).reset_index(drop=True)
        twin_plot_nn_df["test_acc"] = twin_plot_nn_df["test_acc"] / 100
        twin_plot_nn_df = twin_plot_nn_df.dropna()

    fig, ax = plt.subplots(3, 1, figsize=(16, 18))
    markers = {"training": "s", "reference": "X"}
    color_palette = sns.color_palette("colorblind", 5)
    color_map = {
        'WERM': color_palette[0],
        'MMD': color_palette[1],
        'AdvReg': color_palette[2],
        'AdvReg-RT': color_palette[3],
        'WERM-ES': color_palette[4],
    }
    
    # purchase
    purchase_run_names = {
        "weighted_erm": "weighted_erm-bs128",
        "weighted_erm-es": "WERM-ES",
        "weighted_erm-bs512": "WERM",
        "mmd": "MMD",
        "mmd-no-warmup": "mmd-no-warmup",
        "adv_reg": "AdvReg", 
        "adv_reg-ref_term": "AdvReg-RT", 
    }
    
    twin_plot_purchase_df = twin_plot_df[twin_plot_df["dataset"] == "purchase"]
    twin_plot_purchase_df["run_name"] = twin_plot_purchase_df["run_name"].apply(
        lambda x: purchase_run_names[x] if x in purchase_run_names else x)
    twin_plot_purchase_df["Defense Method"] = twin_plot_purchase_df["run_name"]
    twin_plot_purchase_df["Target Dataset"] = twin_plot_purchase_df["target_dataset"]
    
    sns.scatterplot(
        data=twin_plot_purchase_df[
            (twin_plot_purchase_df["test_acc"] > 0.6) & 
            ~(twin_plot_purchase_df["run_name"].isin(
                ["mmd-no-warmup", "weighted_erm-bs128"]))#, "WERM-ES"])
    
        ], 
        x="test_acc", y="conf_acc", hue="Defense Method", style="Target Dataset", markers=markers,
        palette=color_map, marker=10, s=100, ax=ax[0])
    ax[0].set_ylabel("MIA Accuracy", fontsize=12)
    ax[0].set_xlabel("Test Accuracy", fontsize=12)
    ax[0].set_title("Purchase100 - Comprehensive Tradeoff Analysis", fontsize=14)
    
    # texas
    texas_run_names = {
        "weighted_erm": "WERM",
        "weighted_erm-es": "WERM-ES",
        "weighted_erm-bs512": "weighted_erm-bs512",
        "mmd": "mmd-warmup",
        "mmd-no-warmup": "MMD",
        "adv_reg": "AdvReg", 
        "adv_reg-ref_term": "AdvReg-RT",
    }
    
    twin_plot_texas_df = twin_plot_df[twin_plot_df["dataset"] == "texas"]
    twin_plot_texas_df["run_name"] = twin_plot_texas_df["run_name"].apply(
        lambda x: texas_run_names[x] if x in texas_run_names else x)
    twin_plot_texas_df["Defense Method"] = twin_plot_texas_df["run_name"]
    twin_plot_texas_df["Target Dataset"] = twin_plot_texas_df["target_dataset"]
    
    
    sns.scatterplot(
        data=twin_plot_texas_df[
            (twin_plot_texas_df["test_acc"] > 0.35) & 
            ~(twin_plot_texas_df["run_name"].isin(["mmd-warmup", "weighted_erm-bs512"]))#, "WERM-ES"]))
        ], 
        x="test_acc", y="conf_acc", hue="Defense Method", style="Target Dataset", markers=markers,
        palette=color_map, marker=10, s=100, ax=ax[1])
    ax[1].set_ylabel("MIA Accuracy", fontsize=12)
    ax[1].set_xlabel("Test Accuracy", fontsize=12)
    ax[1].set_title("Texas100 - Comprehensive Tradeoff Analysis", fontsize=14)
    
    # cifar
    cifar_run_names = {
        "weighted_erm": "WERM",
        "weighted_erm-es": "WERM-ES",
        "weighted_erm-bs512": "weighted_erm-bs512",
        "mmd": "mmd-warmup",
        "mmd-no-warmup": "MMD",
        "adv_reg": "AdvReg", 
        "adv_reg-ref_term": "AdvReg-RT",
    }
    twin_plot_cifar_df = twin_plot_df[twin_plot_df["dataset"] == "cifar"]
    twin_plot_cifar_df["run_name"] = twin_plot_cifar_df["run_name"].apply(
        lambda x: cifar_run_names[x] if x in cifar_run_names else x)
    twin_plot_cifar_df["Defense Method"] = twin_plot_cifar_df["run_name"]
    twin_plot_cifar_df["Target Dataset"] = twin_plot_cifar_df["target_dataset"]
    
    
    sns.scatterplot(
        data=twin_plot_cifar_df[
            (twin_plot_cifar_df["dataset"] == "cifar") & 
            ~(twin_plot_cifar_df["run_name"].isin(["mmd-warmup", "weighted_erm-bs512"]))#, "WERM-ES"]))
        ], 
        x="test_acc", y="conf_acc", hue="Defense Method", style="Target Dataset", markers=markers,
        palette=color_map, marker=10, s=100, ax=ax[2])
    ax[2].set_ylabel("MIA Accuracy", fontsize=12)
    ax[2].set_xlabel("Test Accuracy", fontsize=12)
    ax[2].set_title("CIFAR100 - Comprehensive Tradeoff Analysis", fontsize=14)
    fig.savefig("utility-privacy-analysis.png")


def pcc_evaluation(by_run_df, best_valid_acc):
    print("Running PCC evaluation...")
    by_run_df2 = deepcopy(by_run_df)
    by_run_df2["emp_priv_ratio"] = (by_run_df2["conf_acc_train"] - 0.5) / (by_run_df2["conf_acc_ref"] - 0.5)
    by_run_df2["the_priv_ratio"] = by_run_df2["alpha"] / (1 - by_run_df2["alpha"])

    datasets = ["purchase", "texas", "cifar"]
    run_names = {
        "purchase": {
            "weighted_erm": "weighted_erm-bs128",
            "weighted_erm-es": "WERM-ES",
            "weighted_erm-bs512": "WERM",
            "mmd": "MMD",
            "mmd-no-warmup": "mmd-no-warmup",
            "adv_reg": "AdvReg", 
            "adv_reg-ref_term": "AdvReg-RT", 
        },
        "texas": {
            "weighted_erm": "WERM",
            "weighted_erm-es": "WERM-ES",
            "weighted_erm-bs512": "weighted_erm-bs512",
            "mmd": "mmd-warmup",
            "mmd-no-warmup": "MMD",
            "adv_reg": "AdvReg", 
            "adv_reg-ref_term": "AdvReg-RT",
        },
        "cifar": {
            "weighted_erm": "WERM",
            "weighted_erm-es": "WERM-ES",
            "weighted_erm-bs512": "weighted_erm-bs512",
            "mmd": "mmd-warmup",
            "mmd-no-warmup": "MMD",
            "adv_reg": "AdvReg", 
            "adv_reg-ref_term": "AdvReg-RT",
        }
    }
    
    def change_run_name(row):
        row_dataset = row["dataset"]
        return run_names[row_dataset][row["run_name"]]
    
    by_run_df2["run_name"] = by_run_df2.apply(change_run_name, axis=1)
    by_run_df2 = by_run_df2[~(by_run_df2["run_name"].isin(
        ["weighted_erm-bs512", "weighted_erm-bs128", "mmd-warmup", "mmd-no-warmup"]
    ))]

    by_run_df2 = by_run_df2[~((by_run_df2["alpha"] == 1.) & (by_run_df2["run_name"].isin(["WERM", "WERM-ES"])))]
    by_run_df2["conf_acc_ref"] = by_run_df2["conf_acc_ref"].apply(lambda x: 0.501 if x < 0.5 else x)
    by_run_df2["emp_priv_ratio"] = (by_run_df2["conf_acc_train"] - 0.5) / (by_run_df2["conf_acc_ref"] - 0.5)
    by_run_df2["the_priv_ratio"] = by_run_df2.apply(
        lambda row: row["alpha"] / (1 - row["alpha"]) if row["run_name"] in ["WERM", "WERM-ES"] else 1 / row["alpha"], axis=1)

    by_run_df2_ds = by_run_df2[by_run_df2["dataset"] != ""]
    
    # only include alpha values that are common among all datasets for overall analysis
    werm_corrcoef_relative_privacy = np.corrcoef(
        by_run_df2_ds[(by_run_df2_ds["run_name"] == "WERM") & ~(by_run_df2_ds["alpha"].isin([0.98, 0.999, 0.995]))]["emp_priv_ratio"], 
        by_run_df2_ds[(by_run_df2_ds["run_name"] == "WERM") & ~(by_run_df2_ds["alpha"].isin([0.98, 0.999, 0.995]))]["the_priv_ratio"]
    )
    for dataset in datasets:
        pcc_ds = np.corrcoef(
            by_run_df2_ds[(by_run_df2_ds["dataset"] == dataset) & (by_run_df2_ds["run_name"] == "WERM") & 
                          ~(by_run_df2_ds["alpha"].isin([0.98, 0.999, 0.995]))]["emp_priv_ratio"], 
            by_run_df2_ds[(by_run_df2_ds["dataset"] == dataset) & (by_run_df2_ds["run_name"] == "WERM") & 
                          ~(by_run_df2_ds["alpha"].isin([0.98, 0.999, 0.995]))]["the_priv_ratio"]
        )[0][1]
        print(f"WERM PCC {dataset}: {pcc_ds}")
    print(f"WERM PCC Overall: {werm_corrcoef_relative_privacy[0][1]}")

    mmd_corrcoef_relative_privacy = np.corrcoef(
        by_run_df2_ds[(by_run_df2_ds["run_name"] == "MMD") & (by_run_df2_ds["emp_priv_ratio"] != np.inf)]["emp_priv_ratio"], 
        by_run_df2_ds[(by_run_df2_ds["run_name"] == "MMD") & (by_run_df2_ds["emp_priv_ratio"] != np.inf)]["the_priv_ratio"]
    )
    for dataset in datasets:
        pcc_ds = np.corrcoef(
            by_run_df2_ds[(by_run_df2_ds["dataset"] == dataset) & (by_run_df2_ds["run_name"] == "MMD") & (by_run_df2_ds["emp_priv_ratio"] != np.inf)
                         ]["emp_priv_ratio"], 
            by_run_df2_ds[(by_run_df2_ds["dataset"] == dataset) & (by_run_df2_ds["run_name"] == "MMD") & (by_run_df2_ds["emp_priv_ratio"] != np.inf)
                         ]["the_priv_ratio"]
        )[0][1]
        print(f"MMD PCC {dataset}: {pcc_ds}")
    print(f"MMD PCC Overall: {mmd_corrcoef_relative_privacy[0][1]}")

    advreg_corrcoef_relative_privacy = np.corrcoef(
        by_run_df2_ds[(by_run_df2_ds["run_name"] == "AdvReg") & (by_run_df2_ds["emp_priv_ratio"] != np.inf)]["emp_priv_ratio"], 
        by_run_df2_ds[(by_run_df2_ds["run_name"] == "AdvReg") & (by_run_df2_ds["emp_priv_ratio"] != np.inf)]["the_priv_ratio"]
    )
    for dataset in datasets:
        pcc_ds = np.corrcoef(
            by_run_df2_ds[(by_run_df2_ds["dataset"] == dataset) & (by_run_df2_ds["run_name"] == "AdvReg") & 
                          (by_run_df2_ds["emp_priv_ratio"] != np.inf)]["emp_priv_ratio"], 
            by_run_df2_ds[(by_run_df2_ds["dataset"] == dataset) & (by_run_df2_ds["run_name"] == "AdvReg") & 
                          (by_run_df2_ds["emp_priv_ratio"] != np.inf)]["the_priv_ratio"]
        )[0][1]
        print(f"AdvReg PCC {dataset}: {pcc_ds}")
    print(f"AdvReg PCC Overall: {advreg_corrcoef_relative_privacy[0][1]}")

def runtime_evaluation():
    print("Running Runtime evaluation...")
    runtime_werm()
    runtime_adv_reg()
    runtime_mmd()


def evaluate_experiments():
    parser = argparse.ArgumentParser(description="Choose an evaluation method.")
    parser.add_argument("--evaluation", choices=["utility-privacy", "PCC", "runtime"], required=True, help="Evaluation method to be executed")
    parser.add_argument("--best_valid_acc", action='store_true', default=False, help="Use best validation accuracy if set, otherwise use the last epoch's accuracy.")
    args = parser.parse_args()

    if args.evaluation == "utility-privacy":
        exp_data = get_experiment_data()
        utility_privacy_evaluation(exp_data, args.best_valid_acc)
    elif args.evaluation == "pcc":
        exp_data = get_experiment_data()
        pcc_evaluation(exp_data, args.best_valid_acc)
    elif args.evaluation == "runtime":
        runtime_evaluation()
    else:
        print("Invalid evaluation choice!")

if __name__ == "__main__":
    evaluate_experiments()

    


