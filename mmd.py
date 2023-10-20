from models import *
from train_utils import *
from eval_utils import *
from train import *
from tqdm import tqdm

def execute_mmd(test_one_seed):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    run_datasets = ["purchase", "texas", "cifar"]
    if test_one_seed:
        random_seeds = [5]
    else:
        random_seeds = list(np.arange(5, 15))    
    use_validation = True
    mmd_weights = [0.1, 0.2, 0.35, 0.7, 1.5]
    ref_to_train_ratio = 1.
    mmd_scale = 1.
    start_mmd_epochs = [-1, 1]

    total_experiments = len(run_datasets) * len(random_seeds) * len(mmd_weights) * len(start_mmd_epochs)
    experiment_index = 0

    for dataset in run_datasets:
        for random_seed in random_seeds:
            for mmd_weight in mmd_weights:
                for start_mmd_epoch in start_mmd_epochs:                    
    
                    set_seed(random_seed)
    
                    run_name = f"weight{mmd_weight}-rttr{ref_to_train_ratio}"
    
                    if start_mmd_epoch == -1:
                        run_name += "-no-warmup"

                    print(f"Running Experiment {experiment_index}/{total_experiments} - Dataset: {dataset}, Random Seed: {random_seed}, Run Name: {run_name}") 
                    experiment_index += 1    
    
                    best_valid_acc_state_dict = None
                    best_total_valid_loss_state_dict = None
                    best_valid_acc = 0.
                    best_valid_acc_epoch = -1
                    best_valid_loss = 1e5
                    best_valid_loss_epoch = -1
    
                    if random_seed in [5, 10]:
                        load_randomization = True
                    else:
                        load_randomization = False
    
                    if dataset == "texas":
                        if start_mmd_epoch == -1:
                            epochs = 16
                        else:
                            epochs = 8
                        num_features = 6169
                        train_classifier_ratio = 0.15
                        train_test_ratio = 0.4
                        train_attack_ratio = train_classifier_ratio * ref_to_train_ratio
                        train_valid_ratio = 1 - train_classifier_ratio - train_attack_ratio - train_test_ratio
                        data_is_numpy = True
                        batch_size = 512
                    elif dataset == "purchase":
                        if start_mmd_epoch == -1:
                            epochs = 40
                        else:
                            epochs = 20
                        num_features = 600
                        train_classifier_ratio = 0.1
                        train_test_ratio = 0.4
                        train_attack_ratio = train_classifier_ratio * ref_to_train_ratio
                        train_valid_ratio = 1 - train_classifier_ratio - train_attack_ratio - train_test_ratio
                        data_is_numpy = True
                        batch_size = 512
                    elif dataset == "cifar":
                        if start_mmd_epoch == -1:
                            epochs = 16
                        else:
                            epochs = 8
                        train_classifier_ratio, train_attack_ratio, train_valid_ratio = None, None, None
                        data_is_numpy = False  
                        batch_size = 512
                    else:
                        raise ValueError("not handled dataset")
    
                    train_classifier_data, train_classifier_label, train_attack_data, train_attack_label, valid_data, valid_label, test_data, test_label = load_data(
                        dataset=dataset, load_randomization=load_randomization, use_validation=use_validation,
                        train_classifier_ratio=train_classifier_ratio, 
                        train_attack_ratio=train_attack_ratio, 
                        train_valid_ratio=train_valid_ratio
                    )
                    if dataset == "cifar":
                        model = resnet18(pretrained=False)
                        model.fc = nn.Linear(512, 100)
                    else:
                        model = TabularClassifier(num_features=num_features)
                    model = model.to(device)
                    criterion = nn.CrossEntropyLoss(reduction='none')
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    
                    for epoch in tqdm(range(epochs)):
                        if dataset in ["purchase", "texas"]:
                            train_classifier_data_tensor = torch.from_numpy(train_classifier_data).type(torch.FloatTensor)
                            train_classifier_label_tensor = torch.from_numpy(train_classifier_label).type(torch.LongTensor)
    
                            train_attack_data_tensor = torch.from_numpy(train_attack_data).type(torch.FloatTensor)
                            train_attack_label_tensor = torch.from_numpy(train_attack_label).type(torch.LongTensor)
    
                            valid_data_tensor = torch.from_numpy(valid_data).type(torch.FloatTensor)
                            valid_label_tensor = torch.from_numpy(valid_label).type(torch.LongTensor)
    
                            test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
                            test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)
                        elif dataset == "cifar":
                            train_classifier_data_tensor = train_classifier_data.type(torch.FloatTensor)
                            train_classifier_label_tensor = train_classifier_label.type(torch.LongTensor)
    
                            train_attack_data_tensor = train_attack_data.type(torch.FloatTensor)
                            train_attack_label_tensor = train_attack_label.type(torch.LongTensor)
    
                            valid_data_tensor = valid_data.type(torch.FloatTensor)
                            valid_label_tensor = valid_label.type(torch.LongTensor)
    
                            test_data_tensor = test_data.type(torch.FloatTensor)
                            test_label_tensor = test_label.type(torch.LongTensor)
                        else:
                            raise ValueError("unhandled dataset")
                        
                        r = np.arange(len(train_classifier_data_tensor))
                        np.random.shuffle(r)
                        train_classifier_data_tensor = train_classifier_data_tensor[r]
                        train_classifier_label_tensor = train_classifier_label_tensor[r]
                        
                        r = np.arange(len(train_attack_data_tensor))
                        np.random.shuffle(r)
                        train_attack_data_tensor = train_attack_data_tensor[r]
                        train_attack_label_tensor = train_attack_label_tensor[r]
    
    #                     print('\nEpoch: [%d | %d]' % (epoch, epochs))
    
                        # train with weights
                        train_loss, train_acc = train(
                            train_classifier_data_tensor, train_classifier_label_tensor, model, criterion, optimizer, 
                            batch_size, epoch, device, 
                            mmd_weight=mmd_weight, mmd_scale=mmd_scale, start_mmd_epoch=start_mmd_epoch, 
                            ref_data=train_attack_data_tensor, ref_labels=train_attack_label_tensor, 
                            unique_labels=True, mmd_ref_term=True
                        )
    
                        # get loss with data splits
                        train_loss, train_acc = test(train_classifier_data_tensor, train_classifier_label_tensor, model, criterion, 128, epoch, device)
    
                        ref_loss, ref_acc = test(train_attack_data_tensor, train_attack_label_tensor, model, criterion, 128, epoch, device)
    
                        valid_loss, valid_acc = test(valid_data_tensor, valid_label_tensor, model, criterion, 128, epoch, device)
    
                        test_loss, test_acc = test(test_data_tensor, test_label_tensor, model, criterion, 128, epoch, device)
    
                        # get privacy attack metrics per epoch
                        # test attack eval - training data
                        corr_acc_train, conf_acc_train, entr_acc_train, mod_entr_acc_train = evaluation_metrics(
                            model, train_classifier_data, train_classifier_label, test_data, test_label, data_is_numpy, device=device)
    
                        # test attack eval - ref data
                        corr_acc_ref, conf_acc_ref, entr_acc_ref, mod_entr_acc_ref = evaluation_metrics(
                            model, train_attack_data, train_attack_label, test_data, test_label, data_is_numpy, device=device)
                        
    #                     print(f'Train Acc: {train_acc}, Ref Acc: {ref_acc}, Valid Acc: {valid_acc}, Test Acc: {test_acc}')
    #                     print(f'Train Loss: {train_loss}, Ref Loss: {ref_loss}, Valid Loss: {valid_loss}, Test Loss: {test_loss}')
    #                     print(f"Conf Attack Train: {conf_acc_train}, Conf Attack Ref: {conf_acc_ref}")                      
    #                     print(f'Gap Attack: {1/2 + (train_acc / 100 - test_acc / 100) / 2}')
    
                        filename = f'seed{random_seed}/mmd-regularization/train-{run_name}'
    
                        if valid_acc.item() > best_valid_acc:
                            best_valid_acc = valid_acc.item()
                            best_valid_acc_epoch = epoch
                            best_valid_acc_state_dict = deepcopy(model.state_dict())
    
                        if valid_loss.item() < best_valid_loss:
                            best_valid_loss = valid_loss.item()
                            best_valid_loss_epoch = epoch
                            best_total_valid_loss_state_dict = deepcopy(model.state_dict())
                        
                        save_checkpoint({        
                                'epoch': epoch,
                                'test_acc': test_acc,
                                'test_loss': test_loss,
                                'train_acc': train_acc,
                                'train_loss': train_loss,
                                'valid_acc': valid_acc,
                                'valid_loss': valid_loss,
                                'ref_acc': ref_acc,
                                'ref_loss': ref_loss,
                                'conf_acc_train': conf_acc_train,
                                'conf_acc_ref': conf_acc_ref
                            }, filename=filename, filename_end='Depoch%d'%epoch, checkpoint=f'./{dataset}_checkpoints')
    
                    # save best models
                    save_checkpoint(
                        {"state_dict": best_valid_acc_state_dict}, 
                        checkpoint=f'./{dataset}_checkpoints',
                        filename=filename,
                        filename_end='best_valid_acc_model'
                    )
                    save_checkpoint(
                        {"state_dict": best_total_valid_loss_state_dict}, 
                        checkpoint=f'./{dataset}_checkpoints',
                        filename=filename,
                        filename_end='best_valid_total_loss_model'
                    )
    
    #                 print(f"Best Valid Acc: {best_valid_acc}, Epoch: {best_valid_acc_epoch}")
    #                 print(f"Best Valid Loss: {best_valid_loss}, Epoch: {best_valid_loss_epoch}")


def runtime_mmd():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    run_datasets = ["purchase", "texas", "cifar"]
    random_seeds = list(np.arange(5, 15))
    use_validation = True
    mmd_weights = [0.7]
    ref_to_train_ratio = 1.
    mmd_scale = 1.
    start_mmd_epochs = [-1]

    time_per_epoch_mmd = {"cifar": [], "purchase": [], "texas": []}
    for dataset in run_datasets:
        for random_seed in random_seeds:
            for mmd_weight in mmd_weights:
                for start_mmd_epoch in start_mmd_epochs:                    
    
                    set_seed(random_seed)
    
                    run_name = f"weight{mmd_weight}-rttr{ref_to_train_ratio}"
    
                    if start_mmd_epoch == -1:
                        run_name += "-no-warmup"
    
                    # print(dataset, random_seed, run_name)
    
                    best_valid_acc_state_dict = None
                    best_total_valid_loss_state_dict = None
                    best_valid_acc = 0.
                    best_valid_acc_epoch = -1
                    best_valid_loss = 1e5
                    best_valid_loss_epoch = -1
    
                    if random_seed in [5, 10]:
                        load_randomization = True
                    else:
                        load_randomization = False
    
                    if dataset == "texas":
                        if start_mmd_epoch == -1:
                            epochs = 1
                        else:
                            epochs = 1
                        num_features = 6169
                        train_classifier_ratio = 0.15
                        train_test_ratio = 0.4
                        train_attack_ratio = train_classifier_ratio * ref_to_train_ratio
                        train_valid_ratio = 1 - train_classifier_ratio - train_attack_ratio - train_test_ratio
                        data_is_numpy = True
                        batch_size = 512
                    elif dataset == "purchase":
                        if start_mmd_epoch == -1:
                            epochs = 1
                        else:
                            epochs = 1
                        num_features = 600
                        train_classifier_ratio = 0.1
                        train_test_ratio = 0.4
                        train_attack_ratio = train_classifier_ratio * ref_to_train_ratio
                        train_valid_ratio = 1 - train_classifier_ratio - train_attack_ratio - train_test_ratio
                        data_is_numpy = True
                        batch_size = 512
                    elif dataset == "cifar":
                        if start_mmd_epoch == -1:
                            epochs = 1
                        else:
                            epochs = 1
                        train_classifier_ratio, train_attack_ratio, train_valid_ratio = None, None, None
                        data_is_numpy = False  
                        batch_size = 512
                    else:
                        raise ValueError("not handled dataset")
    
                    train_classifier_data, train_classifier_label, train_attack_data, train_attack_label, valid_data, valid_label, test_data, test_label = load_data(
                        dataset=dataset, load_randomization=load_randomization, use_validation=use_validation,
                        train_classifier_ratio=train_classifier_ratio, 
                        train_attack_ratio=train_attack_ratio, 
                        train_valid_ratio=train_valid_ratio
                    )
                    if dataset == "cifar":
                        model = resnet18(pretrained=False)
                        model.fc = nn.Linear(512, 100)
                    else:
                        model = TabularClassifier(num_features=num_features)
                    model = model.to(device)
                    criterion = nn.CrossEntropyLoss(reduction='none')
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    
                    start_time_mmd = time.perf_counter()
                    for epoch in range(epochs):
                        if dataset in ["purchase", "texas"]:
                            train_classifier_data_tensor = torch.from_numpy(train_classifier_data).type(torch.FloatTensor)
                            train_classifier_label_tensor = torch.from_numpy(train_classifier_label).type(torch.LongTensor)
    
                            train_attack_data_tensor = torch.from_numpy(train_attack_data).type(torch.FloatTensor)
                            train_attack_label_tensor = torch.from_numpy(train_attack_label).type(torch.LongTensor)
    
                            valid_data_tensor = torch.from_numpy(valid_data).type(torch.FloatTensor)
                            valid_label_tensor = torch.from_numpy(valid_label).type(torch.LongTensor)
    
                            test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
                            test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)
                        elif dataset == "cifar":
                            train_classifier_data_tensor = train_classifier_data.type(torch.FloatTensor)
                            train_classifier_label_tensor = train_classifier_label.type(torch.LongTensor)
    
                            train_attack_data_tensor = train_attack_data.type(torch.FloatTensor)
                            train_attack_label_tensor = train_attack_label.type(torch.LongTensor)
    
                            valid_data_tensor = valid_data.type(torch.FloatTensor)
                            valid_label_tensor = valid_label.type(torch.LongTensor)
    
                            test_data_tensor = test_data.type(torch.FloatTensor)
                            test_label_tensor = test_label.type(torch.LongTensor)
                        else:
                            raise ValueError("unhandled dataset")
                        
                        r = np.arange(len(train_classifier_data_tensor))
                        np.random.shuffle(r)
                        train_classifier_data_tensor = train_classifier_data_tensor[r]
                        train_classifier_label_tensor = train_classifier_label_tensor[r]
                        
                        r = np.arange(len(train_attack_data_tensor))
                        np.random.shuffle(r)
                        train_attack_data_tensor = train_attack_data_tensor[r]
                        train_attack_label_tensor = train_attack_label_tensor[r]
    
                        # train with weights
                        train_loss, train_acc = train(
                            train_classifier_data_tensor, train_classifier_label_tensor, model, criterion, optimizer, 
                            batch_size, epoch, device, 
                            mmd_weight=mmd_weight, mmd_scale=mmd_scale, start_mmd_epoch=start_mmd_epoch, 
                            ref_data=train_attack_data_tensor, ref_labels=train_attack_label_tensor, 
                            unique_labels=True, mmd_ref_term=True
                        )
    
                    end_time_mmd = time.perf_counter()
                    training_time = end_time_mmd - start_time_mmd
    #                 print(f"Dataset: {dataset}, Seed: {random_seed}, Time: {training_time}")
                    time_per_epoch_mmd[dataset].append(training_time)
        print(f"Algorithm: MMD - Dataset: {dataset}, Mean Training Time: {np.mean(time_per_epoch_mmd[dataset])}")