from models import *
from train_utils import *
from eval_utils import *
from train import *
from tqdm import tqdm


def execute_werm(test_one_seed):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    run_datasets = ["purchase", "texas", "cifar"]
    if test_one_seed:
        random_seeds = [5]
    else:
        random_seeds = list(np.arange(5, 15))
    use_validation = True
    weight_props = {
        "purchase": [0.5, 0.7, 0.9, 0.97, 0.98, 1.],
        "texas": [0.5, 0.7, 0.9, 0.97, 0.999, 1.],
        "cifar": [0.5, 0.7, 0.9, 0.97, 0.9975, 1.],
    }
    ref_to_train_ratio = 1.0
    batch_sizes = [512, 128]

    total_experiments = len(run_datasets) * len(random_seeds) * np.sum([len(weight_props[d]) for d in run_datasets]) * len(batch_sizes)
    experiment_index = 0
    
    for dataset in run_datasets:
        for random_seed in random_seeds:
            for weight_prop in weight_props[dataset]:
                for batch_size in batch_sizes:                
                    set_seed(random_seed)
                    
                    if dataset == "texas":
                        epochs = 4
                        num_features = 6169
                        train_classifier_ratio = 0.15
                        train_test_ratio = 0.4
                        train_attack_ratio = train_classifier_ratio * ref_to_train_ratio
                        train_valid_ratio = 1 - train_classifier_ratio - train_attack_ratio - train_test_ratio
                        data_is_numpy = True
                    elif dataset == "purchase":
                        epochs = 20
                        num_features = 600
                        train_classifier_ratio = 0.1
                        train_test_ratio = 0.4
                        train_attack_ratio = train_classifier_ratio * ref_to_train_ratio
                        train_valid_ratio = 1 - train_classifier_ratio - train_attack_ratio - train_test_ratio
                        data_is_numpy = True
                    elif dataset == "cifar":
                        epochs = 25
                        train_classifier_ratio, train_attack_ratio, train_valid_ratio = None, None, None
                        data_is_numpy = False
                    else:
                        raise ValueError("not handled dataset")
                       
                    run_name = f"weight{weight_prop}-rttr{ref_to_train_ratio}"
                    if batch_size != 128:
                        run_name += f"-bs{batch_size}"
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
        #             print(model.features[0].weight)
                    model = torch.nn.DataParallel(model).to(device)
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
    
                        train_data_comb_tensor = torch.cat([train_classifier_data_tensor, train_attack_data_tensor])
                        train_label_comb_tensor = torch.cat([train_classifier_label_tensor, train_attack_label_tensor])
    
                        weight_prop_train_tensor = torch.ones(len(train_classifier_data_tensor)).to(device) * weight_prop
                        weight_prop_attack_tensor = torch.ones(len(train_attack_data_tensor)).to(device) * (1 - weight_prop)
    
                        weight_prop_comb_tensor = torch.cat([weight_prop_train_tensor, weight_prop_attack_tensor])
    
                        r = np.arange(len(train_data_comb_tensor))
                        np.random.shuffle(r)
    
                        train_data_comb_tensor = train_data_comb_tensor[r]
                        train_label_comb_tensor = train_label_comb_tensor[r]
                        weight_prop_comb_tensor = weight_prop_comb_tensor[r]
    
    #                     print('\nEpoch: [%d | %d]' % (epoch, epochs))
    
                        # train with weights
                        train_loss, train_acc = train(train_data_comb_tensor, train_label_comb_tensor, model, criterion, optimizer,
                                                      batch_size, epoch, device, loss_weights=weight_prop_comb_tensor)
    
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
    
    #                     print (f'Train Acc: {train_acc}, Ref Acc: {ref_acc}, Valid Acc: {valid_acc}, Test Acc: {test_acc}')
    #                     print (f'Train Loss: {train_loss}, Ref Loss: {ref_loss}, Valid Loss: {valid_loss}, Test Loss: {test_loss}')
    #                     print(f"Conf Attack Train: {conf_acc_train}, Conf Attack Ref: {conf_acc_ref}")              
        #                 print(f'Gap Attack: {1/2 + (train_acc / 100 - test_acc / 100) / 2}')
    
                        filename = f'seed{random_seed}/weighted-erm/train-{run_name}'
    
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


def runtime_werm():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    run_datasets = ["purchase", "texas", "cifar"]
    random_seeds = list(np.arange(5, 15))
    use_validation = True
    weight_props = {
        "purchase": [0.7],
        "texas": [0.7],
        "cifar": [0.7],
    }
    ref_to_train_ratio = 1.0
    batch_sizes = [512]

    time_per_epoch_werm = {"cifar": [], "purchase": [], "texas": []}
    for dataset in run_datasets:
        for random_seed in random_seeds:
            for weight_prop in weight_props[dataset]:
                for batch_size in batch_sizes:                
                    set_seed(random_seed)
                    
                    if dataset == "texas":
                        epochs = 1
                        num_features = 6169
                        train_classifier_ratio = 0.15
                        train_test_ratio = 0.4
                        train_attack_ratio = train_classifier_ratio * ref_to_train_ratio
                        train_valid_ratio = 1 - train_classifier_ratio - train_attack_ratio - train_test_ratio
                        data_is_numpy = True
                    elif dataset == "purchase":
                        epochs = 1
                        num_features = 600
                        train_classifier_ratio = 0.1
                        train_test_ratio = 0.4
                        train_attack_ratio = train_classifier_ratio * ref_to_train_ratio
                        train_valid_ratio = 1 - train_classifier_ratio - train_attack_ratio - train_test_ratio
                        data_is_numpy = True
                    elif dataset == "cifar":
                        epochs = 1
                        train_classifier_ratio, train_attack_ratio, train_valid_ratio = None, None, None
                        data_is_numpy = False
                    else:
                        raise ValueError("not handled dataset")
                       
                    run_name = f"weight{weight_prop}-rttr{ref_to_train_ratio}"
                    if batch_size != 128:
                        run_name += f"-bs{batch_size}"
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
        #             print(model.features[0].weight)
                    model = torch.nn.DataParallel(model).to(device)
                    criterion = nn.CrossEntropyLoss(reduction='none')
                    optimizer = optim.Adam(model.parameters(), lr=0.001)                
                    
                    start_time_werm = time.perf_counter()
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
    
                        train_data_comb_tensor = torch.cat([train_classifier_data_tensor, train_attack_data_tensor])
                        train_label_comb_tensor = torch.cat([train_classifier_label_tensor, train_attack_label_tensor])
    
                        weight_prop_train_tensor = torch.ones(len(train_classifier_data_tensor)).to(device) * weight_prop
                        weight_prop_attack_tensor = torch.ones(len(train_attack_data_tensor)).to(device) * (1 - weight_prop)
    
                        weight_prop_comb_tensor = torch.cat([weight_prop_train_tensor, weight_prop_attack_tensor])
    
                        r = np.arange(len(train_data_comb_tensor))
                        np.random.shuffle(r)
    
                        train_data_comb_tensor = train_data_comb_tensor[r]
                        train_label_comb_tensor = train_label_comb_tensor[r]
                        weight_prop_comb_tensor = weight_prop_comb_tensor[r]
    
                        # train with weights
                        train_loss, train_acc = train(train_data_comb_tensor, train_label_comb_tensor, model, criterion, optimizer,
                                                      batch_size, epoch, device, loss_weights=weight_prop_comb_tensor)
    
                    end_time_werm = time.perf_counter()
                    training_time = end_time_werm - start_time_werm
    #                 print(f"Dataset: {dataset}, Seed: {random_seed}, Time: {training_time}")
                    time_per_epoch_werm[dataset].append(training_time)
        print(f"Algorithm: WERM - Dataset: {dataset}, Mean Training Time: {np.mean(time_per_epoch_werm[dataset])}")