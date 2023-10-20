from models import *
from train_utils import *
from eval_utils import *
from train import *
from tqdm import tqdm

def execute_adv_reg(test_one_seed):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    run_datasets = ["purchase", "texas", "cifar"]
    if test_one_seed:
        random_seeds = [5]
    else:
        random_seeds = list(np.arange(5, 15))   
    att_warmup_epochs = 0
    verbose = False

    alphas = {
        "cifar": [1e-6, 1e-3, 1e-1, 1.],
        "purchase": [1., 2., 3., 6., 10., 20.],
        "texas": [1., 2., 3., 6., 10., 20.]
    }
    train_params_list = [
        {"training_style": "coin_flip", "att_loss": "mse", "non_member_loss_term": True, "ref_to_train_ratio": 1.,
         "int_epochs_att": 20, "int_epochs_clf": 1, "batch_size_clf": 128, "batch_size_att": 128},
        {"training_style": "coin_flip", "att_loss": "mse", "non_member_loss_term": False, "ref_to_train_ratio": 1.,
         "int_epochs_att": 20, "int_epochs_clf": 1, "batch_size_clf": 128, "batch_size_att": 128},
    #     {"training_style": "standard", "att_loss": "bce", "non_member_loss_term": True, "ref_to_train_ratio": 1.,
    #      "int_epochs_att": 20, "int_epochs_clf": 1, "batch_size_clf": 128, "batch_size_att": 128},
    #     {"training_style": "standard", "att_loss": "bce", "non_member_loss_term": False, "ref_to_train_ratio": 1.,
    #      "int_epochs_att": 20, "int_epochs_clf": 1, "batch_size_clf": 128, "batch_size_att": 128},   
    #     {"training_style": "code", "att_loss": "mse", "non_member_loss_term": False, "ref_to_train_ratio": 1.,
    #      "int_epochs_att": 76, "int_epochs_clf": 76, "batch_size_clf": 128, "batch_size_att": 128},
    ]

    
    total_experiments = len(run_datasets) * len(train_params_list) * len(random_seeds) * np.sum([len(alphas[d]) for d in run_datasets])
    experiment_index = 0

    for dataset in run_datasets:
        for train_params in train_params_list:
            if train_params["training_style"] == "code":
                use_validation = True
            else:
                use_validation = True
            
            # get data segment proportions
            if dataset == "texas":
                num_features = 6169
                ext_epochs = 20
                data_is_numpy = True
                if train_params["training_style"] == "code":
                    train_classifier_ratio, train_attack_ratio, train_valid_ratio = 0.1485222040695084, 0.3, 0.2
                    num_batches_att = 50
                else:
                    train_classifier_ratio = 0.15
                    train_test_ratio = 0.4
                    train_attack_ratio = train_classifier_ratio * train_params["ref_to_train_ratio"]
                    train_valid_ratio = 1 - train_classifier_ratio - train_attack_ratio - train_test_ratio
                    num_batches_att = None
            elif dataset == "purchase":
                num_features = 600
                ext_epochs = 40
                data_is_numpy = True
                if train_params["training_style"] == "code":   
                    train_classifier_ratio, train_attack_ratio, train_valid_ratio = 0.1, 0.15, 0.25
                    num_batches_att = 52
                else:
                    train_classifier_ratio = 0.1
                    train_test_ratio = 0.4
                    train_attack_ratio = train_classifier_ratio * train_params["ref_to_train_ratio"]
                    train_valid_ratio = 1 - train_classifier_ratio - train_attack_ratio - train_test_ratio
                    num_batches_att = None
            elif dataset == "cifar":
                train_classifier_ratio, train_attack_ratio, train_valid_ratio = None, None, None
                data_is_numpy = False
                ext_epochs = 30
                num_batches_att = None
            else:
                raise ValueError("not handled dataset")
            
            # set run name
            if train_params["non_member_loss_term"]:
                f_tag = "nf"
            else:
                f_tag = "of"
            run_name = f'{train_params["training_style"]}-{train_params["att_loss"]}-{f_tag}-{train_params["ref_to_train_ratio"]}-{ext_epochs}-{train_params["int_epochs_att"]}-{train_params["int_epochs_clf"]}'      
            if train_params["batch_size_clf"] != 128:
                run_name += f"-bs{train_params['batch_size_clf']}"   
            
            for alpha in alphas[dataset]:
                for random_seed in random_seeds:    
                    if random_seed in [5, 10]:
                        load_randomization = True
                    else:
                        load_randomization = False
    
                    print(f"Running Experiment {experiment_index}/{total_experiments} - Dataset: {dataset}, Random Seed: {random_seed}, Run Name: {run_name}") 
                    experiment_index += 1
    
                    # set random seed
                    set_seed(random_seed)
    
                    # set training params
                    best_valid_acc_state_dict = None
                    best_total_valid_loss_state_dict = None
                    best_valid_acc = 0.
                    best_valid_acc_epoch = -1
                    best_valid_loss = 1e5
                    best_valid_loss_epoch = -1
    
                    # load data
                    train_classifier_data, train_classifier_label, train_attack_data, train_attack_label, valid_data, valid_label, test_data, test_label = load_data(
                        dataset=dataset, load_randomization=load_randomization, use_validation=use_validation,
                        train_classifier_ratio=train_classifier_ratio, 
                        train_attack_ratio=train_attack_ratio, 
                        train_valid_ratio=train_valid_ratio
                    )
    
                    # instantiate training objects
                    if dataset == "cifar":
                        model = resnet18(pretrained=False)
                        model.fc = nn.Linear(512, 100)
                    else:
                        model = TabularClassifier(num_features=num_features)
                    model = torch.nn.DataParallel(model).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    attack_model = InferenceAttack_HZ(100)
                    attack_model = torch.nn.DataParallel(attack_model).to(device)
                    if train_params["training_style"] == "code":
                        attack_criterion = nn.MSELoss()
                        squared_loss = False
                        log_loss = False
                    else:
                        if train_params["att_loss"] == "mse":                    
                            attack_criterion = nn.MSELoss()
                            squared_loss = True
                            log_loss = False
                            if train_params["training_style"] == "code":
                                squared_loss = False
                        elif train_params["att_loss"] == "bce":
                            attack_criterion = nn.BCELoss()
                            squared_loss = False
                            log_loss = True
                        else:
                            raise ValueError(f"unhandled attack loss: {train_params['att_loss']}")
                    attack_optimizer = optim.Adam(attack_model.parameters(), lr=0.0001)
                    
                    for epoch in tqdm(range(ext_epochs)):
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
    
    #                     print('\nEpoch: [%d | %d]' % (epoch, ext_epochs))
    
                        if epoch == 0:                 
                            train_loss, train_acc = train(
                                train_classifier_data_tensor, train_classifier_label_tensor, model, criterion, optimizer, 
                                train_params["batch_size_clf"], epoch, device
                            )
    
                            for i in range(att_warmup_epochs):
                                train_attack("standard",
                                    train_classifier_data_tensor, train_classifier_label_tensor, 
                                    train_attack_data_tensor, train_attack_label_tensor, 
                                    model, attack_model, criterion, attack_criterion, optimizer, attack_optimizer,
                                    train_params["batch_size_att"], epoch, device, None, None, i
                                )
                                
                        else:
                            mi_losses = []
                            if train_params["training_style"] == "standard":
                                # train attack model
                                for i in range(train_params["int_epochs_att"]):
                                    at_loss, at_acc = train_attack(train_params["training_style"],
                                        train_classifier_data_tensor, train_classifier_label_tensor, 
                                        train_attack_data_tensor, train_attack_label_tensor, 
                                        model, attack_model, criterion, attack_criterion, optimizer, attack_optimizer,
                                        train_params["batch_size_att"], epoch, device, None, None, i)
    
                                # these values correspond to the best attack model vs. previous classifier
                                with torch.no_grad():
                                    # MI loss - members
                                    tr_mi_loss_o, tr_mi_loss_n = calculate_mi_loss(
                                        train_classifier_data_tensor, train_classifier_label_tensor, model, attack_model, train_params["att_loss"], is_ref_data=False)  
                                    # MI loss - reference - non-members
                                    at_mi_loss_o, at_mi_loss_n = calculate_mi_loss(
                                        train_attack_data_tensor, train_attack_label_tensor, model, attack_model, train_params["att_loss"], is_ref_data=True)
                                    # MI loss - non-members - test
                                    te_mi_loss_o, te_mi_loss_n = calculate_mi_loss(
                                        test_data_tensor, test_label_tensor, model, attack_model, train_params["att_loss"], is_ref_data=True)
                                    # MI loss - non-members - validation
                                    if use_validation:
                                        vd_mi_loss_o, vd_mi_loss_n = calculate_mi_loss(
                                            valid_data_tensor, valid_label_tensor, model, attack_model, train_params["att_loss"], is_ref_data=True)
                                    else:
                                        vd_mi_loss_o, vd_mi_loss_n = None, None
    
                                # train classifier
                                for i in range(train_params["int_epochs_clf"]):
                                    tr_loss, tr_acc = train_privately(
                                        training_style=train_params["training_style"], train_data=train_classifier_data_tensor, labels=train_classifier_label_tensor,                                  
                                        model=model, inference_model=attack_model, criterion=criterion, optimizer=optimizer, 
                                        batch_size=train_params["batch_size_clf"], epoch=epoch, device=device, 
                                        num_batchs=None, skip_batch=None, alpha=alpha, 
                                        attack_data=train_attack_data_tensor, attack_labels=train_attack_label_tensor, 
                                        i=i, squared_loss=squared_loss, log_loss=log_loss, non_member_loss_term=train_params["non_member_loss_term"]
                                    )
    
                            elif train_params["training_style"] == "coin_flip":
                                clf_selections = 0
                                internal_epochs = train_params["int_epochs_att"] + train_params["int_epochs_clf"]
                                for r in range(internal_epochs):
                                    clf_train_proportion = train_params["int_epochs_clf"] / internal_epochs
                                    model_train_sequence = []
                                    # batch size for classifier and attack model must be equal
                                    for i in range(int(train_classifier_data.shape[0] / train_params["batch_size_clf"])):
                                        model_to_train = np.random.binomial(n=1, p=clf_train_proportion)
                                        model_train_sequence.append(model_to_train)
                                        if model_to_train == 0:
                                            at_loss, at_acc = train_attack(train_params["training_style"],
                                                train_classifier_data_tensor, train_classifier_label_tensor, 
                                                train_attack_data_tensor, train_attack_label_tensor, 
                                                model, attack_model, criterion, attack_criterion, optimizer, attack_optimizer,
                                                train_params["batch_size_att"], epoch, device, None, None, i)
                                        else:
                                            tr_loss, tr_acc = train_privately(
                                                training_style=train_params["training_style"], train_data=train_classifier_data_tensor, labels=train_classifier_label_tensor,                                  
                                                model=model, inference_model=attack_model, criterion=criterion, optimizer=optimizer, 
                                                batch_size=train_params["batch_size_clf"], epoch=epoch, device=device, 
                                                num_batchs=None, skip_batch=None, alpha=alpha, 
                                                attack_data=train_attack_data_tensor, attack_labels=train_attack_label_tensor, 
                                                i=i, squared_loss=squared_loss, log_loss=log_loss, non_member_loss_term=train_params["non_member_loss_term"]
                                            )
                                            clf_selections += 1
                                if verbose:
                                    print("clf selections", clf_selections)
    
                            elif train_params["training_style"] == "code":
                                code_train_metrics = []
                                for i in range(76):
                                    at_loss, at_acc = train_attack(train_params["training_style"],
                                        train_classifier_data_tensor, train_classifier_label_tensor, train_attack_data_tensor,
                                        train_attack_label_tensor, model, attack_model, criterion, attack_criterion, optimizer, attack_optimizer,
                                        train_params["batch_size_att"], epoch, device, num_batches_att, (i*num_batches_att)%150)
                                    tr_loss, tr_acc = train_privately(
                                        training_style=train_params["training_style"], train_data=train_classifier_data_tensor, labels=train_classifier_label_tensor,                                  
                                        model=model, inference_model=attack_model, criterion=criterion, 
                                        optimizer=optimizer, batch_size=train_params["batch_size_clf"],
                                        epoch=epoch, device=device, num_batchs=2, skip_batch=(2*i)%152, alpha=alpha, 
                                        attack_data=train_attack_data_tensor, attack_labels=train_attack_label_tensor, 
                                        i=i, squared_loss=squared_loss, log_loss=log_loss, non_member_loss_term=train_params["non_member_loss_term"]
                                    )
                                    # after this value it skips all batches and returns None, None
                                    if dataset == "texas" and i == 38:
                                        code_train_metrics.append((tr_loss, tr_acc))
                                if dataset == "texas":
                                    tr_loss, tr_acc = code_train_metrics[0]
    
                            else:
                                raise ValueError(f"unhandled training style: {training_style}")                            
                                
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
    # #                     print(f'Gap Attack: {1/2 + (train_acc / 100 - test_acc / 100) / 2}')
    
                        filename = f'seed{random_seed}/alpha{alpha}/train-{run_name}'
    
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


def runtime_adv_reg():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    run_datasets = ["purchase", "texas", "cifar"]
    random_seeds = np.arange(5, 15)
    att_warmup_epochs = 0
    verbose = False

    alphas = {
        "cifar": [1.],
        "purchase": [1.],
        "texas": [1.]
    }
    train_params_list = [
        {"training_style": "coin_flip", "att_loss": "mse", "non_member_loss_term": False, "ref_to_train_ratio": 1.,
         "int_epochs_att": 20, "int_epochs_clf": 1, "batch_size_clf": 128, "batch_size_att": 128},
    ]

    time_per_epoch_advreg = {"cifar": [], "purchase": [], "texas": []}
    for dataset in run_datasets:
        for train_params in train_params_list:
            if train_params["training_style"] == "code":
                use_validation = True
            else:
                use_validation = True
            
            # get data segment proportions
            if dataset == "texas":
                num_features = 6169
                ext_epochs = 1
                data_is_numpy = True
                if train_params["training_style"] == "code":
                    train_classifier_ratio, train_attack_ratio, train_valid_ratio = 0.1485222040695084, 0.3, 0.2
                    num_batches_att = 1
                else:
                    train_classifier_ratio = 0.15
                    train_test_ratio = 0.4
                    train_attack_ratio = train_classifier_ratio * train_params["ref_to_train_ratio"]
                    train_valid_ratio = 1 - train_classifier_ratio - train_attack_ratio - train_test_ratio
                    num_batches_att = None
            elif dataset == "purchase":
                num_features = 600
                ext_epochs = 1
                data_is_numpy = True
                if train_params["training_style"] == "code":   
                    train_classifier_ratio, train_attack_ratio, train_valid_ratio = 0.1, 0.15, 0.25
                    num_batches_att = 52
                else:
                    train_classifier_ratio = 0.1
                    train_test_ratio = 0.4
                    train_attack_ratio = train_classifier_ratio * train_params["ref_to_train_ratio"]
                    train_valid_ratio = 1 - train_classifier_ratio - train_attack_ratio - train_test_ratio
                    num_batches_att = None
            elif dataset == "cifar":
                train_classifier_ratio, train_attack_ratio, train_valid_ratio = None, None, None
                data_is_numpy = False
                ext_epochs = 30
                num_batches_att = None
            else:
                raise ValueError("not handled dataset")
            
            # set run name
            if train_params["non_member_loss_term"]:
                f_tag = "nf"
            else:
                f_tag = "of"
            run_name = f'{train_params["training_style"]}-{train_params["att_loss"]}-{f_tag}-{train_params["ref_to_train_ratio"]}-{ext_epochs}-{train_params["int_epochs_att"]}-{train_params["int_epochs_clf"]}'      
            if train_params["batch_size_clf"] != 128:
                run_name += f"-bs{train_params['batch_size_clf']}"   
            
            for alpha in alphas[dataset]:
                for random_seed in random_seeds:    
                    if random_seed in [5, 10]:
                        load_randomization = True
                    else:
                        load_randomization = False
    
                    # print(dataset, run_name, alpha, random_seed)
    
                    # set random seed
                    set_seed(random_seed)
    
                    # set training params
                    best_valid_acc_state_dict = None
                    best_total_valid_loss_state_dict = None
                    best_valid_acc = 0.
                    best_valid_acc_epoch = -1
                    best_valid_loss = 1e5
                    best_valid_loss_epoch = -1
    
                    # load data
                    train_classifier_data, train_classifier_label, train_attack_data, train_attack_label, valid_data, valid_label, test_data, test_label = load_data(
                        dataset=dataset, load_randomization=load_randomization, use_validation=use_validation,
                        train_classifier_ratio=train_classifier_ratio, 
                        train_attack_ratio=train_attack_ratio, 
                        train_valid_ratio=train_valid_ratio
                    )
    
                    # instantiate training objects
                    if dataset == "cifar":
                        model = resnet18(pretrained=False)
                        model.fc = nn.Linear(512, 100)
                    else:
                        model = TabularClassifier(num_features=num_features)
                    model = torch.nn.DataParallel(model).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    attack_model = InferenceAttack_HZ(100)
                    attack_model = torch.nn.DataParallel(attack_model).to(device)
                    if train_params["training_style"] == "code":
                        attack_criterion = nn.MSELoss()
                        squared_loss = False
                        log_loss = False
                    else:
                        if train_params["att_loss"] == "mse":                    
                            attack_criterion = nn.MSELoss()
                            squared_loss = True
                            log_loss = False
                            if train_params["training_style"] == "code":
                                squared_loss = False
                        elif train_params["att_loss"] == "bce":
                            attack_criterion = nn.BCELoss()
                            squared_loss = False
                            log_loss = True
                        else:
                            raise ValueError(f"unhandled attack loss: {train_params['att_loss']}")
                    attack_optimizer = optim.Adam(attack_model.parameters(), lr=0.0001)
                    
                    start_time_advreg = time.perf_counter()
                    for epoch in range(ext_epochs):
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
    
    #                     print('\nEpoch: [%d | %d]' % (epoch, ext_epochs))
    
                        mi_losses = []
                        if train_params["training_style"] == "standard":
                            # train attack model
                            for i in range(train_params["int_epochs_att"]):
                                at_loss, at_acc = train_attack(train_params["training_style"],
                                    train_classifier_data_tensor, train_classifier_label_tensor, 
                                    train_attack_data_tensor, train_attack_label_tensor, 
                                    model, attack_model, criterion, attack_criterion, optimizer, attack_optimizer,
                                    train_params["batch_size_att"], epoch, device, None, None, i)
    
                            # these values correspond to the best attack model vs. previous classifier
                            with torch.no_grad():
                                # MI loss - members
                                tr_mi_loss_o, tr_mi_loss_n = calculate_mi_loss(
                                    train_classifier_data_tensor, train_classifier_label_tensor, model, attack_model, train_params["att_loss"], is_ref_data=False)  
                                # MI loss - reference - non-members
                                at_mi_loss_o, at_mi_loss_n = calculate_mi_loss(
                                    train_attack_data_tensor, train_attack_label_tensor, model, attack_model, train_params["att_loss"], is_ref_data=True)
                                # MI loss - non-members - test
                                te_mi_loss_o, te_mi_loss_n = calculate_mi_loss(
                                    test_data_tensor, test_label_tensor, model, attack_model, train_params["att_loss"], is_ref_data=True)
                                # MI loss - non-members - validation
                                if use_validation:
                                    vd_mi_loss_o, vd_mi_loss_n = calculate_mi_loss(
                                        valid_data_tensor, valid_label_tensor, model, attack_model, train_params["att_loss"], is_ref_data=True)
                                else:
                                    vd_mi_loss_o, vd_mi_loss_n = None, None
    
                            # train classifier
                            for i in range(train_params["int_epochs_clf"]):
                                tr_loss, tr_acc = train_privately(
                                    training_style=train_params["training_style"], train_data=train_classifier_data_tensor, labels=train_classifier_label_tensor,                                  
                                    model=model, inference_model=attack_model, criterion=criterion, optimizer=optimizer, 
                                    batch_size=train_params["batch_size_clf"], epoch=epoch, device=device, 
                                    num_batchs=None, skip_batch=None, alpha=alpha, 
                                    attack_data=train_attack_data_tensor, attack_labels=train_attack_label_tensor, 
                                    i=i, squared_loss=squared_loss, log_loss=log_loss, non_member_loss_term=train_params["non_member_loss_term"]
                                )
    
                        elif train_params["training_style"] == "coin_flip":
                            clf_selections = 0
                            internal_epochs = train_params["int_epochs_att"] + train_params["int_epochs_clf"]
                            for r in range(internal_epochs):
                                clf_train_proportion = train_params["int_epochs_clf"] / internal_epochs
                                model_train_sequence = []
                                # batch size for classifier and attack model must be equal
                                for i in range(int(train_classifier_data.shape[0] / train_params["batch_size_clf"])):
                                    model_to_train = np.random.binomial(n=1, p=clf_train_proportion)
                                    model_train_sequence.append(model_to_train)
                                    if model_to_train == 0:
                                        at_loss, at_acc = train_attack(train_params["training_style"],
                                            train_classifier_data_tensor, train_classifier_label_tensor, 
                                            train_attack_data_tensor, train_attack_label_tensor, 
                                            model, attack_model, criterion, attack_criterion, optimizer, attack_optimizer,
                                            train_params["batch_size_att"], epoch, device, None, None, i)
                                    else:
                                        tr_loss, tr_acc = train_privately(
                                            training_style=train_params["training_style"], train_data=train_classifier_data_tensor, labels=train_classifier_label_tensor,                                  
                                            model=model, inference_model=attack_model, criterion=criterion, optimizer=optimizer, 
                                            batch_size=train_params["batch_size_clf"], epoch=epoch, device=device, 
                                            num_batchs=None, skip_batch=None, alpha=alpha, 
                                            attack_data=train_attack_data_tensor, attack_labels=train_attack_label_tensor, 
                                            i=i, squared_loss=squared_loss, log_loss=log_loss, non_member_loss_term=train_params["non_member_loss_term"]
                                        )
                                        clf_selections += 1
                            if verbose:
                                print("clf selections", clf_selections)
    
                        elif train_params["training_style"] == "code":
                            code_train_metrics = []
                            for i in range(76):
                                at_loss, at_acc = train_attack(train_params["training_style"],
                                    train_classifier_data_tensor, train_classifier_label_tensor, train_attack_data_tensor,
                                    train_attack_label_tensor, model, attack_model, criterion, attack_criterion, optimizer, attack_optimizer,
                                    train_params["batch_size_att"], epoch, device, num_batches_att, (i*num_batches_att)%150)
                                tr_loss, tr_acc = train_privately(
                                    training_style=train_params["training_style"], train_data=train_classifier_data_tensor, labels=train_classifier_label_tensor,                                  
                                    model=model, inference_model=attack_model, criterion=criterion, 
                                    optimizer=optimizer, batch_size=train_params["batch_size_clf"],
                                    epoch=epoch, device=device, num_batchs=2, skip_batch=(2*i)%152, alpha=alpha, 
                                    attack_data=train_attack_data_tensor, attack_labels=train_attack_label_tensor, 
                                    i=i, squared_loss=squared_loss, log_loss=log_loss, non_member_loss_term=train_params["non_member_loss_term"]
                                )
    
                        else:
                            raise ValueError(f"unhandled training style: {training_style}")
                    end_time_advreg = time.perf_counter()
                    training_time = end_time_advreg - start_time_advreg
    #                 print(f"Dataset: {dataset}, Seed: {random_seed}, Time: {training_time}")
                    time_per_epoch_advreg[dataset].append(training_time)
        print(f"Algorithm: AdvReg - Dataset: {dataset}, Mean Training Time: {np.mean(time_per_epoch_advreg[dataset])}")
                            
