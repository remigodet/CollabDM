import os
import time
import copy
import argparse
import numpy as np
import torch

from client import Client
from utils import get_dataset_mtt, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, \
    partition_data, number_sign_augment, parser_bool

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet_GBN', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--num_clients', type=int, default=5, help='the number of clients participating in distillation')
    parser.add_argument('--frac', type=float, default=1.0, help='proportion of clients participating during each iteration')
    parser.add_argument('--beta', type=float, default=0.5, help='parameter for dirichlet distribution')
    parser.add_argument('--partition', type=str, default="dirichlet", help='distribution of data partition: dirichlet/iid')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=400, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=10.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=100, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result_test', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--optim', type=str, default='sgd', help='choice of optimizer')
    parser.add_argument('--aug_num', type=int, default=1, help='number of augmentations')

    parser_bool(parser, 'zca', False)
    parser_bool(parser, 'aug', False)

    args = parser.parse_args()
    args.method = 'DM'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if args.partition == 'iid':
        args.batch_real = 512

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # get evaluation architectures
    eval_it_pool = np.arange(0, args.Iteration+1, 50).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    # get dataset
    args.channel, args.im_size, args.num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset_mtt(
        args.dataset, args.data_path, args=args)
    print("Dataset %s and number of classes: %d"%(args.dataset, args.num_classes))

    print("Generating random seeds for DM models...")
    # generate random seeds for embeddings
    MAXINT = 0x7FFFFFFF
    seeds = np.random.randint(MAXINT, size=args.Iteration+1)
    num_participants = max(int(args.frac * args.num_clients), 1)
    client_seeds = [[] for c in range(args.num_clients)]
    for i in (range(args.Iteration+1)):
        idxs_clients = np.random.choice(range(args.num_clients), num_participants, replace=False)
        for idx in idxs_clients:
            client_seeds[idx].append((i, seeds[i]))

    ''' organize the real dataset '''

    # partition training data across clients
    client_data_idx = partition_data(dst_train, args)

    # initialize server data structures
    embeddings_real = {}
    image_syn = torch.Tensor([]).to('cpu')
    label_syn = torch.LongTensor([]).to('cpu')

    client_ipc = int(args.ipc/args.num_clients) + 1

    print("Running Client Algorithms...")
    # run client algorithms
    for idx in (range(args.num_clients)):
        print("// CLIENT: %d" % idx)
        data_idx = client_data_idx[idx]

        client = Client(dst_train, data_idx, args)

        # distill local data
        client_syn = client.distribution_matching_idm(client_ipc, args)
        client_syn_labels = torch.tensor([np.ones(client_ipc) * i for i in range(args.num_classes)], dtype=torch.long,
                                         requires_grad=False, device='cpu').view(-1)

        with torch.no_grad():
            client_embedding = client.compute_embeddings(client_seeds[idx], args)

        # 'Send' embeddings and local synthetic data to server
        embeddings_real[idx] = client_embedding

        image_syn = torch.cat((image_syn, client_syn), 0)
        label_syn = torch.cat((label_syn, client_syn_labels),0)

    ''' run server algorithm '''
    # sort synthetic images according to label
    indices_class = [[] for c in range(args.num_classes)]
    for i, lab in enumerate(label_syn):
        indices_class[lab].append(i)

    print("\n SERVER algorithm")
    # initialize server synthetic data
    server_syn = torch.randn(size=(args.num_classes * args.ipc, args.channel, args.im_size[0], args.im_size[1]),
                             dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(args.num_classes)], dtype=torch.long,
                             requires_grad=False, device=args.device).view(-1)

    if args.init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(args.num_classes):
            # images already downscaled at local level
            idx_shuffle = np.random.permutation(indices_class[c])[:args.ipc]
            server_syn.data[c * args.ipc:(c + 1) * args.ipc] = image_syn[idx_shuffle].detach().data
    else:
        print('initialize synthetic data from random noise')

    # initialize the optimizer for synthetic data
    optimizer_img = torch.optim.SGD([server_syn, ], lr=args.lr_img, momentum=0.5)
    optimizer_img.zero_grad()

    print('%s training begins' % get_time())

    for it in (range(args.Iteration + 1)):

        # evaluate synthetic data
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
                args.model, model_eval, it))

                print('DSA augmentation strategy: \n', args.dsa_strategy)
                print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                accs = []
                for it_eval in range(args.num_eval):
                    eval_seed = int(time.time() * 1000) % 100000
                    net_eval = get_network(model_eval, args.channel, args.num_classes, eval_seed, args.im_size).to(args.device)  # get a random model
                    image_syn_eval, label_syn_eval = copy.deepcopy(server_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification

                    if args.aug:
                        image_syn_eval, label_syn_eval = number_sign_augment(image_syn_eval, label_syn_eval)
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval,testloader, args)
                    accs.append(acc_test)
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs), model_eval, np.mean(accs), np.std(accs)))

        # train synthetic data
        seed = seeds[it]
        loss_avg = 0

        if 'BN' not in args.model or args.model == 'ConvNet_GBN':

            # get client embeddings for this iteration
            client_outputs = [[] for _ in range(args.num_classes)]
            for cl in range(args.num_clients):
                cl_embeddings = embeddings_real[cl]
                if it in cl_embeddings:
                    # client participated in this iteration
                    for c in range(args.num_classes):
                        real_embedding = cl_embeddings[it][c]
                        # retrieve the embedding for each class
                        client_outputs[c].append(real_embedding)

            mean_client_outputs = [torch.tensor([]) for c in range(args.num_classes)]
            for c in range(args.num_classes):
                output = torch.stack(client_outputs[c])
                # get mean of embeddings taken across all clients in iteration 'it'
                mean_client_outputs[c] = torch.mean(output, dim=0).to(args.device)

            # compute difference in mean of embeddings
            for image_sign, image_temp in [['syn', server_syn]]:
                loss = torch.tensor(0.0).to(args.device)

                net = get_network(args.model, args.channel, args.num_classes, seed, args.im_size).to(args.device)  # get a random model
                net.train()
                for param in list(net.parameters()):
                    param.requires_grad = False

                embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed  # for GPU parallel

                for c in range(args.num_classes):
                    loss_c = torch.tensor(0.0).to(args.device)

                    img_syn = image_temp[c * args.ipc:(c + 1) * args.ipc].reshape((args.ipc, args.channel, args.im_size[0], args.im_size[1]))
                    lab_syn = label_syn[c * args.ipc:(c + 1) * args.ipc]
                    assert args.aug_num == 1

                    if args.aug:
                        img_syn, lab_syn = number_sign_augment(img_syn, lab_syn)

                    if args.dsa:
                        img_syn_list = list()
                        for aug_i in range(args.aug_num):
                            img_syn_list.append(DiffAugment(img_syn, args.dsa_strategy, seed=seed+c+aug_i, param=args.dsa_param))
                        img_syn = torch.cat(img_syn_list)

                    mean_output_real = mean_client_outputs[c]
                    output_syn = embed(img_syn)

                    loss_c += torch.sum((mean_output_real - torch.mean(output_syn, dim=0)) ** 2)

                    optimizer_img.zero_grad()
                    loss_c.backward()
                    optimizer_img.step()

                    loss += loss_c.item()

            if image_sign == 'syn':
                loss_avg += loss.item()

        loss_avg /= (args.num_classes)

        if it % 50 == 0:
            print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))


if __name__ == '__main__':
    main()


