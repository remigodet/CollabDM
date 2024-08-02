import gc
import time
import numpy as np
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
from utils import get_time, get_network, DiffAugment, downscale, number_sign_augment

class Client:
    def __init__(self, dst_train, dst_index, args):

        self.indices_class = [[] for c in range(args.num_classes)]

        # organise the real dataset
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in dst_index]
        labels_all = [dst_train[i][1] for i in dst_index]
        print("client data size: ", len(images_all))
        for i, lab in (enumerate(labels_all)):
            self.indices_class[lab].append(i)
        self.images_all = torch.cat(images_all, dim=0).detach().to('cpu')

    def get_images(self, c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
        return self.images_all[idx_shuffle]

    def compute_embeddings(self, seeds, args):
        embeddings = {}
        it = 0

        # iterate through seeds
        for pair in seeds:

            (embedding_ID, seed) = pair

            # generate network from seed
            net = get_network(args.model, args.channel, args.num_classes, seed, args.im_size).to(args.device)
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            if 'BN' not in args.model or args.model == 'ConvNet_GBN':
                output_real = []
                for c in range(args.num_classes):
                    img_real = self.get_images(c, args.batch_real).to(args.device)

                    if args.dsa:
                        img_real_list = list()
                        for aug_i in range(args.aug_num):
                            aug_seed = seed + c + aug_i
                            img_real_list.append(DiffAugment(img_real, args.dsa_strategy, seed=aug_seed, param=args.dsa_param))
                        img_real = torch.cat(img_real_list)

                    data_embedding = net.embed(img_real).detach()

                    embedding_mean = torch.mean(data_embedding, dim=0).detach().to('cpu')
                    output_real.append(embedding_mean)
                    del img_real
                    del data_embedding

                embeddings[embedding_ID] = output_real

            it += 1

            del net
            gc.collect()
            torch.cuda.empty_cache()

        return embeddings

    def distribution_matching_idm(self, ipc, args):

        lr_img = 1.0
        batch_real = 256
        iteration = 400

        # initialize the synthetic data
        image_syn = torch.randn(size=(args.num_classes*ipc, args.channel, args.im_size[0], args.im_size[1]),
                                dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(ipc) * i for i in range(args.num_classes)], dtype=torch.long,
                                 requires_grad=False, device=args.device).view(-1)

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(args.num_classes):
                if not args.aug:
                    image_syn.data[c * ipc:(c + 1) * ipc] = self.get_images(c, ipc).detach().data
                else:
                    half_size = args.im_size[0] // 2
                    image_syn.data[c * ipc:(c + 1) * ipc, :, :half_size, :half_size] = downscale(
                        self.get_images(c, ipc), 0.5).detach().data
                    image_syn.data[c * ipc:(c + 1) * ipc, :, half_size:, :half_size] = downscale(
                        self.get_images(c, ipc), 0.5).detach().data
                    image_syn.data[c * ipc:(c + 1) * ipc, :, :half_size, half_size:] = downscale(
                        self.get_images(c, ipc), 0.5).detach().data
                    image_syn.data[c * ipc:(c + 1) * ipc, :, half_size:, half_size:] = downscale(
                        self.get_images(c, ipc), 0.5).detach().data
        else:
            print('initialize synthetic data from random noise')

        ''' training '''
        if args.optim == 'sgd':
            optimizer_img = torch.optim.SGD([image_syn, ], lr=lr_img, momentum=0.5)
        elif args.optim == 'adam':
            optimizer_img = torch.optim.Adam([image_syn, ], lr=lr_img)
        else:
            raise NotImplemented()
        optimizer_img.zero_grad()
        print('%s training begins' % get_time())

        # train syntheitc data
        for it in range(iteration + 1):

            loss_avg = 0

            # update synthetic data
            if 'BN' not in args.model or args.model == 'ConvNet_GBN':  # for ConvNet
                for image_sign, image_temp in [['syn', image_syn]]:
                    loss = torch.tensor(0.0).to(args.device)
                    seed = int(time.time() * 1000) % 100000
                    net = get_network(args.model, args.channel, args.num_classes, seed, args.im_size).to(args.device)  # get a random model
                    net.train()
                    for param in list(net.parameters()):
                        param.requires_grad = False

                    embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed  # for GPU parallel

                    for c in range(args.num_classes):
                        loss_c = torch.tensor(0.0).to(args.device)
                        img_real = self.get_images(c, batch_real).to(device = args.device)
                        img_syn = image_temp[c * ipc:(c + 1) * ipc].reshape(
                            (ipc, args.channel, args.im_size[0], args.im_size[1]))
                        lab_syn = label_syn[c * ipc:(c + 1) * ipc]

                        assert args.aug_num == 1

                        if args.aug:
                            img_syn, lab_syn = number_sign_augment(img_syn, lab_syn)

                        if args.dsa:
                            img_real_list = list()
                            img_syn_list = list()
                            for aug_i in range(args.aug_num):
                                seed = int(time.time() * 1000) % 100000
                                img_real_list.append(DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param))
                                img_syn_list.append(DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param))
                            img_real = torch.cat(img_real_list)
                            img_syn = torch.cat(img_syn_list)

                        output_real = embed(img_real).detach()
                        output_syn = embed(img_syn)

                        loss_c += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)
                        if args.aug:
                            loss_c += torch.sum((torch.std(output_real, dim=0) - torch.std(output_syn, dim=0)) ** 2)

                        optimizer_img.zero_grad()
                        loss_c.backward()
                        optimizer_img.step()

                        loss += loss_c.item()

                if image_sign == 'syn':
                    loss_avg += loss.item()

            loss_avg /= (args.num_classes)

            if it % 100 == 0:
                print('%s iter = %04d, loss = syn:%.4f' % (get_time(), it, loss_avg))

        return image_syn.detach().cpu()
