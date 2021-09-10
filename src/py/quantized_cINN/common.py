"""
common classes and functions
partly helpers

depends on: {}
based on:   https://github.com/VLL-HD/conditional_INNs
"""
__all__ = ["baseCONFIG",                  # config baseclass
           "Data", "one_hot",             # MNIST structure and cond.
           "Visualizer", "img_tile",      # helpers for visualization
           "show_training_data",
           "sample_outputs",              # reverse generation
           ]

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.functional import avg_pool2d, interpolate

from sklearn.decomposition import PCA


from tqdm import tqdm
from time import time

########################################################################

class baseCONFIG:
    """
    Namspace for configuration. Baseclass with unusable parameters.
    """

    @classmethod
    def str(cls):
        string = ""
        string += "==="*30 + "\n"
        for p, v in cls.__dict__.items():
            if p[0] == "_" or p == "str":
                continue
            string += f"  {p:25}\t{v}\n"
        string += "==="*30 + "\n"
        return string

    @classmethod
    def exists(cls, attr):
        return attr in cls.__dict__

########################################################################

class MNISTData:

    def __init__(self, config: object):
        from torch.utils.data import Dataset, DataLoader, TensorDataset
        import torchvision.transforms as T
        import torchvision.datasets as D

        self.c = config

        self.train_data = D.MNIST(self.c.mnist_data,
                                  train=True,
                                  download=True,
                                  transform=T.Compose([T.ToTensor(), self.normalize]))
        self.test_data = D.MNIST(self.c.mnist_data,
                                 train=False,
                                 download=True,
                                 transform=T.Compose([T.ToTensor(), self.normalize]))

        # Sample a fixed batch of 1024 validation examples
        self.val_x, self.val_l = zip(*list(self.train_data[i] for i in range(1024)))
        self.val_x = torch.stack(self.val_x, 0).to(config.device)
        self.val_l = torch.LongTensor(self.val_l).to(config.device)

        # Exclude the validation batch from the training data
        self.train_data.data = self.train_data.data[1024:]
        self.train_data.targets = self.train_data.targets[1024:]

        # Add the noise-augmentation to the (non-validation) training data:
        augm_func = lambda x: x + self.c.add_image_noise * torch.randn_like(x)
        self.train_data.transform = T.Compose([self.train_data.transform, augm_func])

        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.c.batch_size,
                                       shuffle=True,
                                       num_workers=self.c.n_workers,
                                       pin_memory=True, drop_last=True)
        self.test_loader = DataLoader(self.test_data,
                                      batch_size=self.c.batch_size,
                                      shuffle=False,
                                      num_workers=self.c.n_workers,
                                      pin_memory=True, drop_last=True)

    def unnormalize(self, x):
        return x * self.c.data_std + self.c.data_mean

    def normalize(self, x):
        return (x - self.c.data_mean) / self.c.data_std


class MNISTDataPreprocessed:

    def __init__(self, config: object):
        from torch.utils.data import Dataset, DataLoader, TensorDataset
        import torchvision.transforms as T
        import torchvision.datasets as D
        import torch.nn as nn

        self.c = config

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.train_data = D.MNIST(self.c.mnist_data,
                                  train=True,
                                  download=True,
                                  transform=T.Compose([T.ToTensor(), self.maxpool]))
        self.test_data = D.MNIST(self.c.mnist_data,
                                 train=False,
                                 download=True,
                                 transform=T.Compose([T.ToTensor(), self.maxpool]))

        # Sample a fixed batch of 1024 validation examples
        self.val_x, self.val_l = zip(*list(self.train_data[i] for i in range(1024)))
        self.val_x = torch.stack(self.val_x, 0).to(config.device)
        self.val_l = torch.LongTensor(self.val_l).to(config.device)

        # Exclude the validation batch from the training data
        self.train_data.data = self.train_data.data[1024:]
        self.train_data.targets = self.train_data.targets[1024:]

        # Add the noise-augmentation to the (non-validation) training data:
        #augm_func = lambda x: x + self.c.add_image_noise * torch.randn_like(x)
        #self.train_data.transform = T.Compose([self.train_data.transform, augm_func])

        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.c.batch_size,
                                       shuffle=True,
                                       num_workers=self.c.n_workers,
                                       pin_memory=True, drop_last=True)
        self.test_loader = DataLoader(self.test_data,
                                      batch_size=self.c.batch_size,
                                      shuffle=False,
                                      num_workers=self.c.n_workers,
                                      pin_memory=True, drop_last=True)


def one_hot(labels, out=None, scale=1.):
    '''
    Convert LongTensor labels (contains labels 0-9), to a one hot vector.
    Can be done in-place using the out-argument (faster, re-use of GPU memory)
    '''
    if out is None:
        out = torch.zeros(labels.shape[0], 10).to(labels.device)
    else:
        out.zeros_()

    out.scatter_(dim=1, index=labels.view(-1,1), value=scale)
    return out

########################################################################

class Visualizer:
    def __init__(self, config):
        loss_labels = config.loss_names
        try:
            loss_labels.__iter__
        except AttributeError:
            loss_labels = [loss_labels]

        self.n_losses = len(loss_labels)
        self.loss_labels = loss_labels
        self.counter = 0
        self.start_time = time()

        header = 'Epoch\tTime'
        for l in loss_labels:
            header += f'\t\t{l:s}'

        print(header)

    def update_losses(self, losses, *args):
        try:
            losses.__iter__
        except AttributeError:
            losses = [losses]

        print('\r', '    '*20, end='')
        line = f'\r{self.counter:d}\t{(time() - self.start_time)/60:.2f}'
        for l in losses:
            line += f'\t\t{l:.4f}'

        print(line)
        self.counter += 1

    def update_images(self, *args):
        pass

    def update_hist(self, *args):
        pass

    def close(self):
        pass

class LiveVisualizer(Visualizer):

    def __init__(self, config):
        super().__init__(config)

        self.dir = config.save_dir

        # loss plots
        self.losses_hist = []
        self.l_fig, self.l_axes = plt.subplots(nrows=self.n_losses,
                                               ncols=1,
                                               figsize=(5, 3*self.n_losses))

    def update_losses(self, losses):
        super().update_losses(losses)

        self.losses_hist.append(losses)

        if self.n_losses > 1:
            for i, ax in enumerate(self.l_axes):
                x = np.arange(self.counter)
                y = np.array(self.losses_hist)[i, :]
                ax.clear()
                ax.plot(x, y)
                ax.set_title(f"Loss '{self.loss_labels[i]}'")
                ax.set_xlabel("Epoch")
        else:
            x = np.arange(self.counter)
            y = np.array(self.losses_hist)
            self.l_axes.clear()
            self.l_axes.plot(x, y)
            self.l_axes.set_title(f"Loss '{self.loss_labels[0]}'")
            self.l_axes.set_xlabel("Epoch")

        self.l_fig.tight_layout()
        self.l_fig.savefig(f"{self.dir}/LiveVisualizer_loss.pdf")

#        def update_images(self, *img_list):

#            w = img_list[0].shape[2]
#            k = 0
#            k_img = 0

#            show_img = np.zeros((3, w*n_imgs, w*n_imgs), dtype=np.uint8)
#            img_list_np = []
#            for im in img_list:
#                im_np = im.cpu().data.numpy()
#                img_list_np.append(np.clip((255. * im_np), 0, 255).astype(np.uint8))

#            for i in range(n_imgs):
#                for j in range(n_imgs):
#                    show_img[:, w*i:w*i+w, w*j:w*j+w] = img_list_np[k][k_img]

#                    k += 1
#                    if k >= len(img_list_np):
#                        k = 0
#                        k_img += 1

#            show_img = zoom(show_img, (1., c.preview_upscale, c.preview_upscale), order=0)

#            self.viz.image(show_img, win = self.imgs)

#        def update_hist(self, data):
#            for i in range(n_plots):
#                for j in range(n_plots):
#                    try:
#                        self.axes[i,j].clear()
#                        self.axes[i,j].hist(data[:, i*n_plots + j], bins=20, histtype='step')
#                    except ValueError:
#                        pass

#            self.fig.tight_layout()
#            self.viz.matplot(self.fig, win=self.hist)

    def close(self):
        self.l_fig.close()
"""
visualizer = Visualizer(c.loss_names)

def show_loss(losses, logscale=True):
    visualizer.update_losses(losses, logscale)

def show_imgs(*imgs):
    visualizer.update_images(*imgs)

def show_hist(data):
    visualizer.update_hist(data.data)

def close():
    visualizer.close()
"""

def img_tile(imgs, row_col = None, transpose = False, channel_first=True, channels=3):
    '''tile a list of images to a large grid.
    imgs:       iterable of images to use
    row_col:    None (automatic), or tuple of (#rows, #columns)
    transpose:  Wheter to stitch the list of images row-first or column-first
    channel_first: if true, assume images with CxWxH, else WxHxC
    channels:   3 or 1, number of color channels '''
    if row_col == None:
        sqrt = np.sqrt(len(imgs))
        rows = np.floor(sqrt)
        delt = sqrt - rows
        cols = np.ceil(rows + 2*delt + delt**2 / rows)
        rows, cols = int(rows), int(cols)
    else:
        rows, cols = row_col

    if channel_first:
        h, w = imgs[0].shape[1], imgs[0].shape[2]
    else:
        h, w = imgs[0].shape[0], imgs[0].shape[1]

    show_im = np.zeros((rows*h, cols*w, channels))

    if transpose:
        def iterator():
            for i in range(rows):
                for j in range(cols):
                    yield i, j

    else:
        def iterator():
            for j in range(cols):
                for i in range(rows):
                    yield i, j

    k = 0
    for i, j in iterator():

            im = imgs[k]
            if channel_first:
                im = np.transpose(im, (1, 2, 0))

            show_im[h*i:h*i+h, w*j:w*j+w] = im

            k += 1
            if k == len(imgs):
                break

    return np.squeeze(show_im)

def show_training_data(digit, n_imgs, save_as=None):
    '''Show some validation images (if you want to look for interesting examples etc.)
    digit:      int 0-9, show images of this digit
    n_imgs:     show this many images
    save_as:    None, or filename, to save the image file'''
    imgs = []
    while len(imgs) < n_imgs ** 2:
        color, label, img = next(iter(data.train_loader))
        imgs += list(color[label==digit])

    img_show = img_tile(imgs, (n_imgs, n_imgs))
    plt.figure()
    plt.imshow(img_show)
    if save_as:
        plt.imsave(save_as,  img_show)

########################################################################

def sample_outputs(config):
    '''Produce a random latent vector with sampling temperature sigma'''
    output_dim = np.prod(config.img_size)
    return config.sampling_temperature * torch.randn(config.batch_size, output_dim).to(config.device)

def make_testcond(config):
    """
    :param config: config namespace/class
    :returns: test_cond
    """
    #test_cond = list(range(10))*(config.batch_size//10 + 1)
    #test_cond = torch.LongTensor(test_cond)
    #test_cond = test_l[:config.batch_size].to(config.device)
    test_cond = (list(range(10))*(config.batch_size//10 + 1))[:config.batch_size]
    test_cond = torch.LongTensor(test_cond).to(config.device)
    return test_cond

########################################################################

def interpolation(model, config, train_conf):
    """
    Interpolate between to random latent vectors.
    temp:       Sampling temperature
    n_steps:    Interpolation steps
    seeds:      Optional 2-tuple of seeds for the two random latent vectors
    save_as:    Optional filename to save the image.

    :param model:
    :param config:
    :param train_conf:
    :returns: None
    """

    test_cond = make_testcond(config)
    temp = config.sampling_temperature

    # determine 'n_steps'
    if config.exists("eval_steps_interploation"):
        n_steps = config.eval_steps_interploation
    else:
        n_steps = 12

    # sample first latend vector
    if config.exists("eval_seeds_interpolation"):
        torch.manual_seed(config.eval_seeds_interpolation[0])

    z_sample_0 = sample_outputs(config)
    z_0 = z_sample_0[0].expand_as(z_sample_0)

    if config.exists("eval_seeds_interpolation"):
        torch.manual_seed(config.eval_seeds_interpolation[1])

    z_sample_1 = sample_outputs(config)
    z_1 = z_sample_1[1].expand_as(z_sample_1)

    interpolation_steps = np.linspace(0., 1., n_steps, endpoint=True)
    interp_imgs = []

    for t in interpolation_steps:
        with torch.no_grad():
            im, _ = model.reverse_sample((1.-t) * z_0 + t * z_1, test_cond, jac=False)
            interp_imgs.extend(im[:10].cpu().data.numpy())
            #im = im.cpu().data.numpy()
            #interp_imgs.extend(np.array([im[i:i+1] for i in range(10)]))

    img_show = img_tile(interp_imgs, (10, n_steps), transpose=False, channels=1)

    plt.figure()
    plt.imshow(img_show, cmap='gray', vmin=0, vmax=1)
    plt.imsave(config.save_dir + "/interpolation_" + train_conf + ".png",
               img_show, cmap='gray', vmin=0, vmax=1)
    plt.close()

def style_transfer(model, data, index_ins, config, train_conf):
    """
    Perform style transfer as described in the cINN paper.
    index_in:   Index of the validation image to use for the transfer.
    save_as:    Optional filename to save the image.

    :param model:
    :param data:
    :param index_ins:
    :param config:
    :returns: None
    """

    # magic number:
    number = 9

    test_cond = make_testcond(config)

    x_test = []
    c_test = []
    for x, cc in data.test_loader:
        x_test.append(x)
        c_test.append(cc)
    x_test = torch.cat(x_test, dim=0).to(config.device)
    c_test = torch.cat(c_test, dim=0).to(config.device)

    for index_in in index_ins:

        if c_test[index_in] != number:
            return
        cond = torch.tensor([number]).to(config.device)

        with torch.no_grad():
            z_reference, _ = model(x_test[index_in:index_in+1], cond, jac=False)
            z_reference = torch.cat([z_reference]*10, dim=0)

            imgs_generated, _ = model.reverse_sample(z_reference, test_cond[:10], jac=False)
            #imgs_generated = imgs_generated.view(-1, 1, *config.img_size)

        ref_img = x_test[index_in, 0].cpu()

        img_show = img_tile(imgs_generated.cpu(), (1,10), transpose=False, channel_first=True, channels=1)

        plt.figure()
        plt.subplot(1,2,1)
        plt.xlabel(str(index_in))
        plt.imshow(ref_img, cmap='gray', vmin=0, vmax=1)
        plt.subplot(1,2,2)
        plt.imshow(img_show, cmap='gray', vmin=0, vmax=1)

        plt.imsave(config.save_dir + f"style_transfer_{index_in}_" + train_conf + ".png",
                   img_show, cmap='gray', vmin=0, vmax=1)
        plt.close()

def val_set_pca(model, data, config, I=0,C=9):
    """
    Perform PCA uing the latent codes of the validation set, to identify disentagled
    and semantic latent dimensions.
    I:
    C:
    save_as:    Optional filename to save the image.

    :param model:
    :param data:
    :param config:
    :param I: Index of the validation image to use for the transfer.
    :param C: Which digit to use (0-9).
    :returns: None
    """

    pca_dir = "pca_images"
    save_name = "digit_{}_component_{}.png"

    x_test = []
    c_test = []
    for x, cc in data.test_loader:
        x_test.append(x)
        c_test.append(cc)
    x_test = torch.cat(x_test, dim=0).to(config.device)
    c_test = torch.cat(c_test, dim=0).to(config.device)
    cond = torch.zeros(len(c_test), model.cond_size).cuda()
    cond.scatter_(1, c_test.view(-1,1), 1.)

    with torch.no_grad():
        z_all, _ = model(x_test, c_test, jac=False)
        z_all = z_all.cpu().data.numpy()

    pca = PCA(whiten=True)
    pca.fit(z_all)
    u = pca.transform(z_all)

    u_grid = np.zeros((config.pca_gridsize, u.shape[1]))

    U = np.linspace(-config.pca_extent,
                    config.pca_extent,
                    config.pca_gridsize)


    for i, u_i in enumerate(U):
        for j, w in config.pca_weights[I]:
            u_grid[i, j] = u_i * w

    z_grid = pca.inverse_transform(u_grid)
    _z_grid = torch.Tensor(z_grid).to(config.device)
    grid_cond = C * torch.ones(gridsize).cuda()

    with torch.no_grad():
        imgs, _ = model.reverse_sample(_z_grid, grid_cond, jac=False)
        imgs = imgs.view(-1, 1, 28, 28).cpu()

    img_show = img_tile(imgs, (1, gridsize), transpose=False, channel_first=True, channels=1)

    if not pca_dir in os.listdir(config.save_dir):
        os.mkdir(config.save_dir + "/" + pca_dir)

    filename = config.save_dir + "/" + pca_dir + "/" + save_name.format(C, I)
    plt.imsave(filename, img_show, cmap='gray', vmin=0, vmax=1)
    plt.imshow(img_show, cmap='gray', vmin=0, vmax=1)

def temperature(temp=None, rows=10, columns=24, save_as=None):
    """
    Show the effect of changing sampling temperature.
    temp:       If None, interpolate between 0 and 1.5 in `columns` steps.
                If float, use it as the sampling temperature.
    rows:       Number of rows (10=1 for each digit) to show
    columns:    Number of columns (interpolation steps for temperature)
    save_as:    Optional filename to save the image.
    """

    temperature_imgs = []
    temp_steps = np.linspace(0., 1.5, columns, endpoint=True)

    ticks = [ (i+0.5) * c.img_dims[1] for i in range(len(temp_steps))]
    labels = [ '%.2f' % (s) for s in temp_steps ]

    for s in temp_steps:

        if temp is None:
            z_sample = sample_outputs(s)
        else:
            z_sample = sample_outputs(temp)

        z_sample[:] = z_sample[0]

        with torch.no_grad():
            temperature_imgs.append(model.model(z_sample, test_cond, rev=True).cpu().data.numpy())

    imgs = [temperature_imgs[i][j:j+1] for j in range(rows) for i in range(len(temp_steps))]
    img_show = img_tile(imgs, (columns, rows), transpose=False, channel_first=True, channels=1)

    if save_as:
        plt.imsave(save_as,  img_show, cmap='gray', vmin=0, vmax=1)

def val_loss(model, data, config):
    '''prints the final validiation loss of the model'''

    with torch.no_grad():
        z, log_j = model(data.val_x, data.val_l, jac=True)
        nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / np.prod(config.img_size)

    print('Validation loss:')
    print(nll_val.item())

def show_samples(model, data, config, label, train_conf):
    '''produces and shows cINN samples for a given label (0-9)'''

    N_samples = 100
    l = torch.LongTensor(N_samples).to(config.device)
    l[:] = label

    z = 1.0 * torch.randn(N_samples, np.prod(config.img_size)).to(config.device)

    with torch.no_grad():
        samples = model.reverse_sample(z, l)[0].cpu().numpy()
        if not config.maxpool:
            samples = data.unnormalize(samples)

    full_image = np.zeros((config.img_size[0] * 10, config.img_size[1] * 10))

    for k in range(N_samples):
        i, j = k // 10, k % 10
        full_image[config.img_size[0] * i : config.img_size[0] * (i + 1),
                   config.img_size[1] * j : config.img_size[1] * (j + 1)] = samples[k, 0]

    full_image = np.clip(full_image, 0, 1)
    plt.figure()
    plt.title(F'Generated digits for c={label}')
    plt.imshow(full_image, vmin=0, vmax=1, cmap='gray')
    plt.savefig(config.save_dir + f"/eval_{label}_" + train_conf + ".png")
