"""
MNIST cINN: normal

depends on: {FrEIA}
based on:   https://github.com/VLL-HD/conditional_INNs
"""

__all__ = ["CONFIG", "MNISTcINN_normal", "train", "evaluate"]

# imports
import numpy as np
from collections import OrderedDict

import torch
import torch.optim
import torch.nn as nn

from tqdm import tqdm
from time import time

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from .common import one_hot, MNISTData, baseCONFIG, \
                   Visualizer, LiveVisualizer, sample_outputs, \
                   style_transfer, interpolation, val_loss, show_samples

########################################################################
# configuration

class CONFIG(baseCONFIG):
    """
    Namspace for configuration
    """
    # Data
    data_mean = 0.0
    data_std = 1.0
    add_image_noise = 0.02

    img_size = (28, 28)
    device = "cuda"
    n_workers = 4

    # Training
    lr = 1e-4
    batch_size = 512
    decay_by = 0.01
    weight_decay = 1e-5
    betas = (0.9, 0.999)

    n_epochs = 120 * 12
    n_its_per_epoch = 2**16

    init_scale = 0.03
    pre_low_lr = 1
    
    clip_grad_norm = 10.0

    # Architecture
    n_blocks = 24
    internal_width = 512
    clamping = 1.5
    fc_dropout = 0.0

    # Logging/preview
    loss_names = ['L']
    preview_upscale = 3                         # Scale up the images for preview
    sampling_temperature = 0.8                  # Sample at a reduced temperature for the preview
    progress_bar = True                         # Show a progress bar of each epoch
    eval_steps_interploation = 12
    eval_seeds_interpolation  = (51, 89)

    # Validation
    pca_weights = [
        [(0,0.55)],
        [(1,0.1), (3, 0.4), (4, 0.5)],
        [(2,0.33), (3, 0.33), (1, -0.33)]]
    pca_gridsize = 10
    pca_extent = 8.


    # Paths
    mnist_data = "../../../mnist_data"
    save_dir = "../../../out/MNIST_floating"

    load_file = "../../../out/MNIST_floating/mnist_minimal_checkpoint.pt"
    filename = "../../../out/MNIST_floating/mnist_minimal_cinn.pt"
    #mnist_data = "mnist_data"
    #save_dir = "output/normal"

    #load_file = "output/normal/checkpoint.pt"
    #filename = "output/normal/mnist_cinn.pt"

    checkpoint_save_interval =  120 * 3
    checkpoint_save_overwrite = True
    checkpoint_on_error = True

########################################################################
# model definition

class MNISTcINN_normal(nn.Module):
    """
    Conditional INN model for MNIST
    """

    def __init__(self, config: object=CONFIG):
        super().__init__()
        self.c = config

        self.cinn = self.build_inn()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = self.c.init_scale * torch.randn_like(p)

        self.cinn.to(self.c.device)

        gamma = (self.c.decay_by)**(1. / self.c.n_epochs)
        self.optimizer = torch.optim.Adam(self.trainable_parameters,
                                          lr=self.c.lr,
                                          betas=self.c.betas,
                                          eps=1e-6,
                                          weight_decay=self.c.weight_decay)
        self.weight_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                step_size=1,
                                                                gamma=gamma)

    def build_inn(self):

        # -> this network is to small
        #def fc_subnet(ch_in, ch_out):
        #    return nn.Sequential(nn.Linear(ch_in, self.c.internal_width),
        #                         nn.ReLU(),
        #                         nn.Linear(self.c.internal_width, ch_out))

        def fc_subnet(ch_in, ch_out):
            """
            subnet-builder
            Replace depricated 'F_fully_connected' in FrEIA
            e.g. commit 2b725533045058e647bab4c7c4382a94db5025ca
            """
            width = self.c.internal_width
            dropout = self.c.fc_dropout
            net = OrderedDict([
                ("fuco1", nn.Linear(ch_in, width)),
                ("drop1", nn.Dropout(p=dropout)),
                ("relu1", nn.ReLU()),
                ("fuco2", nn.Linear(width, width)),
                ("drop2", nn.Dropout(p=dropout)),
                ("relu2", nn.ReLU()),
                ("fuco2b", nn.Linear(width, width)),
                ("drop2b", nn.Dropout(p=dropout)),
                ("relu2b", nn.ReLU()),
                ("fuco3", nn.Linear(width, ch_out))])
            return nn.Sequential(net)

        cond = Ff.ConditionNode(10)

        nodes = [Ff.InputNode(1, *self.c.img_size)]
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(self.c.n_blocks):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom,
                                 {"seed": k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {"subnet_constructor": fc_subnet,
                                  "clamp": self.c.clamping},
                                 conditions=cond))

        nodes += [cond, Ff.OutputNode(nodes[-1])]
        return Ff.ReversibleGraphNet(nodes, verbose=False)

    def forward(self, x, l, jac=True):
        return self.cinn(x, c=one_hot(l), jac=jac)

    def reverse_sample(self, z, l, jac=True):
        return self.cinn(z, c=one_hot(l), rev=True, jac=jac)

    def save(self, name):
        save_dict = {"opt": self.optimizer.state_dict(),
                     "net": self.cinn.state_dict(),
                     "lr": self.weight_scheduler.state_dict()}
        torch.save(save_dict, name)

    def load(self, name):
        state_dicts = torch.load(name)
        self.cinn.load_state_dict(state_dicts["net"])
        try:
            self.optimizer.load_state_dict(state_dicts["opt"])
        except ValueError:
            print("Cannot load optimizer for some reason or other")
        try:
            self.weight_scheduler.load_state_dict(state_dicts["lr"])
        except ValueError:
            print("Cannot load optimizer for some reason or other")


########################################################################
# loss definition
# => not used

def MMD(x, y):
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*xy

    dxx = torch.clamp(dxx, 0., np.inf)
    dyy = torch.clamp(dyy, 0., np.inf)
    dxy = torch.clamp(dxy, 0., np.inf)

    XX, YY, XY = (Variable(torch.zeros(xx.shape).cuda()),
                  Variable(torch.zeros(xx.shape).cuda()),
                  Variable(torch.zeros(xx.shape).cuda()))

    for cw in CONFIG.kernel_widths:
        for a in CONFIG.kernel_powers:
            XX += cw**a * (cw + 0.5 * dxx / a)**-a
            YY += cw**a * (cw + 0.5 * dyy / a)**-a
            XY += cw**a * (cw + 0.5 * dxy / a)**-a

    return torch.mean(XX + YY - 2.*XY)

def moment_match(x, y):
    return (torch.mean(x) - torch.mean(y))**2 + (torch.var(x) - torch.var(y))**2



########################################################################
# helper

def train(config):
    model = MNISTcINN_normal(config)
    data = MNISTData(config)

    viz = LiveVisualizer(config)

    try:
        for i_epoch in range(-config.pre_low_lr, config.n_epochs):

            nll_mean = []
            data_iter = iter(data.train_loader)

            if i_epoch < 0:
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = config.lr * 2e-2

            for i_batch, (x, l) in tqdm(enumerate(data_iter),
                                        total=min(len(data.train_loader),
                                        config.n_its_per_epoch),
                                        leave=False,
                                        mininterval=1.,
                                        disable=(not config.progress_bar),
                                        ncols=83):

                x, l = x.cuda(), l.cuda()
                z, log_j = model(x, l)

                nll = torch.mean(z**2) / 2 - torch.mean(log_j) / np.prod(config.img_size)
                nll.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters,
                                               config.clip_grad_norm)

                nll_mean.append(nll.item())

                model.optimizer.step()
                model.optimizer.zero_grad()

            model.weight_scheduler.step()

            if i_epoch > 1 - config.pre_low_lr:
                viz.update_losses(np.mean(nll_mean))

            # Since LiveVisualizer is not supportet, this is useless
            #with torch.no_grad():
            #    samples = sample_outputs(config)
            #
            #    test_l = list(range(10))*(config.batch_size//10 + 1)
            #    test_l = torch.LongTensor(test_l)
            #    test_l = test_l[:config.batch_size].to(config.device)
            #
            #    rev_imgs = model.reverse_sample(samples, test_l)
            #    ims = [rev_imgs]

            if (i_epoch % config.checkpoint_save_interval) == 0:
                model.save(config.filename + '_checkpoint_%.4i' % (i_epoch * (1-config.checkpoint_save_overwrite)))

        model.save(config.filename)

    except BaseException as b:
        if config.checkpoint_on_error:
            model.save(config.filename + "_ABORT")
        raise b

def evaluate(config):
    model = MNISTcINN_normal(config)
    model.load(config.filename)
    model.cinn.train(False)
    data = MNISTData(config)

    #for s in tqdm(range(0, 256)):
    #    torch.manual_seed(s)
    #    temperature(0.88, columns=1, save_as='./images/samples/T_%.4i.png' % (s))
    #    plt.title(str(s))


    index_ins = [284, 394, 422, 759, 639, 599, 471, 449, 448, 426]
    style_transfer(model, data, index_ins, config)

    interpolation(model, config)

    #for j in range(3):
    #    plt.figure()
    #    for i in range(10):
    #        plt.subplot(10, 1, i+1)
    #        val_set_pca(I=j, C=i)
    #    plt.title(str(j))

    val_loss(model, data, config)

    for i in range(10):
        show_samples(model, data, config, i)



########################################################################
# execution

if __name__ == "__main__":
    config = CONFIG()

    import argparse
    parser = argparse.ArgumentParser(description=config.str())
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-e", "--eval", action="store_true")
    parser.add_argument("-d", "--downloadMNIST", action="store_true")
    args = parser.parse_args()

    print(config.str())

    if args.downloadMNIST:
        data = MNISTData(config)

    if args.train:
        # model training
        train(config)

    if args.eval:
        # model evaluation
        evaluate(config)

    print("Done! Exit normaly.")
