"""
MNIST cINN: hx

depends on: {FrEIA}
based on:   https://github.com/VLL-HD/conditional_INNs
"""

__all__ = ["CONFIG", "MNISTcINN_hx", "train", "evaluate"]

# imports
from collections import OrderedDict

from time import time
from tqdm import tqdm

import numpy as np

import torch
import torch.optim
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from _hxtorch.constants import input_activation_max
import hxtorch.nn as hxnn

from common import one_hot, MNISTData, MNISTDataPreprocessed, baseCONFIG, \
                   style_transfer, interpolation, val_loss, show_samples

from layers import DynamicScaling, FixedRangeScaling

########################################################################
# configuration


class CONFIG(baseCONFIG):
    """
    Namspace for configuration
    """
    # Data
    data_mean = 0.128
    data_std = 0.305
    add_image_noise = 0.08

    maxpool = False

    img_size = (28, 28)
    device = "cpu"
    n_workers = 4

    # Training
    lr = 5e-4
    batch_size = 256
    #decay_by = 0.01
    weight_decay = 1e-5
    gamma = 0.1
    milestones = [20, 40]
    betas = (0.9, 0.999)

    n_epochs = 5

    init_scale = 0.03
    pre_low_lr = 0

    clip_grad_norm = 10.0

    mock = False

    # Architecture
    n_blocks = 20
    internal_width = 128
    clamping = 1.0
    fc_dropout = 0.0

    # Logging/preview
    loss_names = ['L']
    preview_upscale = 3         # Scale up the images for preview
    sampling_temperature = 0.8  # Sample at a reduced temperature for preview
    progress_bar = True         # Show a progress bar of each epoch
    eval_steps_interploation = 12
    eval_seeds_interpolation = (51, 89)

    # Validation
    pca_weights = [
        [(0, 0.55)],
        [(1, 0.1), (3, 0.4), (4, 0.5)],
        [(2, 0.33), (3, 0.33), (1, -0.33)]]
    pca_gridsize = 10
    pca_extent = 8.

    # Paths
    mnist_data = "../../../mnist_data"
    save_dir = "../../../out/MNIST_hx"

    load_file = "../../../out/MNIST_hx/mnist_hx_checkpoint.pt"
    filename = "../../../out/MNIST_hx/mnist_hx_cinn.pt"

    loss_means_filename = save_dir + f"/val_losses_means_{n_epochs}e_{mock}m.txt"
    loss_filename = save_dir + f"/val_losses_{n_epochs}e_{mock}m.txt"

    checkpoint_save_interval = 1
    checkpoint_save_overwrite = True
    checkpoint_on_error = True

########################################################################
# model definition


class MNISTcINN_hx(nn.Module):
    """
    Conditional INN model for MNIST
    """

    def __init__(self, config: object = CONFIG):
        super().__init__()
        self.c = config

        self.cinn = self.build_inn()

        self.trainable_parameters = []
        for name, param in self.cinn.named_parameters():
            if param.requires_grad and not name.split(".")[-2][-2:] == "__":
                self.trainable_parameters.append(param)
            continue

        self.cinn.to(self.c.device)

        self.optimizer = torch.optim.Adam(self.trainable_parameters,
                                          lr=self.c.lr,
                                          weight_decay=self.c.weight_decay)
        self.weight_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            #step_size=1,
            milestones=self.c.milestones,
            gamma=self.c.gamma)

    def build_inn(self):

        def scale_input_total(x_in: torch.Tensor) -> torch.Tensor:
            """
            Scales the tensor to the maximal input range of BrainScaleS-2.
            """
            max_abs = torch.max(torch.abs(x_in))
            factor = input_activation_max / max_abs \
                if max_abs != 0 else 1
            return x_in * factor

        def fc_subnet(ch_in, ch_out):
            net = OrderedDict([
                ("hx_input1", FixedRangeScaling(features=ch_in,
                                                max_out=31,
                                                per_feature=True)),
                ("hx_lin_1", hxnn.Linear(in_features=ch_in,
                                         out_features=self.c.internal_width,
                                         bias=False,
                                         num_sends=2,
                                         wait_between_events=2,
                                         mock=self.c.mock,
                                         signed_input=True,
                                         )),
                ("relu1", hxnn.ConvertingReLU(shift=1,
                                              mock=self.c.mock)),
                ("hx_input2", FixedRangeScaling(features=self.c.internal_width,
                                                max_out=input_activation_max,
                                                per_feature=True)),
                ("hx_lin_2", hxnn.Linear(in_features=self.c.internal_width,
                                         out_features=ch_out,
                                         bias=False,
                                         num_sends=3,
                                         wait_between_events=2,
                                         mock=self.c.mock,
                                         signed_input=True,
                                         )),
                ("hx_output", DynamicScaling(features=ch_out))
            ])
            return nn.Sequential(net)

        cond = Ff.ConditionNode(10)

        nodes = [Ff.InputNode(1, *self.c.img_size)]
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(self.c.n_blocks):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom,
                                 {"seed": k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {"subnet_constructor": fc_subnet,
                                  "clamp": self.c.clamping,
                                  "clamp_activation": torch.sigmoid},
                                 conditions=cond))

        nodes += [cond, Ff.OutputNode(nodes[-1])]
        return Ff.ReversibleGraphNet(nodes, verbose=False)

    def forward(self, x, label, jac=True):
        return self.cinn(x, c=one_hot(label, scale=31.), jac=jac)

    def reverse_sample(self, z, label, jac=True):
        return self.cinn(z, c=one_hot(label, scale=31.), rev=True, jac=jac)

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
# helper

def train(config):
    model = MNISTcINN_hx(config)
    ###
    #model.load(config.filename)
    ###
    if config.maxpool:
        data = MNISTDataPreprocessed(config)
    else:
        data = MNISTData(config)

    t_start = time()

    nll_mean = []

    # memorize evolution of losses
    val_losses_means = np.array([])
    val_losses = np.array([])

    try:
        for i_epoch in range(-config.pre_low_lr, config.n_epochs):
            if i_epoch < 0:
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = config.lr * 2e-2

            for i_batch, (x, label) in tqdm(enumerate(data.train_loader),
                                            total=len(data.train_loader),
                                            leave=False,
                                            mininterval=1.,
                                            disable=(not config.progress_bar),
                                            ncols=83):

                x, label = x.to(config.device), label.to(config.device)
                z, log_j = model(x, label)

                nll = torch.mean(z**2) / 2 - torch.mean(log_j) / np.prod(
                    config.img_size)
                nll.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters,
                                               config.clip_grad_norm)

                nll_mean.append(nll.item())

                model.optimizer.step()
                model.optimizer.zero_grad()

            with torch.no_grad():
                model.eval()
                z, log_j = model(data.val_x, data.val_l)
                nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / np.prod(
                    config.img_size)
                model.train()

            print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' %
                  (i_epoch, i_batch, len(data.train_loader),
                   (time() - t_start) / 60., np.mean(nll_mean), nll_val.item(),
                   model.optimizer.param_groups[0]['lr'],
                   ), flush=True)

            val_losses_means = np.append(val_losses_means, np.mean(nll_mean))
            val_losses = np.append(val_losses, nll_val.item())

            nll_mean = []

            model.weight_scheduler.step()

            # save model at checkpoints
            if (i_epoch % config.checkpoint_save_interval) == 0:
                model.save(config.filename + '_checkpoint_%.4i' %
                           (i_epoch * (1-config.checkpoint_save_overwrite)))

        # save model and losses
        model.save(config.filename)
        np.savetxt(config.loss_means_filename, val_losses_means)
        np.savetxt(config.loss_filename, val_losses)

    except BaseException as b:
        if config.checkpoint_on_error:
            model.save(config.filename + "_ABORT")
        raise b


def evaluate(config):
    model = MNISTcINN_hx(config)
    model.load(config.filename)
    model.eval()
    if config.maxpool:
        data = MNISTDataPreprocessed(config)
    else:
        data = MNISTData(config)

    train_config = f"width{config.internal_width}_epochs{config.n_epochs}"

    index_ins = [284, 394, 422, 759, 639, 599, 471, 449, 448, 426]
    style_transfer(model, data, index_ins, config, train_config)

    interpolation(model, config, train_config)

    val_loss(model, data, config)

    for i in range(10):
        show_samples(model, data, config, i, train_config)


########################################################################
# execution


if __name__ == "__main__":
    config = CONFIG()

    import argparse
    parser = argparse.ArgumentParser(description=config.str())
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-e", "--eval", action="store_true")
    parser.add_argument("-m", "--maxpool", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("-d", "--downloadMNIST", action="store_true")
    args = parser.parse_args()

    if args.downloadMNIST:
        data = MNISTData(config)

    if args.maxpool:
        config.maxpool = True
        config.img_size = (14, 14)
        config.data_mean = None
        config.data_std = None

        config.save_dir = "../../../out/MNIST_hx_maxpool"
        config.load_file = config.save_dir + "/mnist_hx_maxpool_checkpoint.pt"
        config.filename = config.save_dir + "/mnist_hx_maxpool_cinn.pt"

    if args.mock:
        config.mock = True

    if args.train:
        # model training
        print(config.str())
        train(config)

    if args.eval:
        # model evaluation
        print(config.str())
        evaluate(config)

    print("Done! Exit normaly.")
