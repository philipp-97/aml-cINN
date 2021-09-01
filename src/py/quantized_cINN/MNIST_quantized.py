"""
MNIST cINN: quantized

depends on: {FrEIA}
based on:   https://github.com/VLL-HD/conditional_INNs
"""

__all__ = ["CONFIG", "MNISTcINN_quantized", "train", "evaluate"]

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

from common import one_hot, MNISTData, baseCONFIG, \
                   Visualizer, LiveVisualizer, sample_outputs, \
                   style_transfer, interpolation, val_loss, show_samples

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

    img_size = (28, 28)
    device = "cuda"
    n_workers = 16

    # Training
    lr = 5e-4
    batch_size = 256
    #decay_by = 0.01
    weight_decay = 1e-5
    gamma = 0.1
    milestones = [20, 40]
    betas = (0.9, 0.999)

    n_epochs = 60

    init_scale = 0.03
    pre_low_lr = 0

    clip_grad_norm = 10.0

    # Architecture
    n_blocks = 20
    internal_width = 512
    clamping = 1.0
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
    save_dir = "../../../out/MNIST_quantized"

    load_file = "../../../out/MINST_quantized/mnist_minimal_checkpoint.pt"
    filename = "../../../out/MNIST_quantized/mnist_minimal_cinn.pt"

    checkpoint_save_interval =  20
    checkpoint_save_overwrite = True
    checkpoint_on_error = True

########################################################################
# model definition

class MNISTcINN_quantized(nn.Module):
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

        #gamma = (self.c.decay_by)**(1. / self.c.n_epochs)
        self.optimizer = torch.optim.Adam(self.trainable_parameters,
                                          lr=self.c.lr,
                                          weight_decay=self.c.weight_decay)
        self.weight_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                #step_size=1,
                                                                milestones=self.c.milestones,
                                                                gamma=self.c.gamma)

    def build_inn(self):

        def fc_subnet(ch_in, ch_out):
            """
            subnet-builder
            """
            width = self.c.internal_width
            dropout = self.c.fc_dropout
            net = OrderedDict([
                ("quant", torch.quantization.QuantStub()),
                ("fuco1", nn.Linear(ch_in, self.c.internal_width)),
                ("relu1", nn.ReLU()),
                ("fuco2", nn.Linear(self.c.internal_width, ch_out)),
                ("dequa", torch.quantization.DeQuantStub())])
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

    def fuse_model(self):
        # fuse the activations to preceding layers, where applicable
        # this needs to be done manually depending on the model architecture
        for module in self.cinn.modules():
            if type(module) == nn.Sequential:
                torch.quantization.fuse_modules(module, ["fuco1", "relu1"], inplace=True)


        #return module_fused

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
# helper

def pre_training_quantization(model_fp32):
    # model must be set to train mode for QAT logic to work
    model_fp32.train()

    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'fbgemm' for server inference and
    # 'qnnpack' for mobile inference. Other quantization configurations such
    # as selecting symmetric or assymetric quantization and MinMax or L2Norm
    # calibration techniques can be specified here.
    #module_fp32 = model_fp32.fuse_model()
    model_fp32.fuse_model()

    return model_fp32_prepared

def post_training_quantization(model_fp32):

    return res

def train(config):
    model = MNISTcINN_quantized(config)
    #model = pre_training_quantization(model)
    data = MNISTData(config)

    t_start = time()

    nll_mean = []


    try:
        for i_epoch in range(-config.pre_low_lr, config.n_epochs):
            if i_epoch < 0:
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = config.lr * 2e-2

            for i_batch, (x, l) in tqdm(enumerate(data.train_loader),
                                        total=len(data.train_loader),
                                        leave=False,
                                        mininterval=1.,
                                        disable=(not config.progress_bar),
                                        ncols=83):

                x, l = x.to(config.device), l.to(config.device)
                z, log_j = model(x, l)

                nll = torch.mean(z**2) / 2 - torch.mean(log_j) / np.prod(config.img_size)
                nll.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters,
                                               config.clip_grad_norm)

                nll_mean.append(nll.item())

                model.optimizer.step()
                model.optimizer.zero_grad()

                if not i_batch % 50:
                    with torch.no_grad():
                        z, log_j = model(data.val_x, data.val_l)
                        nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / np.prod(config.img_size)

                    print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (i_epoch,
                                                                    i_batch, len(data.train_loader),
                                                                    (time() - t_start)/60.,
                                                                    np.mean(nll_mean),
                                                                    nll_val.item(),
                                                                    model.optimizer.param_groups[0]['lr'],
                                                                    ), flush=True)
                    nll_mean = []

            model.weight_scheduler.step()

            #if i_epoch > 1 - config.pre_low_lr:
            #    viz.update_losses(np.mean(nll_mean))

            if (i_epoch % config.checkpoint_save_interval) == 0:
                model.save(config.filename + '_checkpoint_%.4i' % (i_epoch * (1-config.checkpoint_save_overwrite)))

        #model = post_training_quantization(model)

        #####

        config.device = "cpu"
        model.c.device = "cpu"
        data.c.device = "cpu"
        model.to("cpu")

        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, fuses modules where appropriate,
        # and replaces key operators with quantized implementations.
        model.eval()
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        for module in model.cinn.modules():
            module.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        # Prepare the model for QAT. This inserts observers and fake_quants in
        # the model that will observe weight and activation tensors during calibration.
        model = torch.quantization.prepare(model)

        print("before")
        print(model.state_dict()["cinn.module_list.2.subnet1.quant.activation_post_process.activation_post_process.min_val"])
        print(model.state_dict()["cinn.module_list.2.subnet1.quant.activation_post_process.activation_post_process.max_val"])
        print(model.state_dict()["cinn.module_list.2.subnet1.fuco1.0.weight"])
        for k in range(10):
            show_samples(model, data, config, k)
        print("after")
        print(model.state_dict()["cinn.module_list.2.subnet1.quant.activation_post_process.activation_post_process.min_val"])
        print(model.state_dict()["cinn.module_list.2.subnet1.quant.activation_post_process.activation_post_process.max_val"])
        print(model.state_dict()["cinn.module_list.2.subnet1.fuco1.0.weight"])

        model_int8 = torch.quantization.convert(model)
        for k in range(10):
            show_samples(model_int8, data, config, k)
        #####

        model_int8.save(config.filename)

    except BaseException as b:
        if config.checkpoint_on_error:
            model.save(config.filename + "_ABORT")
        raise b

def evaluate(config):
    model = MNISTcINN_quantized(config)
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

    torch.backends.quantized.engine = 'fbgemm'

    if args.downloadMNIST:
        data = MNISTData(config)

    if args.train:
        # model training
        train(config)

    if args.eval:
        # model evaluation
        evaluate(config)

    print("Done! Exit normaly.")
