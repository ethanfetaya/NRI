from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from modules import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--num_atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--num-layers', type=int, default=2,
                    help='Number of LSTM layers.')
parser.add_argument('--suffix', type=str, default='_springs',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='How many batches to wait before logging.')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--motion', action='store_true', default=False,
                    help='Use motion capture data loader.')
parser.add_argument('--non-markov', action='store_true', default=False,
                    help='Use non-Markovian evaluation setting.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

log = None
# Save model and meta-data. Always saves in a new folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    while os.path.isdir(save_folder):
        exp_counter += 1
        save_folder = os.path.join(args.save_folder,
                                   'exp{}'.format(exp_counter))
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    model_file = os.path.join(save_folder, 'model.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))

else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
    args.batch_size, args.suffix)


class RecurrentBaseline(nn.Module):
    """LSTM model for joint trajectory prediction."""

    def __init__(self, n_in, n_hid, n_out, n_atoms, n_layers, do_prob=0.):
        super(RecurrentBaseline, self).__init__()
        self.fc1_1 = nn.Linear(n_in, n_hid)
        self.fc1_2 = nn.Linear(n_hid, n_hid)
        self.rnn = nn.LSTM(n_atoms * n_hid, n_atoms * n_hid, n_layers)
        self.fc2_1 = nn.Linear(n_atoms * n_hid, n_atoms * n_hid)
        self.fc2_2 = nn.Linear(n_atoms * n_hid, n_atoms * n_out)

        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def step(self, ins, hidden=None):
        # Input shape: [num_sims, n_atoms, n_in]
        x = F.relu(self.fc1_1(ins))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.relu(self.fc1_2(x))
        x = x.view(ins.size(0), -1)
        # [num_sims, n_atoms*n_hid]

        x = x.unsqueeze(0)
        x, hidden = self.rnn(x, hidden)
        x = x[0, :, :]

        x = F.relu(self.fc2_1(x))
        x = self.fc2_2(x)
        # [num_sims, n_out*n_atoms]

        x = x.view(ins.size(0), ins.size(1), -1)
        # [num_sims, n_atoms, n_out]

        # Predict position/velocity difference
        x = x + ins

        return x, hidden

    def forward(self, inputs, prediction_steps, burn_in=False, burn_in_steps=1):

        # Input shape: [num_sims, num_things, num_timesteps, n_in]

        outputs = []
        hidden = None

        for step in range(0, inputs.size(2) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, :, step, :]
                else:
                    ins = outputs[step - 1]
            else:
                # Use ground truth trajectory input vs. last prediction
                if not step % prediction_steps:
                    ins = inputs[:, :, step, :]
                else:
                    ins = outputs[step - 1]

            output, hidden = self.step(ins, hidden)

            # Predict position/velocity difference
            outputs.append(output)

        outputs = torch.stack(outputs, dim=2)

        return outputs


model = RecurrentBaseline(args.dims, args.hidden, args.dims,
                          args.num_atoms, args.num_layers, args.dropout)
if args.load_folder:
    model_file = os.path.join(args.load_folder, 'model.pt')
    model.load_state_dict(torch.load(model_file))
    args.save_folder = False

optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# Linear indices of an upper triangular mx, used for loss calculation
triu_indices = get_triu_offdiag_indices(args.num_atoms)

if args.cuda:
    model.cuda()


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def train(epoch, best_val_loss):
    t = time.time()
    loss_train = []
    loss_val = []
    mse_baseline_train = []
    mse_baseline_val = []
    mse_train = []
    mse_val = []

    model.train()
    scheduler.step()
    for batch_idx, (data, relations) in enumerate(train_loader):

        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data), Variable(relations)

        optimizer.zero_grad()

        output = model(data, 100,
                       burn_in=True,
                       burn_in_steps=args.timesteps - args.prediction_steps)

        target = data[:, :, 1:, :]
        loss = nll_gaussian(output, target, args.var)

        mse = F.mse_loss(output, target)
        mse_baseline = F.mse_loss(data[:, :, :-1, :], data[:, :, 1:, :])

        loss.backward()
        optimizer.step()

        loss_train.append(loss.data[0])
        mse_train.append(mse.data[0])
        mse_baseline_train.append(mse_baseline.data[0])

    model.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data, requires_grad=False), Variable(
            relations, requires_grad=False)

        output = model(data, 1)

        target = data[:, :, 1:, :]

        loss = nll_gaussian(output, target, args.var)

        mse = F.mse_loss(output, target)
        mse_baseline = F.mse_loss(data[:, :, :-1, :], data[:, :, 1:, :])

        loss_val.append(loss.data[0])
        mse_val.append(mse.data[0])
        mse_baseline_val.append(mse_baseline.data[0])

    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(loss_train)),
          'mse_train: {:.12f}'.format(np.mean(mse_train)),
          'mse_baseline_train: {:.10f}'.format(np.mean(mse_baseline_train)),
          'nll_val: {:.10f}'.format(np.mean(loss_val)),
          'mse_val: {:.12f}'.format(np.mean(mse_val)),
          'mse_baseline_val: {:.10f}'.format(np.mean(mse_baseline_val)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(loss_val) < best_val_loss:
        torch.save(model.state_dict(), model_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(loss_train)),
              'mse_train: {:.12f}'.format(np.mean(mse_train)),
              'mse_baseline_train: {:.10f}'.format(np.mean(mse_baseline_train)),
              'nll_val: {:.10f}'.format(np.mean(loss_val)),
              'mse_val: {:.12f}'.format(np.mean(mse_val)),
              'mse_baseline_val: {:.10f}'.format(np.mean(mse_baseline_val)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(loss_val)


def test():
    loss_test = []
    mse_baseline_test = []
    mse_test = []
    tot_mse = 0
    tot_mse_baseline = 0
    counter = 0

    model.eval()
    model.load_state_dict(torch.load(model_file))
    for batch_idx, (inputs, relations) in enumerate(test_loader):

        assert (inputs.size(2) - args.timesteps) >= args.timesteps

        if args.cuda:
            inputs = inputs.cuda()
        else:
            inputs = inputs.contiguous()
        inputs = Variable(inputs, volatile=True)

        ins_cut = inputs[:, :, -args.timesteps:, :].contiguous()

        output = model(ins_cut, 1)

        target = ins_cut[:, :, 1:, :]

        loss = nll_gaussian(output, target, args.var)

        mse = F.mse_loss(output, target)
        mse_baseline = F.mse_loss(ins_cut[:, :, :-1, :], ins_cut[:, :, 1:, :])

        loss_test.append(loss.data[0])
        mse_test.append(mse.data[0])
        mse_baseline_test.append(mse_baseline.data[0])

        if args.motion or args.non_markov:
            # RNN decoder evaluation setting

            # For plotting purposes
            output = model(inputs, 100, burn_in=True,
                           burn_in_steps=args.timesteps)

            output = output[:, :, args.timesteps:, :]
            target = inputs[:, :, -args.timesteps:, :]
            mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
            tot_mse += mse.data.cpu().numpy()
            counter += 1

            # Baseline over multiple steps
            baseline = inputs[:, :, -(args.timesteps + 1):-args.timesteps,
                       :].expand_as(
                target)
            mse_baseline = ((target - baseline) ** 2).mean(dim=0).mean(
                dim=0).mean(
                dim=-1)
            tot_mse_baseline += mse_baseline.data.cpu().numpy()

        else:

            # For plotting purposes
            output = model(inputs, 100, burn_in=True,
                           burn_in_steps=args.timesteps)

            output = output[:, :, args.timesteps:args.timesteps + 20, :]
            target = inputs[:, :, args.timesteps + 1:args.timesteps + 21, :]

            mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
            tot_mse += mse.data.cpu().numpy()
            counter += 1

            # Baseline over multiple steps
            baseline = inputs[:, :, args.timesteps:args.timesteps + 1,
                       :].expand_as(
                target)
            mse_baseline = ((target - baseline) ** 2).mean(dim=0).mean(
                dim=0).mean(
                dim=-1)
            tot_mse_baseline += mse_baseline.data.cpu().numpy()

    mean_mse = tot_mse / counter
    mse_str = '['
    for mse_step in mean_mse[:-1]:
        mse_str += " {:.12f} ,".format(mse_step)
    mse_str += " {:.12f} ".format(mean_mse[-1])
    mse_str += ']'

    mean_mse_baseline = tot_mse_baseline / counter
    mse_baseline_str = '['
    for mse_step in mean_mse_baseline[:-1]:
        mse_baseline_str += " {:.12f} ,".format(mse_step)
    mse_baseline_str += " {:.12f} ".format(mean_mse_baseline[-1])
    mse_baseline_str += ']'

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(loss_test)),
          'mse_test: {:.12f}'.format(np.mean(mse_test)),
          'mse_baseline_test: {:.10f}'.format(np.mean(mse_baseline_test)))
    print('MSE: {}'.format(mse_str))
    print('MSE Baseline: {}'.format(mse_baseline_str))
    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('nll_test: {:.10f}'.format(np.mean(loss_test)),
              'mse_test: {:.12f}'.format(np.mean(mse_test)),
              'mse_baseline_test: {:.10f}'.format(np.mean(mse_baseline_test)),
              file=log)
        print('MSE: {}'.format(mse_str), file=log)
        print('MSE Baseline: {}'.format(mse_baseline_str), file=log)
        log.flush()


# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()
test()
if log is not None:
    print(save_folder)
    log.close()
