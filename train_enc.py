from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os

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
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--num-atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--num-classes', type=int, default=2,
                    help='Number of edge types.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='_springs',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='How many batches to wait before logging.')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor')
parser.add_argument('--motion', action='store_true', default=False,
                    help='Use motion capture data loader.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

log = None
# Save model and meta-data. Always saves in a new folder.
if args.save_folder:
    exp_counter = 0
    save_folder = '{}/exp{}/'.format(args.save_folder, exp_counter)
    while os.path.isdir(save_folder):
        exp_counter += 1
        save_folder = os.path.join(args.save_folder,
                                   'exp{}'.format(exp_counter))
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    model_file = os.path.join(save_folder, 'encoder.pt')
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
    args.batch_size, args.suffix)

# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

if args.encoder == 'mlp':
    model = MLPEncoder(args.timesteps * args.dims, args.hidden,
                       args.edge_types,
                       args.dropout, args.factor)
elif args.encoder == 'cnn':
    model = CNNEncoder(args.dims, args.hidden, args.edge_types,
                       args.dropout, args.factor)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# Linear indices of an upper triangular mx, used for loss calculation
triu_indices = get_triu_offdiag_indices(args.num_atoms)

if args.cuda:
    model.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)

best_model_params = model.state_dict()


def train(epoch, best_val_accuracy):
    t = time.time()
    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    model.train()
    scheduler.step()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data, rel_rec, rel_send)

        # Flatten batch dim
        output = output.view(-1, args.num_classes)
        target = target.view(-1)

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        acc = correct / pred.size(0)

        loss_train.append(loss.data[0])
        acc_train.append(acc)

    model.eval()
    for batch_idx, (data, target) in enumerate(valid_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(
            target, volatile=True)
        output = model(data, rel_rec, rel_send)

        # Flatten batch dim
        output = output.view(-1, args.num_classes)
        target = target.view(-1)

        loss = F.cross_entropy(output, target)

        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        acc = correct / pred.size(0)

        loss_val.append(loss.data[0])
        acc_val.append(acc)

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.10f}'.format(np.mean(loss_train)),
          'acc_train: {:.10f}'.format(np.mean(acc_train)),
          'loss_val: {:.10f}'.format(np.mean(loss_val)),
          'acc_val: {:.10f}'.format(np.mean(acc_val)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(acc_val) > best_val_accuracy:
        torch.save(model.state_dict(), model_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(acc_val)


def test():
    t = time.time()
    loss_test = []
    acc_test = []
    model.eval()
    model.load_state_dict(torch.load(model_file))
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(
            target, volatile=True)

        # Limit to same length as train sequence
        data = data[:, :, :args.timesteps, :].contiguous()

        output = model(data, rel_rec, rel_send)
        # Flatten batch dim
        output = output.view(-1, args.num_classes)
        target = target.view(-1)

        loss = F.cross_entropy(output, target)

        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        acc = correct / pred.size(0)

        loss_test.append(loss.data[0])
        acc_test.append(acc)
    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('loss_test: {:.10f}'.format(np.mean(loss_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)))
    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('loss_test: {:.10f}'.format(np.mean(loss_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)), file=log)
        log.flush()
    return np.mean(acc_test)


# Train model
t_total = time.time()
best_val_accuracy = -1.
best_epoch = 0
for epoch in range(args.epochs):
    val_acc = train(epoch, best_val_accuracy)
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
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
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
