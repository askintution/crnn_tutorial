
from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms

import six
import sys
from PIL import Image
import numpy as np
import cv2

import os
import collections

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                raw_text = ''.join([self.alphabet[i - 1] for i in t])
                
                return raw_text
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))

def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img

"""
自定义组件，包含了双向LSTM+全连接层

torch lstm结构:
input(seq_len, batch, input_size)

第一维体现的是序列（sequence）结构,也就是序列的个数，用文章来说，就是每个句子的长度，因为是喂给网络模型，
一般都设定为确定的长度，也就是我们喂给LSTM神经元的每个句子的长度，当然，如果是其他的带有带有序列形式的数据，
则表示一个明确分割单位长度，例如是如果是股票数据内，这表示特定时间单位内，有多少条数据。
这个参数也就是明确这个层中有多少个确定的单元来处理输入的数据。

第二维度体现的是batch_size，也就是一次性喂给网络多少条句子，
或者股票数据中的，一次性喂给模型多少是个时间单位的数据，具体到每个时刻，
也就是一次性喂给特定时刻处理的单元的单词数或者该时刻应该喂给的股票数据的条数

第三位体现的是输入的元素（elements of input），也就是，
每个具体的单词用多少维向量来表示，或者股票数据中 每一个具体的时刻的采集多少具体的值，
比如最低价，最高价，均价，5日均价，10均价，等等


输出:

output(seq_len, batch, hidden_size * num_directions)


"""
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        # 将时间步与样本数两维压缩至同一维度（T×b, h）
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]

        # 从FC层输出后再转为RNN的输出维度排列
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        # 图片高度H必须被16整除
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        # ks: 卷积核尺寸
        ks = [3, 3, 3, 3, 3, 3, 2]

        # ps: 补全尺寸
        ps = [1, 1, 1, 1, 1, 1, 0]

        # ks: 卷积步长
        ss = [1, 1, 1, 1, 1, 1, 1]

        # nm: 通道数
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def conv_relu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        conv_relu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        conv_relu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        conv_relu(2, True)
        conv_relu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        conv_relu(4, True)
        conv_relu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        conv_relu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nclass),)

    def forward(self, input):
        # conv features
        conv = self.cnn(input)

        # 16,512,1,49，相当于把高度压缩了16倍，宽度压缩了4倍
        b, c, h, w = conv.size()

        # 确保卷积特征高度为1，从而能够转化为rnn输入格式
        assert h == 1, "the height of conv must be 1"

        """

        squeeze将h的维度取消，变为(b,c,w) (16,512,1,49) --> (16,512,49)
        permute将(b,c,w)变为(w,b,c) --> w就变成了t，b变成了n (16,512,49) --> (49,16,512)
        """
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)


        # rnn features，(49,16,512) --> (49,16,77)
        output = self.rnn(conv)

        return output

class RawDataset(Dataset):

    def __init__(self, gt_file, gt_dir, transform=None, target_transform=None):

        self.image_list = []
        self.label_list = []

        self.transform = transform
        self.target_transform = target_transform

        gt_lines = open(gt_file, 'r').readlines()

        image_path_list = []
        label_list = []

        lx = {}

        for line in gt_lines:
            if len(line) > 2:
                image_path, label_str = line.split(', ')[:2]

                # 长度大于32的字符串过滤掉，保证CTC正常工作
                if len(label_str) < 32:
                    image_path = os.path.join(gt_dir, image_path)
                    image_path_list.append(image_path)
                    label_str = label_str.replace('\"', '').replace(' ', '').replace('\n', '')
                    label_list.append(label_str)

                for l in label_str:
                    lx[l] = 1

        lx_list = list(lx.keys())
        lx_list.sort()
        lx_str = ''
        for l in lx_list:
            lx_str += l

        # 类别数与字符的对应字典
        print('lexicon:', lx_str)

        self.lexicon = lx_str
        self.nSamples = len(image_path_list)
        self.image_list = image_path_list
        self.label_list = label_list

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1

        image_path = self.image_list[index % self.nSamples]
        label = self.label_list[index % self.nSamples]

        try:
            #img = cv2.imread(image_path, 0)
            img = Image.open(image_path).convert("L")
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)

# 对图片像素进行正则化处理
class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

# 随机顺序采样，如果最后一组数据不够，随机补充其它数据
class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


"""
是否按照原来的宽高比进行缩放
"""
class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)


        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        return images, labels


parser = argparse.ArgumentParser()
# 指定训练集根目录
parser.add_argument('--trainroot', required=True, help='训练集路径')
# 指定验证集根目录
parser.add_argument('--valroot', required=True, help='验证集路径')

parser.add_argument('--workers', type=int, help='提取数据的线程数', default=0)
# 指定每次输入的图片数量，默认为16张
parser.add_argument('--batchSize', type=int, default=16, help='输入批次数量')

# 指定输入图片高度，默认为32个像素
parser.add_argument('--imgH', type=int, default=32, help='输入图像高度，默认为32像素')
# 指定输入图片高度，默认为192个像素
parser.add_argument('--imgW', type=int, default=192, help='输入图像宽度，默认为192像素')
parser.add_argument('--nh', type=int, default=128, help='LSTM隐层单元数')
parser.add_argument('--nepoch', type=int, default=500, help='需要训练的轮数，默认为500轮')

# 是否使用GPU进行训练
parser.add_argument('--cuda', action='store_true', help='使用GPU加速')
parser.add_argument('--ngpu', type=int, default=1, help='使用GPU的个数')
parser.add_argument('--pretrained', default='', help="预训练参数路径")

parser.add_argument('--expr_dir', default='expr', help='保存参数的路径位置')
parser.add_argument('--n_test_disp', type=int, default=1, help='进行测试时显示的样本数')
parser.add_argument('--saveInterval', type=int, default=500, help='隔多少迭代val一次')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--beta1', type=float, default=0.5, help='adam优化器的beta1参数')
parser.add_argument('--adam', action='store_true', help='是否是用adam优化器 (默认为rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='是否是用adadelta优化器 (默认为rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='图片修改尺寸时保持长宽比')
parser.add_argument('--manualSeed', type=int, default=4321, help='')
parser.add_argument('--random_sample', action='store_true', help='是否随机采样')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = RawDataset(gt_file=os.path.join(opt.trainroot, 'gt.txt'), gt_dir=opt.trainroot)
assert train_dataset
if not opt.random_sample:
    sampler = randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

test_dataset = RawDataset(
    gt_file=os.path.join(opt.trainroot, 'gt.txt'), gt_dir=opt.trainroot, transform=resizeNormalize((opt.imgW, opt.imgH)))

nclass = len(train_dataset.lexicon) + 1
nc = 1

converter = strLabelConverter(train_dataset.lexicon)
criterion = torch.nn.CTCLoss(blank=0, reduction='none')


# 权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)

if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    crnn.load_state_dict(torch.load(opt.pretrained))
print(crnn)

def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        t, l = converter.encode(cpu_texts)

        # text = t.cuda()
        # length = l.cuda()
        # image = cpu_images.cuda()

        text = t
        length = l
        image = cpu_images

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text.long(), preds_size.long(), length.long()).sum() / float(batch_size)
        #cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        #preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer,train_iter):

    data = train_iter.next()


    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    t, l = converter.encode(cpu_texts)

    # text = t.cuda()
    # length = l.cuda()
    # image = cpu_images.cuda()
    # preds = crnn(image)
    # preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size)).cuda()

    text = t
    length = l
    image = cpu_images

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

    """
    一个批次样本的序列长度preds_size和实际应该对应的字符串长度length
    """
    cost = criterion(preds, text.long(), preds_size.long(), length.long()).sum() / float(batch_size)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    return cost



if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    criterion = criterion.cuda()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

# loss averager
loss_avg = averager()

for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0

    loss_avg.reset()
    while i < len(train_loader):

        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer,train_iter)
        loss_avg.add(cost)
        i += 1

        print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.nepoch, i, len(train_loader), cost))

    # 保存模型
    if epoch % 5 == 0:
        val(crnn, test_dataset, criterion)
        # torch.save(crnn.module.state_dict(), '{0}/CRNN_{1}.pth'.format(opt.expr_dir, epoch))
        torch.save(crnn.state_dict(), '{0}/CRNN_{1}.pth'.format(opt.expr_dir, epoch))
            

