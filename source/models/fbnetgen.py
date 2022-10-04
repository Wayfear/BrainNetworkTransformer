import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear, GRU
from omegaconf import DictConfig
from .base import BaseModel


class GruKRegion(nn.Module):

    def __init__(self, kernel_size=8, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        self.gru = GRU(kernel_size, kernel_size, layers,
                       bidirectional=True, batch_first=True)

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            Linear(kernel_size*2, kernel_size),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(kernel_size, out_size)
        )

    def forward(self, raw):

        b, k, d = raw.shape

        x = raw.view((b*k, -1, self.kernel_size))

        x, h = self.gru(x)

        x = x[:, -1, :]

        x = x.view((b, k, -1))

        x = self.linear(x)
        return x


class ConvKRegion(nn.Module):

    def __init__(self, k=1, out_size=8, kernel_size=8, pool_size=16, time_series=512):
        super().__init__()
        self.conv1 = Conv1d(in_channels=k, out_channels=32,
                            kernel_size=kernel_size, stride=2)

        output_dim_1 = (time_series-kernel_size)//2+1

        self.conv2 = Conv1d(in_channels=32, out_channels=32,
                            kernel_size=8)
        output_dim_2 = output_dim_1 - 8 + 1
        self.conv3 = Conv1d(in_channels=32, out_channels=16,
                            kernel_size=8)
        output_dim_3 = output_dim_2 - 8 + 1
        self.max_pool1 = MaxPool1d(pool_size)
        output_dim_4 = output_dim_3 // pool_size * 16
        self.in0 = nn.InstanceNorm1d(time_series)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)

        self.linear = nn.Sequential(
            Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(32, out_size)
        )

    def forward(self, x):

        b, k, d = x.shape

        x = torch.transpose(x, 1, 2)

        x = self.in0(x)

        x = torch.transpose(x, 1, 2)
        x = x.contiguous()

        x = x.view((b*k, 1, d))

        x = self.conv1(x)

        x = self.in1(x)
        x = self.conv2(x)

        x = self.in2(x)
        x = self.conv3(x)

        x = self.in3(x)
        x = self.max_pool1(x)

        x = x.view((b, k, -1))

        x = self.linear(x)

        return x


class Embed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def forward(self, x):

        m = torch.einsum('ijk,ipk->ijp', x, x)

        m = torch.unsqueeze(m, -1)

        return m


class GNNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)

        self.fcn = nn.Sequential(
            nn.Linear(8*roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )

    def forward(self, m, node_feature):
        bz = m.shape[0]

        x = torch.einsum('ijk,ijp->ijp', m, node_feature)

        x = self.gcn(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn1(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn2(x)

        x = self.bn3(x)

        x = x.view(bz, -1)

        return self.fcn(x)


class Embed2GraphByLinear(nn.Module):

    def __init__(self, input_dim, roi_num=360):
        super().__init__()

        self.fc_out = nn.Linear(input_dim * 2, input_dim)
        self.fc_cat = nn.Linear(input_dim, 1)

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot

        off_diag = np.ones([roi_num, roi_num])
        rel_rec = np.array(encode_onehot(
            np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(
            np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).cuda()
        self.rel_send = torch.FloatTensor(rel_send).cuda()

    def forward(self, x):

        batch_sz, region_num, _ = x.shape
        receivers = torch.matmul(self.rel_rec, x)

        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=2)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        x = torch.relu(x)

        m = torch.reshape(
            x, (batch_sz, region_num, region_num, -1))
        return m


class FBNETGEN(BaseModel):

    def __init__(self, config: DictConfig):
        super().__init__()

        assert config.model.extractor_type in ['cnn', 'gru']
        assert config.model.graph_generation in ['linear', 'product']
        assert config.dataset.timeseries_sz % config.model.window_size == 0

        self.graph_generation = config.model.graph_generation
        if config.model.extractor_type == 'cnn':
            self.extract = ConvKRegion(
                out_size=config.model.embedding_size, kernel_size=config.model.window_size,
                time_series=config.dataset.timeseries_sz)
        elif config.model.extractor_type == 'gru':
            self.extract = GruKRegion(
                out_size=config.model.embedding_size, kernel_size=config.model.window_size,
                layers=config.model.num_gru_layers)
        if self.graph_generation == "linear":
            self.emb2graph = Embed2GraphByLinear(
                config.model.embedding_size, roi_num=config.dataset.node_sz)
        elif self.graph_generation == "product":
            self.emb2graph = Embed2GraphByProduct(
                config.model.embedding_size, roi_num=config.dataset.node_sz)

        self.predictor = GNNPredictor(
            config.dataset.node_feature_sz, roi_num=config.dataset.node_sz)

    def forward(self, time_seires, node_feature):
        x = self.extract(time_seires)
        x = F.softmax(x, dim=-1)
        m = self.emb2graph(x)
        m = m[:, :, :, 0]

        return self.predictor(m, node_feature), m
