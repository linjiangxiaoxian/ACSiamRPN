import torch
import torch.nn as nn
class SKConv(nn.Module):
    def __init__(self, features, WH, M=2, G=1, r=8, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        #TODO:super: 调用父类方法的关键字

        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=1 + i * 2, stride=stride, padding=i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AvgPool2d(int(WH / stride))
        self.gmp = nn.MaxPool2d(int(WH / stride))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)

        fea_s_gap = self.gap(fea_U).squeeze_()
        # TODO:2020.06.10
        fea_s_gap = fea_s_gap.unsqueeze_(0)

        fea_s_gmp = self.gmp(fea_U).squeeze_()
        # TODO:2020.06.10
        fea_s_gmp = fea_s_gmp.unsqueeze_(0)

        fea_z_gap = self.fc(fea_s_gap)
        # TODO:2020.06.10
        fea_z_gap = fea_z_gap.unsqueeze_(0)

        fea_z_gmp = self.fc(fea_s_gmp)
        # TODO:2020.06.10
        fea_z_gmp = fea_z_gmp.unsqueeze_(0)

        for i, fc in enumerate(self.fcs):
            vector_gap = fc(fea_z_gap).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors_gap = vector_gap
            else:
                attention_vectors_gap = torch.cat([attention_vectors_gap, vector_gap], dim=1)
        for i, fc in enumerate(self.fcs):
            vector_gmp = fc(fea_z_gmp).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors_gmp = vector_gmp
            else:
                attention_vectors_gmp = torch.cat([attention_vectors_gmp, vector_gmp], dim=1)
        # TODO:2020.06.10
        attention_vectors_gap = attention_vectors_gap.squeeze_()
        attention_vectors_gap = attention_vectors_gap.unsqueeze_(0)
        attention_vectors_gmp = attention_vectors_gmp.squeeze_()
        attention_vectors_gmp = attention_vectors_gmp.unsqueeze_(0)


        attention_vectors = attention_vectors_gap + attention_vectors_gmp
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)

        #TODO:open these in test

        # attention_vectors = attention_vectors.permute(1,0,2,3)
        # attention_vectors = attention_vectors.unsqueeze(0)
        # TODO:The tensor size in train
        # print('fea_U',fea_U.size())
        # print('fea_s_gap',fea_s_gap.size())
        # print('fea_s_gmp', fea_s_gmp.size())
        # print('fea_z_gap', fea_z_gap.size())
        # print('fea_z_gmp', fea_z_gmp.size())
        # print('attention_vectors_gap',attention_vectors_gap.size())
        # print('attention_vectors_gmp', attention_vectors_gmp.size())

        # #(128,2,256,1,1)
        # print('attention_vectors',attention_vectors.size())
        # #(128,2,256,6,6)
        # print('feas',feas.size())

        # TODO:The tensor size in test

        #(256,2,1,1)
        # print('attention_vectors',attention_vectors.size())
        #(1,2,256,6,6)
        # print('feas',feas.size())

        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v[:,:,1:-1,1:-1].contiguous()
