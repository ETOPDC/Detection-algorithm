import torch.nn as nn

#基本残差块
class HgResBolck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(HgResBolck, self).__init__()

        self.inplanes = inplanes
        self.outplanes = outplanes
        midplanes =  outplanes //2

        self.bn_1 = nn.BatchNorm2d(inplanes)
        self.conv_1 = nn.Conv2d(inplanes, midplanes, kernel_size=1, stride =stride)
        self.bn_2 = nn.BatchNorm2d(midplanes)
        self.conv_2 = nn.Conv2d(midplanes, midplanes, kernel_size=3, stride=stride)
        self.bn_3 = nn.BatchNorm2d(midplanes)
        self.conv_3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplanes=True)
        if inplanes!= outplanes:
            self.conv_skip = nn.Conv2d(inplanes, outplanes,kernel_size=1, stride=1)


    #Bottle neck
    def forward(self, x):
        residual = x
        out = self.bn_1(x)
        out = self.conv_1(out)
        out = self.relu(out)

        out = self.bn_2(out)
        out = self.conv_2(out)
        out = self.relu(out)

        out = self.bn_3(out)
        out = self.conv_3(out)
        out = self.relu(out)

        if self.inplanes!=self.outplanes:
            residual = self.conv_skip(residual)
        out += residual
        return out

#单个Hourglass Module
class Hourglass(nn.Module):
    def __init__(self, depth, nFeat, nModules, resBlocks):
        super(Hourglass, self).__init__()

        self.depth = depth
        self.nFeat = nFeat
        # num residual modules per location
        self.nModeules = nModules
        self.resBlocks = resBlocks

        self.hg = self._make_hourglass()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_residual(self,n):
        return nn.Sequential(*[self.resBlocks(self.nFeat,self.nFeat) for _ in range(n)])

    def _make_hourglass(self):
        hg = []
        for i in range(self.depth):
            # skip(upper branch); down_path, up_path(lower branch)
            res = [self._make_residual(self.nModeules) for _ in range(3)]
            if i ==(self.depth-1):
                ## extra one for the middle
                res.append(self._make_residual(self.nModeules))
            hg.append(nn.ModuleList(res))

        #hg = [[res,res,...],[],[],[]]
        return nn.ModuleList(hg)


    def _hourglass_forward(self, depth_id, x):
        up_1 = self.hg[depth_id][0](x)
        low_1 = self.downsample(x)
        low_1 = self.hg[depth_id][1](low_1)

        if depth_id == (self.depth -1):
            low_2 = self.hg[depth_id][3](low_1)
        else:
            low_2 = self._hourglass_forward(depth_id+1, low_1)

        low_3 = self.hg[depth_id][2](low_2)
        up_2 = self.upsample(low_3)

        return up_1 + up_2

    def forward(self,x):
        return self._hourglass_forward(0,x)


class HourglassNet(nn.Module):
    def __init__(self, nStacks, nModules, nFeat, nClasses, resBlock=HgResBlock, inplanes=3):
        super(HourglassNet, self).__init__()
        self.nStacks = nStacks
        self.nModules = nModules
        self.nFeat = nFeat
        self.nClasses = nClasses
        self.resBlock = resBlock
        self.inplanes = inplanes
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(nStacks):
            hg.append(Hourglass(depth=4, nFeat=nFeat, nModules=nModules, resBlocks=resBlock))
            res.append(self._make_residual(nModules))
            fc.append(self._make_fc(nFeat, nFeat))
            score.append(nn.Conv2d(nFeat, nClasses, kernel_size=1))
            if i < (nStacks - 1):
                fc_.append(nn.Conv2d(nFeat, nFeat, kernel_size=1))
                score_.append(nn.Conv2d(nClasses, nFeat, kernel_size=1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)
    def _make_head(self):
        self.conv_1 = nn.Conv2d(self.inplanes, 64, kernel_size=7, stride=2, padding=3)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.res_1 = self.resBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.res_2 = self.resBlock(128, 128)
        self.res_3 = self.resBlock(128, self.nFeat)
    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])
    def _make_fc(self, inplanes, outplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True))
    def forward(self, x):
        # head
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.res_1(x)
        x = self.pool(x)
        x = self.res_2(x)
        x = self.res_3(x)
        out = []
        for i in range(self.nStacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < (self.nStacks - 1):
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        return out