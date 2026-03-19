class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        if inter_channels == 0:
            inter_channels = 1
        self.num_subset = num_subset
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.A = nn.Parameter(torch.stack([torch.tensor(A[i]) for i in range(num_subset)], dim=0), requires_grad=False)

        self.tcn = unit_tcn(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        y = None
        for i in range(self.num_subset):
            z = self.convs[i](x, self.A[i])
            y = z if y is None else y + z
        y = self.relu(self.down(x) + self.tcn(y))
        return y

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = unit_gcn(in_channels, 64, A)
        self.l2 = unit_gcn(64, 64, A)
        self.l3 = unit_gcn(64, 64, A)
        self.l4 = unit_gcn(64, 128, A)
        self.l5 = unit_gcn(128, 128, A)
        self.l6 = unit_gcn(128, 128, A)
        self.l7 = unit_gcn(128, 256, A)
        self.l8 = unit_gcn(256, 256, A)
        self.l9 = unit_gcn(256, 256, A)

        self.tcn1 = MultiScale_TemporalConv(64, 64, kernel_size=5, stride=1, dilations=[1, 2], residual=False)
        self.tcn2 = MultiScale_TemporalConv(64, 64, kernel_size=5, stride=1, dilations=[1, 2])
        self.tcn3 = MultiScale_TemporalConv(64, 64, kernel_size=5, stride=1, dilations=[1, 2])
        self.tcn4 = MultiScale_TemporalConv(64, 128, kernel_size=5, stride=2, dilations=[1, 2])
        self.tcn5 = MultiScale_TemporalConv(128, 128, kernel_size=5, stride=1, dilations=[1, 2])
        self.tcn6 = MultiScale_TemporalConv(128, 128, kernel_size=5, stride=1, dilations=[1, 2])
        self.tcn7 = MultiScale_TemporalConv(128, 256, kernel_size=5, stride=2, dilations=[1, 2])
        self.tcn8 = MultiScale_TemporalConv(256, 256, kernel_size=5, stride=1, dilations=[1, 2])
        self.tcn9 = MultiScale_TemporalConv(256, 256, kernel_size=5, stride=1, dilations=[1, 2])

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.tcn1(self.l1(x)) + self.tcn2(self.l2(self.l1(x)))
        x = self.tcn4(self.l4(x)) + self.tcn5(self.l5(x)) + self.tcn6(self.l6(x))
        x = self.tcn7(self.l7(x)) + self.tcn8(self.l8(x)) + self.tcn9(self.l9(x))

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.fc(x)
        return x
