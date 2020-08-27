    model = DenseNet(growth_rate=12, block_config=(4, 8, 16, 12),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=3).cuda()