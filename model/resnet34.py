import torch
import numpy as np

def identity_block2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse,
                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
    filters1, filters2, filters3 = filters

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
    x = tf.layers.conv2d(input_tensor, filters2, kernel_size, use_bias=False, padding='SAME',
                         kernel_initializer=kernel_initializer, name=conv_name_2, reuse=reuse)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
    x = tf.nn.relu(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
    x = tf.layers.conv2d(x, filters3, (kernel_size, kernel_size), use_bias=False, padding='SAME',
                         kernel_initializer=kernel_initializer, name=conv_name_3, reuse=reuse)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)

    x = tf.add(input_tensor, x)
    x = tf.nn.relu(x)
    return x


def conv_block_2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, strides=(2, 2),
                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
    filters1, filters2, filters3 = filters

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
    x = tf.layers.conv2d(input_tensor, filters2, (kernel_size, kernel_size), use_bias=False, strides=strides,
                         padding='SAME', kernel_initializer=kernel_initializer, name=conv_name_2, reuse=reuse)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
    x = tf.nn.relu(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
    x = tf.layers.conv2d(x, filters3, (kernel_size, kernel_size), use_bias=False, padding='SAME',
                         kernel_initializer=kernel_initializer, name=conv_name_3, reuse=reuse)
    x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
    bn_name_4 = 'bn' + str(stage) + '_' + str(block) + '_1x1_shortcut'
    shortcut = tf.layers.conv2d(input_tensor, filters3, (kernel_size, kernel_size), use_bias=False, strides=strides,
                                padding='SAME', kernel_initializer=kernel_initializer, name=conv_name_4, reuse=reuse)
    shortcut = tf.layers.batch_normalization(shortcut, training=is_training, name=bn_name_4, reuse=reuse)

    x = tf.add(shortcut, x)
    x = tf.nn.relu(x)
    return x


def resnet18(input_tensor, is_training=True, pooling_and_fc=True, reuse=False,
             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),inp ='test'):
    x = tf.layers.conv2d(input_tensor, 64, (7, 7), strides=(2, 2), kernel_initializer=kernel_initializer,
                         use_bias=False, padding='SAME', name='conv1_1/3x3_s1', reuse=reuse)
    x = tf.layers.batch_normalization(x, training=is_training, name='bn1_1/3x3_s1', reuse=reuse)
    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool')

    x = tf.nn.relu(x)  # ? need or not

    x1 = identity_block2d(x, 3, [48, 64, 64], stage=2, block='1b', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)
    x1 = identity_block2d(x1, 3, [48, 64, 64], stage=3, block='1c', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)

    x2 = conv_block_2d(x1, 3, [96, 128, 128], stage=3, block='2a', strides=(2, 2), is_training=is_training, reuse=reuse,
                       kernel_initializer=kernel_initializer)
    x2 = identity_block2d(x2, 3, [96, 128, 128], stage=3, block='2b', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)

    x3 = conv_block_2d(x2, 3, [128, 256, 256], stage=4, block='3a', strides=(2, 2), is_training=is_training,
                       reuse=reuse, kernel_initializer=kernel_initializer)
    x3 = identity_block2d(x3, 3, [128, 256, 256], stage=4, block='3b', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)

    x4 = conv_block_2d(x3, 3, [256, 512, 512], stage=5, block='4a', strides=(2, 2), is_training=is_training,
                       reuse=reuse, kernel_initializer=kernel_initializer)
    x4 = identity_block2d(x4, 3, [256, 512, 512], stage=5, block='4b', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)
    feature = tf.nn.avg_pool(x4, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME', name='avg_pool')
    # print('before gap: ', x4)
    x4 = tf.reduce_mean(x4, [1, 2])
    # print('after gap: ', x4)
    # flatten = tf.contrib.layers.flatten(x4)
    # if inp == 'train':
    #     x4 = tf.layers.dropout(x4,rate=0.5)
    prob = tf.layers.dense(x4, 100, reuse=reuse, kernel_initializer=tf.contrib.layers.xavier_initializer())
    # prob = tf.layers.batch_normalization(prob, training=is_training, name='fbn', reuse=reuse)
    # print('prob', prob)

    return prob


def resnet34(input_tensor, is_training=True, pooling_and_fc=True, reuse=False,
             kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
    x = tf.layers.conv2d(input_tensor, 64, (3, 3), strides=(1, 1), kernel_initializer=kernel_initializer,
                         use_bias=False, padding='SAME', name='conv1_1/3x3_s1', reuse=reuse)
    x = tf.layers.batch_normalization(x, training=is_training, name='bn1_1/3x3_s1', reuse=reuse)
    x = tf.nn.relu(x)

    x1 = identity_block2d(x, 3, [48, 64, 64], stage=1, block='1a', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)
    x1 = identity_block2d(x1, 3, [48, 64, 64], stage=1, block='1b', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)
    x1 = identity_block2d(x1, 3, [48, 64, 64], stage=1, block='1c', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)

    x2 = conv_block_2d(x1, 3, [96, 128, 128], stage=2, block='2a', strides=(2, 2), is_training=is_training, reuse=reuse,
                       kernel_initializer=kernel_initializer)
    x2 = identity_block2d(x2, 3, [96, 128, 128], stage=2, block='2b', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)
    x2 = identity_block2d(x2, 3, [96, 128, 128], stage=2, block='2c', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)
    x2 = identity_block2d(x2, 3, [96, 128, 128], stage=2, block='2d', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)

    x3 = conv_block_2d(x2, 3, [128, 256, 256], stage=3, block='3a', strides=(2, 2), is_training=is_training,
                       reuse=reuse, kernel_initializer=kernel_initializer)
    x3 = identity_block2d(x3, 3, [128, 256, 256], stage=3, block='3b', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)
    x3 = identity_block2d(x3, 3, [128, 256, 256], stage=3, block='3c', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)
    x3 = identity_block2d(x3, 3, [128, 256, 256], stage=3, block='3d', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)
    x3 = identity_block2d(x3, 3, [128, 256, 256], stage=3, block='3e', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)
    x3 = identity_block2d(x3, 3, [128, 256, 256], stage=3, block='3f', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)

    x4 = conv_block_2d(x3, 3, [256, 512, 512], stage=4, block='4a', strides=(2, 2), is_training=is_training,
                       reuse=reuse, kernel_initializer=kernel_initializer)
    x4 = identity_block2d(x4, 3, [256, 512, 512], stage=4, block='4b', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)
    x4 = identity_block2d(x4, 3, [256, 512, 512], stage=4, block='4c', is_training=is_training, reuse=reuse,
                          kernel_initializer=kernel_initializer)
    feature =tf.nn.avg_pool(x4,ksize=[1,7,7,1],strides=[1,1,1,1],padding='SAME',name='avg_pool')
    # print('before gap: ', x4)
    x4 = tf.reduce_mean(x4, [1, 2])
    # print('after gap: ', x4)
    # flatten = tf.contrib.layers.flatten(x4)
    prob = tf.layers.dense(x4, 100, reuse=reuse, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           bias_initializer=tf.zeros_initializer(),name='dense')
    # prob = tf.layers.batch_normalization(prob, training=is_training, name='fbn', reuse=reuse)
    # print('prob', prob)

    return prob

def prepare_network(input_tensor, is_training=True, pooling_and_fc=True, reuse=tf.AUTO_REUSE,
             kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
    mean_img = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    with tf.variable_scope('ResNet34', reuse=tf.AUTO_REUSE):
        score = resnet18(input_tensor-mean_img, is_training=is_training, reuse=tf.AUTO_REUSE,
                        kernel_initializer=kernel_initializer,inp='train')
        scope = tf.get_variable_scope()
        scope.reuse_variables()
    train_vars = tf.trainable_variables(scope='ResNet34')

    with tf.variable_scope('ResNet34_stored', reuse=tf.AUTO_REUSE):
        score_stored = resnet18(input_tensor-mean_img, is_training=False, reuse=tf.AUTO_REUSE,
                        kernel_initializer=None,inp='test')
        scope_stored = tf.get_variable_scope()
        scope_stored.reuse_variables()
    train_vars_stored = tf.trainable_variables(scope='ResNet34_stored')
    # train_vars_stored = tf.get_collection(tf.GraphKeys.WEIGHTS,scope='ResNet34_stored')

    g_list = tf.global_variables()
    g_list_stored_loc = int(len(g_list)/2)
    g_list_new = g_list[:g_list_stored_loc]
    g_list_stored = g_list[g_list_stored_loc:]
    bn_moving_vars = [g for g in g_list_new if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list_new if 'moving_variance' in g.name]
    train_vars += bn_moving_vars
    bn_moving_vars_stored = [g for g in g_list_stored if 'moving_mean' in g.name]
    bn_moving_vars_stored += [g for g in g_list_stored if 'moving_variance' in g.name]
    train_vars_stored += bn_moving_vars_stored
    return score,train_vars,score_stored,train_vars_stored

def reading_data_and_preparing_network(x_input):
    ### Network and loss function
    mean_img = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    with tf.variable_scope('ResNet34'):
        with tf.device('/gpu:0'):
            scores =resnet18(x_input-mean_img,is_training=False, reuse=tf.AUTO_REUSE, kernel_initializer=None,inp = 'test')
            graph = tf.get_default_graph()
            op_feature_map_name = 'ResNet34/' + 'avg_pool'
            op_feature_map_ = graph.get_operation_by_name(op_feature_map_name).outputs[0]
            op_feature_map = op_feature_map_[:,0,0,:]
    var_list = tf.trainable_variables(scope='ResNet34')
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list)
    return scores, op_feature_map,saver


def load_class_in_feature_space(num,sess, scores,op_feature_map, label_batch,file_name_batch):
    label_dico = []
    logits = []
    processed_file = []
    Dtot=[]
    for i in range(int(np.ceil(num/64))):
        sc, feat_map_tmp,label_test,pic_name= sess.run([scores, op_feature_map,label_batch,file_name_batch])
        processed_file.extend(pic_name)
        label_dico.extend(label_test)
        logits.append(sc)
        Dtot.append((feat_map_tmp.T) / np.linalg.norm(feat_map_tmp.T, axis=0))  # np.linalg.norm 二范数 归一化
    # logits = np.concatenate(logits)
    label_dico = np.array(label_dico)
    processed_file = np.array(processed_file)
    Dtot = np.concatenate(Dtot, axis=1) # 512 * 样本数量
    return Dtot, label_dico, processed_file