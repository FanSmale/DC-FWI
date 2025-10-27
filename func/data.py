from torch.utils.data import Dataset, DataLoader, TensorDataset
from math import ceil
import numpy as np
from func.utils import *
from func.show import *
import os

"""
这是设置环境变量的一种方法，其中"kmp_duplicate_lib_ok"对应的是一个Intel MKL库的环境变量，
它的作用是控制Intel MKL库是否允许重复链接。
Intel MKL是Intel公司提供的一套数学库，它包括线性代数、傅里叶分析、随机数生成、优化等多个方面的功能。
"""
# 如果不添加该语句程序可能会出现OMP错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 加载训练数据
class DatasetOpenFWI(Dataset):
    """
       Load training data. It contains seismic data and velocity models.
    """
    def __init__(self, para_data_dir, para_train_size, para_start_num, para_train_or_test):
        """

        :param para_data_dir: 数据集目录
        :param para_train_size: 训练集大小
        :param para_start_num: 起始，1 for train, 11 for test
        :param para_train_or_test: 训练or测试
        """
        print("-------------------------------------------------")
        print("Loading the datasets...")

        # ceil（）向上取整
        # 批量读取npy文件
        data_set, label_set = batch_read_npyfile(para_data_dir, ceil(para_train_size / 500), para_start_num, para_train_or_test)

        ###################################################################################
        #                          Vmodel Normalization                                   #
        ###################################################################################
        print("Vmodel Normalization in progress...")
        # 最小-最大归一化适用于数据分布有明显边界的情况
        # 遍历每个数据，从0-(train_size-1)，将速度模型归一化到[0,1]
        # data_set.shape[0]取第0个值，即train_size, 总共有多少个数据
        for i in range(data_set.shape[0]):
            vm = label_set[i][0]
            label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        ###################################################################################
        # torch.from_numpy()可以将NumPy数组直接转换为PyTorch张量，同时保持两者共享底层数据内存的特性。
        # 这意味着对转换后的PyTorch张量进行的任何修改都会反映到原始的NumPy数组上。
        # TensorDataset用于将张量组合成数据集的工具类。它的作用是将多个张量作为输入，构建一个能够以样本对的形式提供数据的数据集。
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index]

    def __len__(self):
        return len(self.seismic_data)


#批量读取 .npy 文件中的地震数据和速度模型
def batch_read_npyfile(para_dataset_dir,
                       para_batch_length,
                       para_start,
                       para_train_or_test="train"):
    """

    :param para_dataset_dir: 数据集路径
    :param para_batch_length: 有多少个npy文件
    :param para_start: 起始，1 for train, 11 for test
    :param para_train_or_test: 读取的数据是用于训练还是测试
    :return:  a pair: (dataset, labelset)
              dataset（500,5,1000,70）地震数据
              labelset（500,1,70,70）速度模型
    """

    seismic_list = []
    vmodel_list = []

    # 1~要读取多少个npy文件
    for i in range(para_start, para_start + para_batch_length):
        ##############################
        ##    Load Seismic Data     ##
        ##############################

        # 地震数据的路径
        seismic_filename = para_dataset_dir + '{}_data/seismic/seismic{}.npy'.format(para_train_or_test, i)
        print("Reading: {}".format(seismic_filename))

        seismic_data = np.load(seismic_filename)
        seismic_list.append(seismic_data)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        # 速度模型的路径
        vmodel_filename = para_dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(para_train_or_test, i)
        print("Reading: {}".format(vmodel_filename))

        vmodel = np.load(vmodel_filename)
        vmodel_list.append(vmodel)

    # np.concatenate()函数：拼接。第一个数据应该是一个数组形式的，所以必须用 () 或者 [] 符号括起来。
    # axis=0是行扩充
    dataset = np.concatenate(seismic_list, axis=0)
    labelset = np.concatenate(vmodel_list, axis=0)
    # print(dataset.shape) # (train_size, 5, 1000, 70)
    return dataset, labelset


class DatasetSEG(Dataset):
    def __init__(self, para_data_dir, para_train_size, para_start_num, para_train_or_test):
        """

        :param para_data_dir: 数据集目录
        :param para_train_size: 训练集大小
        :param para_start_num: 起始，1 for train, 1601 for test, 1600+100（模拟数据），130+10（SEG）
        :param para_train_or_test: 训练or测试
        """
        print("-------------------------------------------------")
        print("Loading the SEG salt datasets...")

        # ceil（）向上取整
        # 批量读取npy文件
        data_set, label_set = batch_read_matfile(para_data_dir, para_train_size, para_start_num, para_train_or_test)

        ###################################################################################
        #                          Vmodel Normalization                                   #
        ###################################################################################
        # print("Vmodel Normalization in progress...")
        # 最小-最大归一化适用于数据分布有明显边界的情况
        # 遍历每个数据，从0-(train_size-1)，将速度模型归一化到[0,1]
        # data_set.shape[0]取第0个值，即train_size, 总共有多少个数据
        # for i in range(data_set.shape[0]):
        #     vm = label_set[i][0]
        #     label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        # 标准化（Z-score归一化）：将数据转换为均值为0，标准差为1的分布
        # for i in range(data_set.shape[0]):
        #     vm = label_set[i][0]
        #     # 计算均值和标准差
        #     mean_val = np.mean(vm)
        #     std_val = np.std(vm)
        #     # 标准化：(x - mean) / std
        #     label_set[i][0] = (vm - mean_val) / (std_val + 1e-8)  # 添加小常数避免除零
        ###################################################################################
        # torch.from_numpy()可以将NumPy数组直接转换为PyTorch张量，同时保持两者共享底层数据内存的特性。
        # 这意味着对转换后的PyTorch张量进行的任何修改都会反映到原始的NumPy数组上。
        # TensorDataset用于将张量组合成数据集的工具类。它的作用是将多个张量作为输入，构建一个能够以样本对的形式提供数据的数据集。
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index]

    def __len__(self):
        return len(self.seismic_data)


def batch_read_matfile(para_dataset_dir,
                       para_batch_length,
                       para_start,
                       para_train_or_test="train",
                       para_inchannels=29):
    """

    :param para_dataset_dir: Path to the dataset
    :param para_batch_length:
    :param para_start:
    :param para_train_or_test:
    :param para_channels:
    :return:
    """

    # {ndarray:(batchsize, 29, 400, 301)}
    data_set = np.zeros([para_batch_length, para_inchannels, DataDim[0], DataDim[1]])
    # {ndarray:(batchsize, 1, 201, 301)}
    label_set = np.zeros([para_batch_length, OutChannel, ModelDim[0], ModelDim[1]])
    # clabel_set = np.zeros([para_batch_length, OutChannel, ModelDim[0], ModelDim[1]])


    # 索引-index:0  值-i:1
    for indx, i in enumerate(range(para_start, para_batch_length + para_start)):

        # Load Seismic Data
        # SimulateData：1600的模拟数据用于训练，1601-1700用于测试
        # SEGSaltData：130的真实数据用于训练，1-10用于测试
        filename_seis = para_dataset_dir + '{}_data/seismic/seismic{}.mat'.format(para_train_or_test, i)
        print("Reading: {}".format(filename_seis))

        # {ndarray:(400,301,29)}
        sei_data = scipy.io.loadmat(filename_seis)["Rec"]

        ## (400, 301, 29) -> (29, 400, 301)

        # {ndarray:(29,301,400)}
        sei_data = sei_data.swapaxes(0, 2)
        # {ndarray:(29,400,301)}
        sei_data = sei_data.swapaxes(1, 2)

        # index第几个数据  ch第几个通道数
        # data_set 四维（index，ch，400，301）
        for ch in range(para_inchannels):
            data_set[indx, ch, ...] = sei_data[ch, ...]

        # Load Velocity Model
        # SimulateData
        filename_label = para_dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(para_train_or_test, i)
        # SEGSaltData
        # filename_label = para_dataset_dir + '{}_data/vmodel/svmodel{}.mat'.format(para_train_or_test, i)
        print("Reading: {}".format(filename_label))

        vm_data = scipy.io.loadmat(filename_label)['vmodel']  # SimulateData:vmodel; SEGSaltData:svmodel
        # print(vm_data)
        label_set[indx, 0, ...] = vm_data
        #clabel_set[indx, 0, ...] = extract_contours(vm_data)


    return data_set, label_set
    # return data_set, [label_set, label_set]


class DatasetTestOpenFWI(Dataset):
    """
       # Load the test data. It includes seismic_data, vmodel, and vmodel_max_min.
       # vmodel_max_min is used for inverting normalization.
    """
    def __init__(self, para_data_dir, para_test_size, para_start_num, para_train_or_test):
        print("---------------------------------")
        print("Loading the  test datasets...")

        # {ndarray:(test_size,1,70,70)}
        data_set, label_set = batch_read_npyfile(para_data_dir, ceil(para_test_size / 500), para_start_num, para_train_or_test)
        # {ndarray:(test_size,2)}
        # np.zeros()返回来一个给定形状和类型的用0填充的数组
        # 两列：第一列放最大值，第二列放最小值
        vmodel_max_min = np.zeros((para_test_size, 2))
        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")
        # for each sample
        print(data_set.shape[0])  # 值为总数据量,例如500
        # i
        for i in range(para_test_size):
            # {ndarray:(70,70)},正常速度值
            vm = label_set[i][0]
            # print(vm)
            vmodel_max_min[i, 0] = np.max(vm)  # 第i行第一列取到当前速度模型的最大值
            vmodel_max_min[i, 1] = np.min(vm)  # 第i行第二列取到当前速度模型的最小值
            label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))  # 归一化到[0,1]之间
            # print(label_set[i][0]) 查看已归一化后的数值
            # print(label_set[i][0].shape) # (70,70)


        # ##################################################################################
        # Testing set
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_test_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_test_size, ...]).float())
        self.vmodel_max_min = TensorDataset(torch.from_numpy(vmodel_max_min[:para_test_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index], self.vmodel_max_min[index]

    def __len__(self):
        return len(self.seismic_data)


class DatasetTestSEG(Dataset):
    """
       # Load the test data. It includes seismic_data, vmodel, and vmodel_max_min.
       # vmodel_max_min is used for inverting normalization.
    """
    def __init__(self, para_data_dir, para_test_size, para_start_num, para_train_or_test):
        print("---------------------------------")
        print("Loading the  test datasets...")

        # {ndarray:(test_size,5,1000,70)}
        data_set, label_set = batch_read_matfile(para_data_dir, para_test_size, para_start_num, para_train_or_test)
        # {ndarray:(test_size,2)}
        # np.zeros()返回来一个给定形状和类型的用0填充的数组
        # 两列：第一列放最大值，第二列放最小值
        vmodel_max_min = np.zeros((para_test_size, 2))
        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        # print("Normalization in progress...")
        # for each sample
        # print(data_set.shape[0])  # 值为总数据量,例如500
        # i
        for i in range(para_test_size):
            # {ndarray:(201,301)},正常速度值
            vm = label_set[i][0]
            # print(vm)
            vmodel_max_min[i, 0] = np.max(vm)  # 第i行第一列取到当前速度模型的最大值
            vmodel_max_min[i, 1] = np.min(vm)  # 第i行第二列取到当前速度模型的最小值
        #     label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))  # 归一化到[0,1]之间
            # print(label_set[i][0]) 查看已归一化后的数值
            # print(label_set[i][0].shape) # (70,70)

        # ##################################################################################
        # Testing set
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_test_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_test_size, ...]).float())
        self.vmodel_max_min = TensorDataset(torch.from_numpy(vmodel_max_min[:para_test_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index], self.vmodel_max_min[index]

    def __len__(self):
        return len(self.seismic_data)


class Dataset_openfwi4_test(Dataset):
    '''
       # Load the test data including velocity model edge.
       # It includes seismic_data, vmodel, edge, and vmodel_max_min.
       # vmodel_max_min is used for inverting normalization.
    '''

    def __init__(self, para_data_dir, para_train_size, para_start_num, para_seismic_flag, para_train_or_test):
        # start_num: 1 for train, 11 for test
        print("---------------------------------")
        print("· Loading the datasets...")

        data_set, label_set, edge_set = batch_read_npyfile4(para_data_dir, para_start_num, ceil(para_train_size / 500), para_seismic_flag, para_train_or_test)
        vmodel_max_min = np.zeros((para_train_size, 2))
        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")
        # for each sample
        for i in range(data_set.shape[0]):
            vm = label_set[i][0]
            vmodel_max_min[i, 0] = np.max(vm)
            vmodel_max_min[i, 1] = np.min(vm)
            label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        # ##################################################################################
        # Training set
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())
        self.edge = TensorDataset(torch.from_numpy(edge_set[:para_train_size, ...]).to(torch.uint8))
        self.vmodel_max_min = TensorDataset(torch.from_numpy(vmodel_max_min[:para_train_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index], self.edge[index], self.vmodel_max_min[index]

    def __len__(self):
        return len(self.seismic_data)


def batch_read_npyfile4(para_dataset_dir,
                       para_start,
                       para_batch_length,
                       para_seismic_flag = "seismic",
                       para_train_or_test = "train"):
    '''
    Batch read seismic gathers and velocity models for .npy file
    including velocity model edge.

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :return:                        a pair: (seismic data, velocity model, contour of velocity model)
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity
                                    model are all (number of read data * 500, channel, height, width)
                                    dataset （500,5,1000,70）  地震数据  未经任何处理
                                    labelset, （500,1,70,70）  速度模型  未经任何处理
                                    edgeset    (500,1,70,70)   二值边缘图像
    '''

    dataset_list = []
    labelset_list = []
    edgeset_list = []

    for i in range(para_start, para_start + para_batch_length):
        ##############################
        ##    Load Seismic Data     ##
        ##############################

        # Determine the seismic data path in the dataset
        filename_seis = para_dataset_dir + '{}_data/{}/seismic{}.npy'.format(para_train_or_test, para_seismic_flag, i)
        print("Reading: {}".format(filename_seis))

        datas = np.load(filename_seis).astype(np.float32)
        dataset_list.append(datas)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        # Determine the velocity model path in the dataset
        filename_label = para_dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(para_train_or_test, i)
        print("Reading: {}".format(filename_label))
        labels = np.load(filename_label).astype(np.float32)
        labelset_list.append(labels)

        ###################################
        ##    Generating Velocity Edge   ##
        ###################################

        print("Generating velocity model profile......")
        edges = np.zeros([500, OutChannel, ModelDim[0], ModelDim[1]])
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                edges[i, j, ...] = extract_contours(labels[i, j, ...])
        edgeset_list.append(edges.astype(np.uint8))

    dataset = np.concatenate(dataset_list, axis=0)
    labelset = np.concatenate(labelset_list, axis=0)
    edgeset = np.concatenate(edgeset_list, axis=0)

    return dataset, labelset, edgeset


if __name__ == '__main__':

    dataset_dir = 'E:/pg/pg0/data/OpenWFI/FlatVel-A/'
    data_set = DatasetTestOpenFWI(dataset_dir, 10, 11,  "test")



