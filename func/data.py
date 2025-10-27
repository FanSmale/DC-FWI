from torch.utils.data import Dataset, DataLoader, TensorDataset
from math import ceil
import numpy as np
from func.utils import *
from func.show import *
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class DatasetOpenFWI(Dataset):
    """
       Load training data. It contains seismic data and velocity models.
    """
    def __init__(self, para_data_dir, para_train_size, para_start_num, para_train_or_test):

        print("-------------------------------------------------")
        print("Loading the datasets...")

        data_set, label_set = batch_read_npyfile(para_data_dir, ceil(para_train_size / 500), para_start_num, para_train_or_test)

        ###################################################################################
        #                          Vmodel Normalization                                   #
        ###################################################################################
        print("Vmodel Normalization in progress...")
        for i in range(data_set.shape[0]):
            vm = label_set[i][0]
            label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        ###################################################################################
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index]

    def __len__(self):
        return len(self.seismic_data)


def batch_read_npyfile(para_dataset_dir,
                       para_batch_length,
                       para_start,
                       para_train_or_test="train"):

    seismic_list = []
    vmodel_list = []

    for i in range(para_start, para_start + para_batch_length):
        ##############################
        ##    Load Seismic Data     ##
        ##############################

        seismic_filename = para_dataset_dir + '{}_data/seismic/seismic{}.npy'.format(para_train_or_test, i)
        print("Reading: {}".format(seismic_filename))

        seismic_data = np.load(seismic_filename)
        seismic_list.append(seismic_data)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        vmodel_filename = para_dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(para_train_or_test, i)
        print("Reading: {}".format(vmodel_filename))

        vmodel = np.load(vmodel_filename)
        vmodel_list.append(vmodel)

    dataset = np.concatenate(seismic_list, axis=0)
    labelset = np.concatenate(vmodel_list, axis=0)
    # print(dataset.shape) # (train_size, 5, 1000, 70)
    return dataset, labelset


class DatasetSEG(Dataset):
    def __init__(self, para_data_dir, para_train_size, para_start_num, para_train_or_test):

        print("-------------------------------------------------")
        print("Loading the SEG salt datasets...")

        data_set, label_set = batch_read_matfile(para_data_dir, para_train_size, para_start_num, para_train_or_test)

        ###################################################################################
    
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

    for indx, i in enumerate(range(para_start, para_batch_length + para_start)):
        filename_seis = para_dataset_dir + '{}_data/seismic/seismic{}.mat'.format(para_train_or_test, i)
        print("Reading: {}".format(filename_seis))

        # {ndarray:(400,301,29)}
        sei_data = scipy.io.loadmat(filename_seis)["Rec"]

        ## (400, 301, 29) -> (29, 400, 301)

        # {ndarray:(29,301,400)}
        sei_data = sei_data.swapaxes(0, 2)
        # {ndarray:(29,400,301)}
        sei_data = sei_data.swapaxes(1, 2)

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
        vmodel_max_min = np.zeros((para_test_size, 2))
        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")
        # for each sample
        print(data_set.shape[0])  
        # i
        for i in range(para_test_size):
            vm = label_set[i][0]
            vmodel_max_min[i, 0] = np.max(vm)  
            vmodel_max_min[i, 1] = np.min(vm)  
            label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))  # 归一化到[0,1]之间

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
        vmodel_max_min = np.zeros((para_test_size, 2))
        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        # print("Normalization in progress...")
        # for each sample
        # print(data_set.shape[0])  
        # i
        for i in range(para_test_size):
            vm = label_set[i][0]
            vmodel_max_min[i, 0] = np.max(vm) 
            vmodel_max_min[i, 1] = np.min(vm)  

        # ##################################################################################
        # Testing set
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_test_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_test_size, ...]).float())
        self.vmodel_max_min = TensorDataset(torch.from_numpy(vmodel_max_min[:para_test_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index], self.vmodel_max_min[index]

    def __len__(self):
        return len(self.seismic_data)

if __name__ == '__main__':

    dataset_dir = 'E:/pg/pg0/data/OpenWFI/FlatVel-A/'
    data_set = DatasetTestOpenFWI(dataset_dir, 10, 11,  "test")




