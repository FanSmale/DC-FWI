from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick

mpl.use('TkAgg')
import scipy

font21 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
}

font18 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 35,
}


def pain_openfwi_seismic_data(para_seismic_data):
    """
    Plotting seismic dataset images of openfwi dataset

    :param para_seismic_data:   Seismic dataset (1000 x 70) (numpy)
    """
    data = cv2.resize(para_seismic_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)
    fig, ax = plt.subplots(figsize=(6.1, 8), dpi = 60)
    im = ax.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)

    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)

    ax.set_xticks(np.linspace(0, 0.7, 7))
    ax.set_yticks(np.linspace(0, 1.0, 9))
    ax.set_xticklabels(labels=[0, 0.11, 0.23, 0.35, 0.47, 0.59, 0.7], size=21)
    ax.set_yticklabels(labels=[0, 0.12, 0.25, 0.37, 0.5, 0.63, 0.75, 0.88, 1.0], size=21)
    # ax.set_xticks(np.linspace(0, 0.7, 5))
    # ax.set_yticks(np.linspace(0, 1.0, 5))
    # ax.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
    # ax.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    plt.show()
    plt.close()


def pain_seg_seismic_data(para_seismic_data):
    """
    Plotting seismic dataset images of SEG salt datasets

    :param para_seismic_data:  Seismic dataset (400 x 301) (numpy)
    :param is_colorbar: Whether to add a color bar (1 means add, 0 is the default, means don't add)
    """
    fig, ax = plt.subplots(figsize=(6.2, 8), dpi=120)

    im = ax.imshow(para_seismic_data, extent=[0, 300, 400, 0], cmap=plt.cm.seismic, vmin=-0.4, vmax=0.44)

    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)

    ax.set_xticks(np.linspace(0, 300, 5))
    ax.set_yticks(np.linspace(0, 400, 5))
    ax.set_xticklabels(labels=[0, 0.75, 1.5, 2.25, 3.0], size=21)
    ax.set_yticklabels(labels=[0.0, 0.50, 1.00, 1.50, 2.00], size=21)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.32)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    plt.show()


def pain_openfwi_velocity_model(num, para_velocity_model, test_result_dir, min_velocity, max_velocity):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_velocity_model: Velocity model (70 x 70) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity model
    :param max_velocity:        Lower limit of velocity in the velocity model
    :return:
    '''

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(para_velocity_model, extent=[0, 0.7, 0.7, 0], vmin=min_velocity, vmax=max_velocity)

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=42)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=42)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.tick_params(axis='y', which='major', pad=15)

    plt.rcParams['font.size'] = 46  # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.8)
    plt.colorbar(im, ax=ax, cax=cax, orientation='vertical',
                 ticks=np.linspace(min_velocity, max_velocity, 5), format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
    plt.subplots_adjust(bottom=0.10, top=0.95, left=0.1, right=0.9)

    # plt.show()
    plt.savefig(test_result_dir + 'pd' + str(num))  
    plt.close('all')

def pain_seg_velocity_model(num, para_velocity_model, test_result_dir, min_velocity, max_velocity):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_velocity_model: Velocity model (201 x 301) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity model
    :param max_velocity:        Lower limit of velocity in the velocity model
    :return:
    '''

    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(para_velocity_model, extent=[0, 3.00, 2.00, 0], vmin=min_velocity, vmax=max_velocity)

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    # ax.set_xticks(ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], labels=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], fontsize=32)
    # ax.set_yticks(ticks=[0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00], labels=[0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00], fontsize=32)
    ax.set_xticks(np.linspace(0, 3, 7))
    ax.set_yticks(np.linspace(0, 2, 5))
    ax.set_xticklabels(labels=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], size=50)
    ax.set_yticklabels(labels=[0, 0.5, 1.0, 1.5, 2.0], size=50)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.tick_params(axis='y', which='major', pad=15)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    plt.rcParams['font.size'] = 48  # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.8)
    plt.colorbar(im, ax=ax, cax=cax, orientation='vertical',
                 ticks=np.linspace(min_velocity, max_velocity,5), format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
    plt.subplots_adjust(bottom=0.10, top=0.95, left=0.1, right=0.9)
    # plt.show()
    plt.savefig(test_result_dir + 'pd' + str(num))  
    plt.close('all')

def plot_openfwi_velocity_compare(num, output, target, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    im = ax[0].matshow(output, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title('Prediction', y=1.1)
    ax[1].matshow(target, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title('Ground Truth', y=1.1)

    for axis in ax:
        # axis.set_xticks(range(0, 70, 10))
        # axis.set_xticklabels(range(0, 1050, 150))
        # axis.set_yticks(range(0, 70, 10))
        # axis.set_yticklabels(range(0, 1050, 150))

        axis.set_xticks(range(0, 70, 10))
        axis.set_xticklabels(range(0, 700, 100))
        axis.set_yticks(range(0, 70, 10))
        axis.set_yticklabels(range(0, 700, 100))


        axis.set_ylabel('Depth (km)', fontsize=12)
        axis.set_xlabel('Position (km)', fontsize=12)

    fig.colorbar(im, ax=ax, shrink=0.75, label='Velocity(m/s)')
    plt.savefig(test_result_dir + 'PD' + str(num)) 
    plt.close('all')


def plot_seg_velocity_compare(num, output, target, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(1, 2, figsize=(11,5))
    im = ax[0].matshow(output, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title('Prediction', y=1.15)
    ax[1].matshow(target, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title('Ground Truth', y=1.15)

    for axis in ax:

        axis.set_xticks(range(0, 301, 50))
        axis.set_xticklabels([0.0,0.5,1.0,1.5,2.0,2.5,3.0],size=8)
        axis.set_yticks(range(0, 201, 25))
        axis.set_yticklabels([0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00],fontsize=8)

        axis.set_ylabel('Depth (km)', fontsize=11)
        axis.set_xlabel('Position (km)', fontsize=11)

    fig.colorbar(im, ax=ax, shrink=0.55, label='Velocity(m/s)')
    plt.savefig(test_result_dir + 'PD' + str(num)) 
    plt.close('all')

def plot_openfwi_velocity_image_1(num, output, target, test_result_dir):
    plt.figure(figsize=(6.5, 6))
    column_index = 35
    pixel_values1, pixel_values2 = [], []
    for y in range(output.shape[0]):
        pixel_value1 = output[y, column_index]
        pixel_value2 = target[y, column_index]
        pixel_values1.append(pixel_value1)
        pixel_values2.append(pixel_value2)
    plt.plot(pixel_values1, color='blue', linewidth=1, label='Prediction')
    plt.plot(pixel_values2, color='red', linewidth=1, label='Ground Truth')
    plt.legend(fontsize=12)
    plt.xticks(range(0, 80, 10), fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Depth (m)', fontsize=12)
    plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.savefig(test_result_dir + 'PDD' + str(num) + '.png')

def plot_openfwi_velocity_image_2(num, output1, output2, target, test_result_dir):
    plt.figure(figsize=(15, 12))
    # int
    column_index = 45
    pixel_values1, pixel_values2, pixel_values3 = [], [], []
    for y in range(output1.shape[0]):
        pixel_value1 = target[y, column_index]
        pixel_value2 = output2[y, column_index]
        pixel_value3 = output1[y, column_index]
        # pixel_value4 = output3[y, column_index]
        pixel_values1.append(pixel_value1)
        pixel_values2.append(pixel_value2)
        pixel_values3.append(pixel_value3)
        # pixel_values4.append(pixel_value4)
    plt.plot(pixel_values1, color='red', linewidth=3, label='Ground Truth')
    plt.plot(pixel_values2, color='blue', linewidth=3, label='DC-FWI')
    plt.plot(pixel_values3, color='orange', linewidth=3, label='w/o dense connection')
    # plt.plot(pixel_values4, color='purple', linewidth=2, label='DCNet')
    plt.legend(fontsize=36, loc='lower right')

    plt.xticks(range(0, 71, 10), [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], fontsize=50)
    plt.yticks(fontsize=50)
    # plt.xlabel('Depth (km)', fontsize=32)
    # plt.ylabel('Velocity (m/s)', fontsize=32)

    plt.savefig(test_result_dir + 'PDD' + str(num) + '_' + str(column_index) + '.png')

def plot_openfwi_velocity_image_3(num, output1, output2, output3, target, test_result_dir):
    plt.figure(figsize=(12, 10))
    # int
    column_index = 25
    pixel_values1, pixel_values2, pixel_values3, pixel_values4 = [], [], [], []
    for y in range(output1.shape[0]):
        pixel_value1 = target[y, column_index]
        pixel_value2 = output1[y, column_index]
        pixel_value3 = output2[y, column_index]
        pixel_value4 = output3[y, column_index]
        pixel_values1.append(pixel_value1)
        pixel_values2.append(pixel_value2)
        pixel_values3.append(pixel_value3)
        pixel_values4.append(pixel_value4)
    plt.plot(pixel_values1, color='black', linewidth= 2, label='GT')
    plt.plot(pixel_values2, color='orange', linewidth=2, label='InversionNet')
    plt.plot(pixel_values3, color='green', linewidth=2, label='DD-Net70')
    plt.plot(pixel_values4, color='red', linewidth=2, label='DC-FWI')
    plt.legend(fontsize=30)

    plt.xticks(range(0, 71, 10), [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], fontsize=36)
    plt.yticks(fontsize=36)
    # plt.xlabel('Depth (km)', fontsize=12)
    # plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.savefig(test_result_dir + 'PDD' + str(num) + '_' + str(column_index) + '.png')


def plot_openfwi_velocity_image_4(num, output1, output2, output3, output4, target, test_result_dir):
    plt.figure(figsize=(12, 10))
    # int
    column_index = 45
    pixel_values1, pixel_values2, pixel_values3, pixel_values4, pixel_values5 = [], [], [], [], []
    for y in range(output1.shape[0]):
        pixel_value1 = target[y, column_index]
        pixel_value2 = output1[y, column_index]
        pixel_value3 = output2[y, column_index]
        pixel_value4 = output3[y, column_index]
        pixel_value5 = output4[y, column_index]

        pixel_values1.append(pixel_value1)
        pixel_values2.append(pixel_value2)
        pixel_values3.append(pixel_value3)
        pixel_values4.append(pixel_value4)
        pixel_values5.append(pixel_value5)

    plt.plot(pixel_values1, color='black', linewidth= 2, label='GT')
    plt.plot(pixel_values2, color='orange', linewidth=2, label='InversionNet')
    plt.plot(pixel_values3, color='green', linewidth=2, label='DD-Net70')
    plt.plot(pixel_values4, color='blue', linewidth=2, label='VelocityGAN')
    plt.plot(pixel_values5, color='red', linewidth=2, label='DC-FWI')

    plt.legend(fontsize=30)

    plt.xticks(range(0, 71, 10), [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], fontsize=36)
    plt.yticks(fontsize=36)
    # plt.xlabel('Depth (km)', fontsize=12)
    # plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.savefig(test_result_dir + 'PDD' + str(num) + '_' + str(column_index) + '.png')

def plot_seg_velocity_image(num, output, target, test_result_dir):
    plt.figure(figsize=(7, 6))
    column_index = 65
    pixel_values1, pixel_values2 = [], []
    for y in range(output.shape[0]):
        pixel_value1 = output[y, column_index]
        pixel_value2 = target[y, column_index]
        pixel_values1.append(pixel_value1)
        pixel_values2.append(pixel_value2)
    plt.plot(pixel_values1, color='blue', linewidth=1, label='Prediction')
    plt.plot(pixel_values2, color='red', linewidth=1, label='Ground Truth')
    plt.legend(fontsize=12)
    plt.xticks(range(0, 225, 25), fontsize=8)
    plt.yticks(fontsize=12)
    plt.xlabel('Depth (m)', fontsize=12)
    plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.subplots_adjust(top=0.92, left=0.15, right=0.95)
    plt.savefig(test_result_dir + 'PDD' + str(num) + '.png')


def plot_seg_velocity_image_2(num, output1, output2, output3, target, test_result_dir):
    plt.figure(figsize=(8, 6))
    # int
    column_index = 150
    pixel_values1, pixel_values2, pixel_values3, pixel_values4 = [], [], [], []
    for y in range(output1.shape[0]):
        pixel_value1 = target[y, column_index]
        pixel_value2 = output1[y, column_index]
        pixel_value3 = output2[y, column_index]
        pixel_value4 = output3[y, column_index]
        pixel_values1.append(pixel_value1)
        pixel_values2.append(pixel_value2)
        pixel_values3.append(pixel_value3)
        pixel_values4.append(pixel_value4)
    plt.plot(pixel_values1, color='red', linewidth=2, label='Ground Truth')
    plt.plot(pixel_values2, color='blue', linewidth=2, label='DC-FWI')
    plt.plot(pixel_values3, color='orange', linewidth=2, label='w/o dense connection')
    plt.plot(pixel_values4, color='yellow', linewidth=2, label='w/o CBAM')
    plt.legend(fontsize=12)

    plt.xticks(range(0, 301, 50), [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Depth (km)', fontsize=12)
    plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.savefig(test_result_dir + 'PDD' + str(num) + '_' + str(column_index) + '.png')

if __name__ == '__main__':
    # np.load('D:/Wang-Linrong/TFDSUNet-main /data/CurveVelA/train_data/seismic/seismic1.npy')
    # D:/Zhang-Xingyi/Data for FWI/SimulateData/train_data/georec_train/georec1.mat
    # D:/Zhang-Xingyi/DD-Net release/data/SEGSimulation/train_data/seismic/seismic1.mat

    # seismic = 'E:/Data/OpenFWI/CurveVelB/train_data/seismic/seismic1.npy'
    # pain_openfwi_seismic_data(np.load(seismic)[1, 2, :]) 
    seismic = 'E:/Data/seg/SEGSimulation/train_data/seismic/seismic1.mat'

    pain_seg_seismic_data(scipy.io.loadmat(seismic)["Rec"][:,:,15])
