import os
from ParamConfig import *

###################################################
####                DATA   PATHS              #####
###################################################
DataSet = 'FlatVelA/'  # CurveVelA  FlatFaultA  FlatVelA   CurveFaultA    SEGSimulation  SEGSaltData
Data_path = 'E:/Data/OpenFWI/' + DataSet  # E:/Data/OpenFWI/  /kaggle/input/dataset/data/OpenWFI/  E:/Data/


###################################################
####            RESULT   PATHS                #####
###################################################
main_dir = 'D:/PyCharm2024.3/PycharmProjects/DCFWI_v2/'

# Check the main directory
if len(main_dir) == 0:
    raise Exception('Please specify path to correct directory!')

if os.path.exists('D:/PyCharm2024.3/PycharmProjects/DCFWI_v2/train_result/' + DataSet):
    train_result_dir = main_dir + 'train_result/' + DataSet  # Replace your dataset path here
    print(True)
else:
    os.makedirs('D:/PyCharm2024.3/PycharmProjects/DCFWI_v2/train_result/' + DataSet)
    train_result_dir = main_dir + 'train_result/' + DataSet
    print(False)

if os.path.exists('D:/PyCharm2024.3/PycharmProjects/DCFWI_v2/test_result/' + DataSet):
    test_result_dir = main_dir + 'test_result/' + DataSet  # Replace your dataset path here
else:
    os.makedirs('D:/PyCharm2024.3/PycharmProjects/DCFWI_v2/test_result/' + DataSet)
    test_result_dir = main_dir + 'test_result/' + DataSet

####################################################
####                   FileName                #####
####################################################

NoiseFlag = False  # If True add noise.
modelName = 'DC_Net70_2_v3_sigmoid_withoutCBAM_newloss_dynamic_loss'

tagM1 = '_TrainSize' + str(TrainSize)
tagM2 = '_Epoch' + str(Epochs)
tagM3 = '_BatchSize' + str(BatchSize)
tagM4 = '_LR' + str(LearnRate)

ModelName = modelName + tagM1 + tagM2 + tagM3 + tagM4

TestModelName = 'DC_Net70_2_v3_sigmoid_withoutCBAM_newloss_dynamic_loss_TrainSize24000_Epoch140_BatchSize20_LR0.0001_epoch130'
# Load pre-trained model

PreModelname = TestModelName + '.pkl'
