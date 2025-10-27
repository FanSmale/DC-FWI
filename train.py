################################################
########            导入库               ########
################################################
import time
import datetime
from model.DC_Net70_2_v3 import *
# from model.InversionNet import *
from func.data import *
# from func.utils import *
from func.loss import *
from torch.utils.tensorboard import SummaryWriter
import math
from pytorch_msssim import SSIM
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

################################################
########             NETWORK            ########
#################################### ############

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
# print(cuda_available)
device = torch.device("cuda" if cuda_available else "cpu")
print(device)

net = DenseNet()

net = net.to(device)


# Optimizer we want to use
optimizer = torch.optim.Adam(net.parameters(), lr=LearnRate)

# If ReUse, it will load saved model from premodelfilepath and continue to train
if ReUse:
    print('***************** Loading pre-training model *****************')
    print('')
    premodel_file = train_result_dir + PreModelname
    net.load_state_dict(torch.load(premodel_file))
    net = net.to(device)
    print('Finish downloading:', str(premodel_file))

################################################
########    LOADING TRAINING DATA       ########
################################################
print('***************** Loading training dataset *****************')

dataset_dir = Data_path
# OpenFWI
trainSet = DatasetOpenFWI(dataset_dir, TrainSize, 1, "train")
train_loader = DataLoader(trainSet, batch_size=BatchSize, shuffle=True)

valSet = DatasetOpenFWI(dataset_dir, ValSize, 1, "test")
val_loader = DataLoader(valSet, batch_size=BatchSize, shuffle=True)


#SEG
# trainSet = DatasetSEG(dataset_dir, TrainSize, 1, "train")
# train_loader = DataLoader(trainSet, batch_size=BatchSize, shuffle=True)
#
# valSet = DatasetSEG(dataset_dir, ValSize, 1601, "test")
# val_loader = DataLoader(valSet, batch_size=BatchSize, shuffle=True)


################################################
########      DYNAMIC LOSS MODULE       ########
################################################
class DynamicMultiLoss(nn.Module):
    def __init__(self, num_losses=3):
        super(DynamicMultiLoss, self).__init__()
        self.num_losses = num_losses
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def get_weights(self):
        with torch.no_grad():
            weights = [torch.exp(-log_var).item() for log_var in self.log_vars]
            weight_sum = sum(weights)
            normalized_weights = [w / weight_sum for w in weights]
            return normalized_weights

    def forward(self, losses):
        assert len(losses) == self.num_losses, "损失项数量不匹配"

        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            reg_term = torch.nn.functional.softplus(self.log_vars[i])
            total_loss += 0.5 * (precision * loss + reg_term)

        return total_loss


class CombinedLoss(nn.Module):
    def __init__(self, dynamic_weighting=True, fixed_weights=None):
        super(CombinedLoss, self).__init__()
        self.dynamic_weighting = dynamic_weighting

        if dynamic_weighting:
            self.weight_module = DynamicMultiLoss(3)
        else:
            if fixed_weights is None:
                fixed_weights = [0.3, 0.7, 0.5]  # 默认值
            self.weights = fixed_weights

    def forward(self, v_pred, v_true):
        l1_loss = F.l1_loss(v_pred, v_true)

        tensor_cuda = torch.abs(v_pred- v_true)  # {Tensor:(,1,70,70)}
        logcosh_loss = torch.mean(torch.log(torch.cosh(tensor_cuda + 1e-8)))

        ssim_module = SSIM(data_range=1.0, size_average=True, channel=1)
        ssim_loss = 1 - ssim_module(v_pred, v_true)

        if self.dynamic_weighting:
            losses = [l1_loss, logcosh_loss, ssim_loss]
            total_loss = self.weight_module(losses)
            current_weights = self.weight_module.get_weights()
            return total_loss, l1_loss, logcosh_loss, ssim_loss, current_weights
        else:
            total_loss = self.weights[0] * l1_loss + self.weights[1] * logcosh_loss + self.weights[2] * ssim_loss
            return total_loss, l1_loss, logcosh_loss, ssim_loss, self.weights


criterion = CombinedLoss(dynamic_weighting=True)

if criterion.dynamic_weighting:
    optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': LearnRate},
        {'params': criterion.weight_module.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4}
    ])

################################################
########            TRAINING            ########
################################################

writer = SummaryWriter(log_dir="tensorboard_logs/fva/DC_Net70_2_v3_sigmoid_withoutCBAM_newloss_dynamic_loss")

print()
print('*******************************************')
print('*******************************************')
print('                Training ...               ')
print('*******************************************')
print('*******************************************')
print()

print('原始地震数据尺寸:%s' % str(DataDim))
print('原始速度模型尺寸:%s' % str(ModelDim))
print('训练规模:%d' % int(TrainSize))
print('训练批次大小:%d' % int(BatchSize))
print('迭代轮数:%d' % int(Epochs))
print('学习率:%.5f' % float(LearnRate))
print('超参数:%.1f' % float(loss_weight))


step = int(TrainSize / BatchSize)
start = time.time()


def train():
    total_loss = 0
    total_loss_l1 = 0
    total_loss_logcosh = 0
    total_loss_ssim = 0

    for i, (seismic_datas, vmodels) in enumerate(train_loader):
        # torch.cuda.empty_cache()

        net.train()

        seismic_datas = seismic_datas[0].to(device)
        # print(seismic_datas[0].shape) # [5,1000,70]
        vmodels = vmodels[0].to(device)

        optimizer.zero_grad()

        if NoiseFlag:
            noise_mean = 0
            noise_std = 0.1
            noise = torch.normal(mean=noise_mean, std=noise_std, size=seismic_datas.shape).to(device)
            seismic_datas = seismic_datas + noise

        outputs = net(seismic_datas)

        outputs = outputs.to(torch.float32)
        vmodels = vmodels.to(torch.float32)

        if torch.isnan(outputs).any():
            print('!!! NaN detected in outputs !!!')
            outputs = torch.nan_to_num(outputs, nan=0.0)

        loss, loss_l1, loss_logcosh, loss_ssim, current_weights = criterion(outputs, vmodels)

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        total_loss += loss.item()
        total_loss_l1 += loss_l1.item()
        total_loss_logcosh += loss_logcosh.item()
        total_loss_ssim += loss_ssim.item()

        loss = loss.to(torch.float32)
        loss.backward()

        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    avg_loss_l1 = total_loss_l1 / len(train_loader)
    avg_loss_logcosh = total_loss_logcosh / len(train_loader)
    avg_loss_ssim = total_loss_ssim / len(train_loader)

    _, _, _, _, current_weights = criterion(outputs, vmodels)
    return avg_loss, avg_loss_l1, avg_loss_logcosh, avg_loss_ssim, current_weights
    # return avg_loss


def validate():
    total_loss = 0
    net.eval()

    with torch.no_grad():
        for i, (seismic_datas, vmodels) in enumerate(val_loader):

            seismic_datas = seismic_datas[0].to(device)
            vmodels = vmodels[0].to(device)

            outputs = net(seismic_datas)

            outputs = outputs.to(torch.float32)
            vmodels = vmodels.to(torch.float32)
            loss, loss_l1, loss_logcosh, loss_ssim, _ = criterion(outputs, vmodels)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


train_loss_list = 0
val_loss_list = 0
train_loss_l1_list = 0
train_loss_logcosh_list = 0
train_loss_ssim_list = 0
weight_history = []

for epoch in range(Epochs):
    epoch_loss = 0.0
    since = time.time()

    # Start recording memory snapshot history
    # torch.cuda.memory._record_memory_history(True)

    train_loss, train_loss_l1, train_loss_logcosh, train_loss_ssim, current_weights= train()
    # train_loss = train()
    val_loss = validate()

    weight_history.append(current_weights)

    writer.add_scalar('Loss/Train_Total', train_loss, epoch)
    writer.add_scalar('Loss/Train_L1', train_loss_l1, epoch)
    writer.add_scalar('Loss/Train_Logcosh', train_loss_logcosh, epoch)
    writer.add_scalar('Loss/Train_SSIM', train_loss_ssim, epoch)
    writer.add_scalar('Loss/Val_Total', val_loss, epoch)
    writer.add_scalars('Weights/L1', {'train': current_weights[0]}, epoch)
    writer.add_scalars('Weights/Logcosh', {'train': current_weights[1]}, epoch)
    writer.add_scalars('Weights/SSIM', {'train': current_weights[2]}, epoch)

    # Show train and val loss every 1 epoch
    if (epoch % 1) == 0:
        print(f"Epoch: {epoch + 1}, Train loss:{train_loss:.4f}, Val loss: {val_loss: .4f}, l1: {train_loss_l1: .6f}, logcosh: {train_loss_logcosh: .6f}, ssim: {train_loss_ssim: .6f}")
        print(f"  Weights: L1={current_weights[0]:.3f}, Logcosh={current_weights[1]:.3f}, SSIM={current_weights[2]:.3f}")
        time_elapsed = time.time() - since
        print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    # Save net parameters every 10 epochs
    if (epoch + 1) % SaveEpoch == 0:
        torch.save(net.state_dict(), train_result_dir + ModelName + '_epoch' + str(epoch +1) + '.pkl')
        print('Trained model saved: %d percent completed' % int((epoch + 1) * 100 / Epochs))

    if epoch % 10 == 0:
        with torch.no_grad():
            log_vars = [var.item() for var in criterion.weight_module.log_vars]
            precisions = [torch.exp(-var).item() for var in criterion.weight_module.log_vars]
            print(f"Log vars: L1: {log_vars[0]:.3f}, LogCosh: {log_vars[1]:.3f}, SSIM: {log_vars[2]:.3f}")
            print(f"Precisions: L1: {precisions[0]:.3f}, LogCosh: {precisions[1]:.3f}, SSIM: {precisions[2]:.3f}")

    train_loss_list = np.append(train_loss_list, train_loss)
    val_loss_list = np.append(val_loss_list, val_loss)
    train_loss_l1_list = np.append(train_loss_l1_list, train_loss_l1)
    train_loss_logcosh_list = np.append(train_loss_logcosh_list, train_loss_logcosh)
    train_loss_ssim_list = np.append(train_loss_ssim_list, train_loss_ssim)

writer.close()

# Record the consuming time
time_elapsed = time.time() - start
print('Training complete in {:.0f}m  {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

np.save(train_result_dir + 'weight_history.npy', np.array(weight_history))

# Save the loss
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }
font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 21,
         }

SaveTrainValidResults_2(train_loss=train_loss_list, val_loss=val_loss_list, l1=train_loss_l1_list, logcosh=train_loss_logcosh_list, ssim=train_loss_ssim_list, SavePath=train_result_dir, ModelName=ModelName, font2=font2, font3=font3)







