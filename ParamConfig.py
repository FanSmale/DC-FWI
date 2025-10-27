# -*- coding: utf-8 -*-
"""
@Time: 2024/6/28

@author: Zeng Zifei
"""

####################################################
####             MAIN PARAMETERS                ####
####################################################
OutChannel = 1  # 输出速度模型的通道数

SimulateData = False# True # If False denotes training the CNN with SEGSaltData
SEG = False
OpenFWI = True
DataSet = 'FlatVelA/'  # CurveVelA  FlatFaultA  FlatVelA   CurveFaultA    SEGSimulation SEGSaltData

ReUse = False  # If False always re-train a network
LearnRate = 1e-4

if OpenFWI:
    DataDim = [1000, 70]  # Dimension of original one-shot seismic dataset
    ModelDim = [70, 70]  # Dimension of one velocity model
    InChannel = 5
    loss_weight = 0.3
else:
    DataDim = [400, 301]  # Dimension of original one-shot seismic dataset
    ModelDim = [201, 301]  # Dimension of one velocity model
    InChannel = 29
    loss_weight = 0.5

dh = 10  # Space interval

####################################################
####             NETWORK PARAMETERS             ####
####################################################

DisplayStep = 50  # Number of steps till outputting stats

if DataSet == "FlatVelA/":
    Epochs = 140
    TrainSize = 24000
    ValSize = 500
    TestSize = 6000
    TestBatchSize = 20
    BatchSize = 20
    SaveEpoch = 10
elif DataSet == "FlatFaultA/":
    Epochs = 400
    TrainSize = 48000
    ValSize = 500
    TestSize = 100
    TestBatchSize = 20
    BatchSize = 20
    SaveEpoch = 10
elif DataSet == "CurveVelA/":
    Epochs = 180
    TrainSize = 24000
    ValSize = 500
    TestSize = 6000
    TestBatchSize = 20
    BatchSize = 20
    SaveEpoch = 10
elif DataSet == "CurveFaultA/":
    Epochs = 160
    TrainSize = 48000
    ValSize = 500
    TestSize = 6000
    TestBatchSize = 20
    BatchSize = 20
    SaveEpoch = 5
elif DataSet == "CurveVelB/":
    Epochs = 180
    TrainSize = 24000
    ValSize = 500
    TestSize = 6000
    TestBatchSize = 20
    BatchSize = 20
    SaveEpoch = 10
elif DataSet == "SEGSimulation/":
    Epochs = 400
    TrainSize = 1600
    ValSize = 20
    TestSize = 100
    TestBatchSize = 10
    BatchSize = 4
    SaveEpoch = 10
elif DataSet == "SEGSaltData/":
    Epochs = 120
    TrainSize = 130
    ValSize = 5
    TestSize = 10
    TestBatchSize = 5
    BatchSize = 10
    SaveEpoch = 10