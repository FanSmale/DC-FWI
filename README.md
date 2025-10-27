# DC-FWI
The source code of paper "Enhancing structural fidelity in seismic inversion via dense connection and CBAM".

Deep learning full waveform inversion (DL-FWI) demonstrates significant potential by learning the complex mapping between seismic data and velocity models.
However, existing DL-FWI methods often suffer from insufficient structural fidelity, manifested as inconsistent velocity predictions in homogeneous regions and blurring at stratigraphic interfaces and faults. 
In this paper, we propose a novel end-to-end architecture (DC-FWI) to enhance structural fidelity in seismic inversion through the synergistic integration of dense connection, convolutional block attention module (CBAM), and a joint loss function based on uncertainty weighting. 

Train code files: train.py(main running program)

The primary procedure for training our method. The models and loss information generated during training will be stored in the train_result folder.

Test code files: test.py (main running program)

The main program for testing the model. The evaluation metric results generated during the test will be stored in the test_result folder.

