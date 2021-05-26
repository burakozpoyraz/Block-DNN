# Block-DNN Based GSM Detector
![Issues Badge](https://img.shields.io/github/issues/burakozpoyraz/Block-DNN)
![Forks Badge](https://img.shields.io/github/forks/burakozpoyraz/Block-DNN)
![Stars Badge](https://img.shields.io/github/stars/burakozpoyraz/Block-DNN)

This project is the implementation of the paper "[Block Deep Neural Network-Based Signal Detector for Generalized Spatial Modulation](https://ieeexplore.ieee.org/document/9165095 "Block Deep Neural Network-Based Signal Detector for Generalized Spatial Modulation")" published by Hasan Albinsaid, Keshav Singh, Sudip Biswas, Chih-Peng Li, and Mohamed-Slim Alouini. The authors proposed a detector based on block deep neaural network (B-DNN) for the generalised spatial modulation (GSM) systems. They compared the proposed B-DNN detector with the conventional detectors, which are maximum likelihood (ML), block zero-forcing (B-ZF), and block minimum mean squared error (B_MMSE). The computer simulations revealed that the proposed B-DNN detector outperforms the conventional block linear detectors, e.g., B-ZF and B-MMSE, and performs similar to the theoretically best detector, ML. However, B-DNN requires less computation time with respect to the conventional detectors.

The original implementation of the paper can be found [here](https://github.com/hasanabs/B_DNN "here").

## Table of Contents
- [Framework](#framework)
  * [Training](#training)
    + [Data Preprocessing](#data-preprocessing)
    + [Hidden Nodes](#hidden-nodes)
    + [Deep Learning Model](#deep-learning-model)
  * [Simulation](#simulation)
- [Implementation](#implementation)
  * [Training](#training-1)
  * [Simulation](#simulation-1)

## Framework
This section provides an overview of the training and simulation algorithms.
### Training
#### *Data Preprocessing*
Before training the model, input and output data are preprocessed in order to obtain the desired input-output pairs. Assuming that the training process takes N<sub>s</sub> iterations, the input data is created as a numpy array with the shape of (N<sub>s</sub>, n<sub>x</sub>), where n<sub>x</sub> is the number of input nodes that changes according to the feature vector generator (FVG) type. The input data is composed of the received signal vector and the channel matrix from transmitter to the receiver, where the received signal model for the training is given as

```math
y = Hx
```

where ```y``` is the (N<sub>r</sub>, 1) received signal vector, ```H``` is the (N<sub>r</sub>, N<sub>t</sub>) channel matrix that is composed of complex Gaussian random variable elements with zero-mean and unit-variance, and ```x``` is the (N<sub>t</sub>, 1) transmitted vector. It should be noted that N<sub>p</sub> out of N<sub>t</sub> symbols are different than zero and drawn from a *M*-ary constellation with the symbols having unit-average power, where these N<sub>p</sub> indices indicate the active transmit antennas, and the rest is zero. It can be seen that, the complex Gaussian noise is omitted for the training.

On the other hand, the output data might seem a little bit more complicated than the input data. It is a list composed of N<sub>p</sub> numpy arrays each of which having the same shape of (N<sub>s</sub>, M). Each numpy array corresponds to the labels of each active transmit antenna. Since the number of columns of each numpy array is M, single row of each numpy array is a one hot vector of the transmitted symbol from respective active transmit antenna and at respective iteration.

#### *Hidden Nodes*
The numbers of hidden nodes for each hidden layer are defined in `HiddenLayerNodes` function which can be used simply as follows:
```python
M = 2 # BPSK
n_h_list = HiddenLayerNodes(M)
```
It should be noted that the nodes are defined as given in the paper; however, they can be changed as required.

#### *Deep Learning Model*
The model is created in accordance with the specifications given in the paper, and can be obtained by using the `Model` function.
```python
M = 2 # BPSK
Np = 2
Nr = 2
FVG_type = "SFVG"
if FVG_type == "SFVG":
    n_x = 2 * Nr + 2 * Nr * Np
elif FVG_type == "JFVG":
    n_x = 1 + Np * Np
elif FVG_type == "CFVG":
    n_x = Nr + Nr * Np
n_y = M
n_h_list = HiddenLayerNodes(M)
B_DNN_model = Model(n_x, n_y, n_h_list, Np)
```
Since the type of the FVG alters the way of creating the inputs of the model, number of input nodes for each FVG type is different.

### Simulation
The simulation algorithm for each detector with the provided parameters is created as the function called `Simulation`, and can be implemented for each detector as given below:
```python
parameter_dict = {"SNRdB_array" : np.arange(0, 25, 4), "Ns" : 100000, "Nt" : 4, "Np" : 2, "Nr" : 2, "M" : 4, "mod_type" : "PSK"}
B_DNN_output_dict = Simulation(parameter_dict, "B-DNN", "SFVG")
ML_output_dict = Simulation(parameter_dict, "ML")
B_ZF_output_dict = Simulation(parameter_dict, "B-ZF")
B_MMSE_output_dict = Simulation(parameter_dict, "B-MMSE")
```
Output of the function is a dictionary including the signal-to-noise ratio (SNR) vector, bit error rate (BER) vector that is composed of the BER values for each SNR value, and the error vector that is composed of the number of bit errors for each SNR value. The `Simulation` function utilizes the functions created for each detector. For a set of parameters, a received signal vector, and a channel matrix, the proposed B-DNN detector is created with the following algorithm:
```python
detected_bit_array = B_DNN_Detector(model, FVG_type, TAC_set, ss, ns, m, y, H)
```
Similar to the B-DNN detector, the conventional detectors are given as follows:
```python
detected_bit_array = ML_Detector(TAC_set, ss, ns, m, y, H)
detected_bit_array = B_ZF_Detector(TAC_set, ss, ns, m, y, H)
detected_bit_array = B_MMSE_Detector(TAC_set, ss, ns, m, y, H, N0)
```

## Implementation
In order to train the proposed model for the B-DNN detector and perform the simulations, two steps are required at the beginning:
1. Download the ZIP file and extract the *Block-DNN* folder.
2. Obtain the path of the folder to use in the notebooks for saving and loading the files.

### Training
Training of the proposed B-DNN detector can be implemented with *B-DNN_Training* notebook. Some of the parameters, such as number of the hidden nodes, number of the input nodes, etc., change according to the desired figure in the paper. This notebook is designed to train the models for the following figures in the paper:
1. Figure 3a (Name = 3a)
2. Figure 3b (Name = 3b)
3. Figure 3c (Name = 3c)
4. Figure 4a (Names = 4a-S, 4a-J, 4a-C)
5. Figure 4b (Names = 4b-2, 4b-4, 4b-16)

There are two steps to train the model with the parameters of one the figures given above:
1. Assign the name of the desired figure to the variable `fig_name` in the ***GSM parameters*** section. It should be noted that there are 3 models to be trained for both Figure 4a and Figure 4b. Thus, each of the model with the names given above should be trained in order to simulate the figures.
2. Assign the folder path to the variable `B_DNN_folder_path` in the ***Training The Model*** section so as to save the model.

Once the steps are completed, the training can be performed by running all the cells. If any custom model with different parameters are desired, then these specific parameters can be arranged in the ***GSM Parameters*** section by adding the new parameters to the else statement of the if / else structure.

The models with the following parameters are already trained and published in *Trained Models* folder:

| N<sub>p</sub>|N<sub>r</sub>|Modulation|FVG Type|Figures|
|:--------------------:|:-------------------:|:-------------:|:------------:|:--------:|
|2|2|BPSK|CFVG|4a|
|2|2|BPSK|JFVG|4a|
|2|2|BPSK|SFVG|4a, 4b|
|2|2|QPSK|SFVG|3a, 4b|
|2|2|16-QAM|SFVG|4b|
|2|4|QPSK|SFVG|3b|

### Simulation
Simulations of the figures can be implemented using *Simulations* notebook. There are two steps to perform simulations:
1. Assign the folder path to the variable `B_DNN_folder_path` in the ***Libraries*** section so as to load the existing models and save the simulation results.
2. Assign the name of the desired figure to the variable `fig_name` in the ***GSM Parameters*** in order to create the dictionary composed of the parameters related to the desired figure.

Once the steps are completed, the simulation can be performed by running all the cells until the ***Plot*** section. After simulation, the results will automatically be saved to the corresponding folder under the *Results* folder. The figure can be visualised by running the ***Plot*** cell. If any custom figure with different parameters are desired, then these specific parameters can be arranged in the ***GSM Parameters*** section by adding custom dictionary to the else statement of the if / else structure. It should be noted that the model that is required for the custom figure should be trained before the simulation. The simulation results of the following figures are already obtained and saved in the ***Results*** folder:
- Figure 3a
- Figure 4a
- Figure 4b

## Citations
- [1] H. Albinsaid, K. Singh, S. Biswas, C. -P. Li and M. -S. Alouini, "Block Deep Neural Network-Based Signal Detector for Generalized Spatial Modulation," in IEEE Communications Letters, vol. 24, no. 12, pp. 2775-2779, Dec. 2020, doi: [10.1109/LCOMM.2020.3015810](https://ieeexplore.ieee.org/document/9165095 "10.1109/LCOMM.2020.3015810").
- arXiv [version](https://arxiv.org/abs/2008.03612 "version")

## Licence
![MIT Licence](https://img.shields.io/github/license/burakozpoyraz/Block-DNN)
