[Pleae click here to view the live document](https://emmacyu.github.io/Sleep-stage-segmentation-CNN/)

### Abstract
Sleep is crucial to human health and identification of sleep stages can be used as indicators for diagnosis of many sleep disorders. The traditional sleep stage classification that relies heavily on physicians’ inspection is expensive and error prone. To mitigate the bottleneck of manual sleep data analysis, this project presents a hybrid multi-head conventional neural networks (CNN) with Long-Short Term Memory (LSTM) method to automatically classify the sleep stages based on single channel electroencephalograph (EEG) data which is in time series format. Three models are built step by step with the same parameters used for the common characteristics in each model. Here, loss and accuracy are used to evaluate the performance of the neural network. The experimental results show that multi-head CNN model can achieve a good accuracy above 73.9% even with only 3 parallel heads and 2 fully connected layers. With only one additional LSTM layer stacked on top of multi-head CNN, the performance improved drastically from 73.9% to 82.143%.

### Disclaimer
The following project is shared for educational purposes only. 
The author and its affiliated institution are not responsible in any manner whatsoever for any damages, 
including any direct, indirect, special, incidental, 
or consequential damages of any character arising as a result of the use or inability to use this software.

The current implementation is not, in any way, intended, nor able to generate code in a real-world context. 
We could not emphasize enough that this project is experimental and shared for educational purposes only. 
Both the source code and the datasets are provided to foster future research in software engineering and are not designed for end users.

### Key components

**SRC**: The src/ directory contains all the scripts and implementations of the experiments in the project. Each process is organized into separate files for clarity and modularity.

- data preparation
- models  
- run.py  

**Data**: The data/ directory includes the inputs (e.g., the EEG data) and outputs (e.g., saved models).

### Set up

#### Prerequisites
- Python 3.7 or newer   
- GPU environment


