# HPML-Final-Project
Optimizing Inference of Convolutional Neural Networks Using Apache TVM

Apache TVM is used to optimize the inference time of 3 ResNet models, namely ResNet18, ResNet50 and ResNet101.

Apacha TVM optimizes the inference time based on the target.
Target is intended hardware(processor) model is going to run on.

Two targets specified  
1) Llvm - For processor (intel core i5)
2) llvm -mcpu=core-avx2 -  Intel core i5 with AVX2

Results
|Model | Optimized by TVM | Unoptimized |
|--- | --- | --- |
|ResNet18 llvm | 104.03ms | 104.60ms|
| ||
|ResNet18 llvm-avx2 | 33.46ms | 33.29ms|
| ||
|ResNet50 llvm | 241.19ms | 240.45ms |
| ||
|ResNet50 llvm-avx2 | 88.26ms | 85.16ms |
| ||
|ResNet101 llvm | 446.21mss | 446.41ms |
| ||
|ResNet101 llvm-avx2 | 155.9ms | 156ms |


The Json files has the TVM results, how it's optimizing

Inorder to run the code create a new conda environment and install the apache TVM on it. 
https://tvm.apache.org/docs/tutorial/install.html use this document to install TVM.
