# HPML-Final-Project
Optimizing Inference of Convolutional Neural Networks Using Apache TVM

Apache TVM is used to optimize the inference time of 3 ResNet models, namely ResNet18, ResNet50 and ResNet101.

Apacha TVM optimizes the inference time based on the target.
Target is intended hardware(processor) model is going to run on.

Two targets specified  
1) Llvm - For processor (intel core i5)
2) llvm -mcpu=core-avx2 -  Intel core i5 with AVX2

Results
Attempt | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Seconds | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269

Inorder to run the code create a new conda environment and install the apache TVM on it. 
https://tvm.apache.org/docs/tutorial/install.html use this document to install TVM.
