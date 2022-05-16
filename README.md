# HPML-Final-Project -  Improving efficiency of inference computation
Optimizing Inference of Convolutional Neural Networks Using Apache TVM

Apache TVM is used to optimize the inference time of 3 ResNet models, namely ResNet18, ResNet50 and ResNet101.

Apacha TVM optimizes the inference time based on the target.
Target is intended hardware(processor) model is going to run on.

Two targets specified  
1) Llvm - For processor (intel core i5)
2) llvm -mcpu=core-avx2 -  Intel core i5 with AVX2

# Results Using Auto TVM
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

# Results Using Auto Scheduler
|Model | Optimized by TVM | Unoptimized |
|--- | --- | --- |
|ResNet18 llvm-avx2 | 28.65ms | 33.29ms|
| ||
|ResNet50 llvm-avx2 | 62.75ms | 85.16ms|

Observations
1)Models compiled with AVX2 enabled perform much better than models compiled without it.
2)Models optimized with TVM perform as good as unoptimized models. 





Inorder to run the code create a new conda environment and install the apache TVM on it. 
https://tvm.apache.org/docs/tutorial/install.html use this document to install TVM.

## Using AutoTVM for scheduling

```sh
$python autotune_resnet_multiple_models.py 
```

First run autotune_resnet_multiple_models.py to optimize the models using AutoTVM as scheduler and it will generate the json files which contain the optimization schedules 


```sh
$python compare_performance.py 
```


Then run the compare_performance.py and it will generate the time profiling results for optimized and unoptimized models

## Using Auto-scheduler for scheduling
The files autoschedule_resnet18_model.py, autoschedule_resnet50_model.py are used to optimize ResNet18 and ResNet50 models using TVM's Auto-Scheduler for scheduling and it will generate the output json files resnet-18-NHWC-B1-llvm.json and resnet-50-NHWC-B1-llvm.json. 

Then run autoschedule_evaluate_resenet50.py and autoschedule_evaluate_resenet18.py files to generate the time profiling results for optimized models using Auto-Scheduler

