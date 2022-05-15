import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
import os 
import numpy as np
import tvm
from tvm import te
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay import testing
import tvm.testing
from tvm.contrib.download import download_testdata
import timeit
import onnx
from PIL import Image

from scipy.special import softmax

import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

#onnx_models = ["resnet18-v2-7","resnet50-v2-7","resnet101-v2-7"]
#targets = ["llvm","llvm -mcpu=core-avx2"] 

onnx_models = ["resnet18-v2-7"]
targets = ["llvm -mcpu=core-avx2"]

def download_load_model(model):
    model_url = "".join(["https://github.com/onnx/models/raw/main/"
    "vision/classification/resnet/model/",
    (model)+".onnx"
    ])
    model_path = download_testdata(model_url, str(model)+".onnx", module="onnx")
    onnx_model = onnx.load(model_path)
    return onnx_model

def download_img_data():
    img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    img_path = download_testdata(img_url, "imagenet_cat.png", module="data")
    resized_image = Image.open(img_path).resize((224, 224))
    img_data = np.asarray(resized_image).astype("float32")
    img_data = np.transpose(img_data, (2, 0, 1))
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev
    img_data = np.expand_dims(norm_img_data, axis=0)
    return img_data
#def preprocess_img():
        
# script_dir = os.path.dirname(__file__)
# rel_path = "resnet101-v2-7.onnx"
# model_path = os.path.join(script_dir, rel_path)

target = "llvm" 
target2 = "llvm -mcpu=core-avx2"

img_data = download_img_data() 
input_name = "data"
shape_dict = {input_name: img_data.shape}
number = 10
repeat = 1
min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
timeout = 10  # in seconds

# create a TVM runner
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)
for model in onnx_models:
    onnx_model = download_load_model(model)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    
    for j,t in enumerate(targets):
        print("Optimizing model: "+model+" Target: "+t)   
        f_name = ""
        #if j==0:f_name = model+".json"
        #else:f_name = model+"_avx2.json"
        f_name = model+"_avx2.json"
        tuning_option = {
        "tuner": "xgb",
        "trials": 1500,
        "early_stopping": 100,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="default"), runner=runner
        ),
        "tuning_records": f_name,
        }

        # begin by extracting the tasks from the onnx model
        tasks = autotvm.task.extract_from_program(mod["main"], target=t, params=params)

        # Tune the extracted tasks sequentially.
        for i, task in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            tuner_obj = XGBTuner(task, loss_type="rank")
            tuner_obj.tune(
                n_trial=min(tuning_option["trials"], len(task.config_space)),
                early_stopping=tuning_option["early_stopping"],
                measure_option=tuning_option["measure_option"],
                callbacks=[
                    autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                    autotvm.callback.log_to_file(tuning_option["tuning_records"]),
                ],
            )
        # with autotvm.apply_history_best(tuning_option["tuning_records"]):
            # with tvm.transform.PassContext(opt_level=3, config={}):
                # lib = relay.build(mod, target=t, params=params)
        # dev = tvm.device(str(t), 0)
        # module = graph_executor.GraphModule(lib["default"](dev))        
        # dtype = "float32"
        # module.set_input(input_name, img_data)
        # module.run()
        # output_shape = (1, 1000)
        # tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
        # labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
        # labels_path = download_testdata(labels_url, "synset.txt", module="data")
        # scores = softmax(tvm_output)
        # scores = np.squeeze(scores)
        # ranks = np.argsort(scores)[::-1]
        # for rank in ranks[0:5]:
            # print("class='%s' with probability=%f" % (labels[rank], scores[rank]))
        # timing_number = 10
        # timing_repeat = 10
        # optimized = (
            # np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
            # * 1000
            # / timing_number
        # )
        # optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}
        # print("optimized: %s" % (optimized))
        # print("unoptimized: %s" % (unoptimized))    
            