# Notes about loading, inferring and fine-tuning LLM



## Huggingface 🤗 



### *classmethod*   `from_pretrained`

```python
# sample1: load model in 8bit
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    load_in_8bit=True, 
    device_map='auto'
)

# sample2: load model in fp16
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    use_cache=not no_gradient_checkpointing,
    torch_dtype=torch.float16,
)
```

#### *parameters* pretrained_model_name_or_path 

* 需要加载的模型名称或者加载模型的本地路径

#### *parameters* cache_dir

* 指定模型下载的位置
* :exclamation: 若使用Azure ML Studio提供的机器，记得第一次下载大模型时一定要将路径指明到`”/mnt/batch/tasks/shared/LS_root/mounts/clusters/xxxxx"`下，该路径下有充足的存储空间。默认的路径下存储空间只有60G，不足以存储大模型。
* 将模型下载到本地之后，可以直接将`pretrained_model_name_or_path`改为本地存储的路径，不必每次都指定`cache_dir`
* 另一种方式：参考[官方文档](https://huggingface.co/docs/transformers/installation?highlight=transformers_cache#cache-setup)直接在`bash.rc`里更改环境变量`HUGGINGFACE_HUB_CACHE`

#### *parameters* device_map [(:link:)](https://huggingface.co/docs/accelerate/usage_guides/big_modeling#loading-weights)

* 在from_pretrained的时候便可以将模型加载到指定的GPU或CPU中， 若不指定，则默认加载到CPU中

* 需要提前安装`accelerate`包（但无需在代码中import）

* 可选的几种参数为"auto", "balanced", "balanced_low_0", "sequential", 或者自定义一个 *dict* 也可

  * `auto` 与 `balanced` 类似，会将**模型平均划分**到所有可用的GPU上，这样可以将batch_size设大一些，注意如果GPU不够，auto会同时利用CPU存储一部分模型参数

    > The options `"auto"` and `"balanced"` produce the same results for now, but the behavior of `"auto"` might change in the future if we find a strategy that makes more sense, while `"balanced"` will stay stable.

  * `balanced_low_0`：会在除了第一张卡（"cuda:0"）之外的所有GPU上平均分配模型参数，这样可以在第一张卡上做一些额外的操作，例如当使用generate函数时存放输出数据

  * `sequential`：先尝试用第一张卡，当第一张卡用完时可以再用第二张卡，以此类推

  * 一个自定义的例子 `device_map={'':Accelerator().process_index}`，将模型全部放入当前process所在的GPU中（process0对应的就是GPU0）

* 查看当前model 的 device_map：`model.hf_device_map`

* 当参数 load_in_8bit 为真时，必须指定device_map

  > load the map into mixed-8bit quantized model not compiled for **CPU**!

* :exclamation:**LIMITATIONS**: 

  > The model parallelism used when your model is split on several GPUs is naive and not optimized, meaning that only one GPU works at a given time and the other sits idle.
  >
  * 目前device_map只能做到最为基础的model parallelism (naive MP)，没有pipeline， 因此每个时刻只有一张卡在运行，效率很低
   <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-gpipe-bubble.png"  />

#### *parameters* load_in_8bit [(:link:)](https://huggingface.co/docs/transformers/main/main_classes/quantization)

* 以8bit精度加载模型

* 需要提前安装`bitsandbytes`包（但无需在代码中import）

* 查看当前model所占的存储空间：`model.get_memory_footprint()`

  * 返回bytes，常用语句

    ```python
    print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")
    ```

  * 可以看到以8bit精度加载的模型所占的内存非常少，starcoder只占了15939.61MB

* 必须指定device_map

* 需要与*classmethod* `prepare_model_for_int8_training` 搭配使用（该函数需要安装peft包），该函数有以下一些功能

  > - casts all the non `int8` modules to full precision (`fp32`) for stability
  > - adds a forward hook to the input embedding layer to calculate the gradients of the input hidden states
  > - enables gradient checkpointing for more memory-efficient training

  load_in_8bit经常与lora搭配使用，lora会将原始model的参数固定，只使用一个低秩矩阵来更新参数，因此在传递梯度时只需要**input embedding** 去开启梯度传播，不需要整个模型去传播梯度（这也是这个function的第二点功能）

#### *parameters* torch_dtype

* 当设置`torch_dtype = torch.float16`时，模型以fp16精度加载，否则默认以fp32精度加载

* :exclamation: 当模型以fp16精度加载的时候，**不要**与`prepare_model_for_int8_training`搭配使用，该函数会将所有非int8精度的数值强制转换成fp32精度，这样就与目的背道而驰了

* 当以fp16精度加载模型并使用LoRA训练模型时，可能会出现以下Runtime Error

  ```bash
  Runtime Error: element 0 of tensors does not require grad and does not have a grad_fn
  ```

  这提示我们需要**在input embeddings开启梯度传播**（这其实是`prepare_model_for_int8_training`函数的第二点功能，但由于我们不能使用这个函数，因此需要手动开启），还问题具体的解决方法是调用方法`model.enable_input_require_grads()`

  > Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping the model weights fixed.

* 因此，当以fp16精度加载模型并使用LoRA来训练时，整体的流程如下

  ```python
  model.from_pretrained(path, torch_dtype=torch.float16)
  model.enable_input_require_grads()
  model.get_peft_model(model, lora_config)
  ```

  

---



### *class*  `TrainingAruguments`

```python
# sample: aruguments for fine-tuning starcoder
training_args = TrainingArguments(
    max_steps=arg.max_steps
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_steps=args.num_warmup_steps,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=not args.no_gradient_checkpointing,
    fp16=True,
    bf16=False,
    weight_decay=args.weight_decay,
    run_name="starcoder-finetuned",
    report_to="wandb",
    deepspeed=args.deepspeed,
)
```

#### *parameters* max_steps

* 总的训练步数，一旦指明之后num_train_epoch便失效了(override)
* 每次step用一个batch_size的样本训练

#### *parameters* gradient_checkpointing

* gradient checkpointing 开启后，在反向传播时会重新计算网络中间激活值

* 开启此功能可以节省计算内存，但与此同时反向传播的速度会更慢（以时间换空间）

#### *parameters* fp16/bf16

* 是否使用fp16或者bf16混合精度运算，注意V100**不支持**bf16精度

* 开启fp16精度计算之后，在多卡训练时可能会出现Runtime Error

  ```bash
  RuntimeError: expected scalar type Half but found Float
  ```

  * 解决办法参考此[:link:](https://stackoverflow.com/questions/75918140/getting-runtimeerror-expected-scalar-type-half-but-found-float-in-aws-p3-instan)：在原始代码的基础上做修改，添加torch.autocast()后错误消失，推测该错误可能与V100上的混合精度运算有关

  * ```python
    with torch.autocast("cuda"): 
        trainer.train()
    ```

  * 注：原博主说在添加这串代码之后可能会出现loss的巨大浮动或者loss为0的情况，但目前的运行结果并没有出现这样的问题

#### *parameters* deepspeed

* 可以是一个deepspeed的配置文件路径，也可以直接传入deepspeed配置 *dict*

---



## DeepSpeed :rocket:

### Trainer Deepspeed Integration [(:link:)](https://huggingface.co/docs/transformers/main/main_classes/deepspeed#deepspeed-integration)

🤗 Transformers 通过Trainer友好集成了deepspeed的核心功能，因此不需要大幅修改原先代码，只需要提供deepspeed的配置文件即可

> Integration of the core DeepSpeed features via [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer). This is an everything-done-for-you type of integration - just supply your custom config file or use our template and you have nothing else to do.

* 对于**Training**过程，deepspeed支持ZeRO stage1, 2, 3与ZeRO-Infinity

* 对于**Inferrring**过程， deepspeed支持ZeRO stage3与ZeRO-Infinity （因为stage2是对梯度做划分，因此对inference没有用）



#### 运行DeepSpeed

若要使用deepspeed，可以使用以下命令行运行

```bash
deepspeed --num_gpus=8 your_program.py <normal cl args> --deepspeed ds_config.json
```

若不指明num_gpus，则默认使用**全部的显卡**

:exclamation:不支持在jupyter notebook上运行多GPU



#### 使用ds_report自查

在实际使用deepspeed运行代码之前，可以先检查一下deepspeed的运行环境，使用命令 `ds_report` 即可查看，需要注意需要保持system installed cuda (nvcc version) 与 pytorch cuda version的匹配，否则在运行时会报错。

```bash
DeepSpeed general environment info:
torch install path ............... ['/anaconda/envs/starcoder/lib/python3.8/site-packages/torch']
torch version .................... 1.12.1
deepspeed install path ........... ['/anaconda/envs/starcoder/lib/python3.8/site-packages/deepspeed']
deepspeed info ................... 0.9.5, unknown, unknown
torch cuda version ............... 11.3
torch hip version ................ None
nvcc version ..................... 11.3
deepspeed wheel compiled w. ...... torch 1.12, cuda 11.3
```

---



### Shared Configuration [(:link:)](https://huggingface.co/docs/transformers/main/main_classes/deepspeed#shared-configuration)

在写deepspeed的configuration时，会注意到有很多参数与Trainer的TrainingAruguments有重复的部分，例如学习率，优化器参数等等，两者很容易混淆，但若是忽略掉这些参数不写，反而有时会引起程序的报错，提示deepspeed config缺少参数。因此官方较为推荐（并且实践起来没有出错）的一个做法便是使用参数`“auto”`，并将配置文件作为参数传递给TrainingAruguments，这样deepspeed可以自动读取Trainer的参数设定，或者让Trainer根据实际情况自动设置参数。

当然也可以自己设置这些value，但是必须确保与Trainer一致，不然可能会造成不可预知的错误。

> Note: currently DeepSpeed doesn’t validate parameter names, so if you misspell any, it’ll use the default setting for the parameter that got misspelled. You can watch the DeepSpeed engine start up log messages to see what values it is going to use.

目前deepspeed config不支持拼写检查，因此用户需要自己检查拼写。

一个简单的配置例子（非完整版），注意其中 `train_batch_size` 与 `train_micro_batch_size_per_gpu` 两个参数必须指定一个

```json
{
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": "auto",
        "betas": "auto",
        "eps": "auto",
        "weight_decay": "auto"
      }
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
        "warmup_min_lr": "auto",
        "warmup_max_lr": "auto",
        "warmup_num_steps": "auto"
      }
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
  }
```

### ZeRO 配置

ZeRO有关的配置是整个ds_config文件中最为重要的部分

ZeRO的实现方法是把参数占用分成三种类型。将这些类型的参数划分：

- `optimizer states`：即优化器的参数状态。例如Adam的动量参数。
- `gradients`：梯度缓存，对应于optimizer。
- `parameters`：模型参数。

DeepSpeed的ZeRO config文件也依据可以分为如下几类：

* ZeRO Stage 1: 划分optimizer states。优化器参数被划分到多个memory上，每个momoey上的进程只负责更新它自己那部分参数
* ZeRO Stage 2: 划分gradient。每个memory只保留它分配到的optimizer state所对应的梯度。
* ZeRO Stage 3: 划分模型参数。ZeRO-3会在forward和backward的时候，自动将模型参数分配到多个memory。

在实际情况中，stage2与3更为实用



#### ZeRO Stage 2 配置 [(:link:)](https://huggingface.co/docs/transformers/main/main_classes/deepspeed#zero2-config)

常用的stage 2配置是

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 1e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 1e8,
    "contiguous_gradients": true
  }
}
```

一些重要的参数说明，这些参数往往是配置文件中最重要的部分，也是在改善性能是需要反复调整的参数，而配置文件中的其余参数官方强烈推荐直接设为`"auto"`：

* **offload_optimizer**：将optimizer状态卸载到CPU或者NVMe上， 并将optimizer的计算卸载到cpu上，启用这个功能可以释放GPU的内存，允许训练更大的模型或者使用更大的batch_size

  * 可选参数有`"cpu"`,  `"nvme"`, `"none"`（不开启）
  * stage2与3均支持

* **allgather_partitions**：在每个步骤结束时从所有GPU中收集更新后的参数

* **allgather_bucket_size**：一次性收集的元素数目，降低该值可以降低所占的GPU显存，但会增加通讯开销

* **overlap_comm**：尝试将梯度减少与反向计算重叠。若设为真，可以以增加GPU RAM为代价来降低延迟

  * > `overlap_comm` uses 4.5x the `allgather_bucket_size` and `reduce_bucket_size` values. So if they are set to 5e8, this requires a 9GB footprint (`5e8 x 2Bytes x 2 x 4.5`). Therefore, if you have a GPU with 8GB or less RAM, to avoid getting OOM-errors you will need to reduce those parameters to about `2e8`, which would require 3.6GB. You will want to do the same on larger capacity GPU as well, if you’re starting to hit OOM.

  * 通过降低`allgather_bucket_size`与`reduce_bucket_size`，可以通过牺牲通讯时间来换取更多GPU显存，这样也可以设置更大的batch_size，**batch_size大小**与**训练速度**是一个需要权衡的要素

* 



## 杂项 :wrench:



### GPU显存

#### kernel的大小

当一个模型被加载到GPU中时，GPU的内核(kernel)也会被加载，因此即使我们放了一个很小的tensor到GPU中，也会看到GPU内存已经被占据了1~2GB，这就是kernel的大小；在实际显存计算中也需要考虑这一部分的影响。
* V100的kernel大小大致为1300MB

#### 训练时的显存占用[(:link:)](https://huggingface.co/docs/transformers/perf_train_gpu_one#anatomy-of-models-memory)

在训练的时候，我们会注意到GPU的显存占用远比单纯加载model时要大得多。在训练时，有如下这些部分占了GPU显存

* model weights：模型的参数，注意当以fp16精度与fp32精度的区别

  > 4 bytes * number of parameters for fp32 training
  >
  > 6 bytes * number of parameters for mixed precision training (maintains a model in fp32 and **one in fp16 in memory**)

* optimizer states 

* gradients :无论是以何种精度训练，梯度一律以fp32精度存储

* forward activations saved for gradient computation : 激活值占了训练时的显存主要部分，与batch_size成正比，存储激活值用于梯度的计算

* temporary buffers 

* functionality-specific memory

