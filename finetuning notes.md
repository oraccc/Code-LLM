# Notes about loading, inferring and fine-tuning LLM



## Huggingface 🤗 



### *classmethod* `from_pretrained`

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

#### *parameters* load_in_8bit[(:link:)](https://huggingface.co/docs/transformers/main/main_classes/quantization)

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

---



### *class* `TrainingAruguments`

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

## DeepSpeed :rocket:



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

