# Notes about loading, inferring and fine-tuning LLM



## Huggingface 🤗 



### *classmethod* `from_pretrained`

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

​		目前device_map只能做到最为基础的model parallelism (naive MP)，没有pipeline， 因此每个时刻只有一张卡在运行，效率很低

​		<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-gpipe-bubble.png"  />

---



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



## DeepSpeed :rocket:



## 杂项 :wrench:



### 训练时GPU显存

* 当一个模型被加载到GPU中时，GPU的内核(kernel)也会被加载，因此即使我们放了一个很小的tensor到GPU中，也会看到GPU内存已经被占据了1~2GB，这就是kernel的大小；在实际显存计算中也需要考虑这一部分的影响。
  * V100的kernel大小大致为1300MB

* 在训练的时候，我们会注意到GPU的显存占用远比单纯加载model时要大得多。在训练时，有如下这些部分占了GPU显存
