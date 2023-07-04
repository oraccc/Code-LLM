# Notes about loading, inferring and fine-tuning LLM



## *class* AutoModelForCausalLM

### *classmethod* `from_pretrained`

#### *parameters* device_map

* 在from_pretrained的时候便可以将模型加载到指定的GPU或CPU中

* 需要提前安装accelerate包（但无需在代码中import）

* > You can let 🤗 Accelerate handle the device map computation by setting `device_map` to one of the supported options (`"auto"`, `"balanced"`, `"balanced_low_0"`, `"sequential"`) or create one yourself, if you want more control over where each layer should go.

  可选的几种参数为"auto", "balanced", "balanced_low_0", "sequential", 或者自定义一个 *dict* 也可

  * `auto` 与 `balanced` 类似，会将模型平均划分到所有可用的GPU上，这样可以将batch_size设大一些，注意如果GPU不够，auto会同时利用CPU存储一部分模型参数

    > The options `"auto"` and `"balanced"` produce the same results for now, but the behavior of `"auto"` might change in the future if we find a strategy that makes more sense, while `"balanced"` will stay stable.

  * `balanced_low_0`：会在除了第一张卡（"cuda:0"）之外的所有GPU上平均分配模型参数，这样可以在第一张卡上做一些额外的操作，例如当使用generate函数时存放输出数据

  * `sequential`：先尝试用第一张卡，当第一张卡用完时可以再用第二张卡，以此类推



