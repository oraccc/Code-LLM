### 2023/7/4

#### finetuning starcoder

| trainer parameters         | value    | notes                                      |
| -------------------------- | -------- | ------------------------------------------ |
| batch_size                 | 1        | per device train/eval batch size           |
| learning_rate              | 1e-4     |                                            |
| num_warmup                 | 100      |                                            |
| lr_scheduler_type          | "cosine" |                                            |
| gradient_accumlation_steps | 16       |                                            |
| seq_length                 | 2048     |                                            |
| fp16                       | True     | use fp16 16-bit (mixed) precision training |


| deepspeed ZeRO stage 3 parameters | value | notes                                                        |
| --------------------------------- | ----- | ------------------------------------------------------------ |
| overlap_comm                      | true  | Attempts to overlap the reduction of the gradients with backward computation |
| contiguous_gradients              | true  | Copies the gradients to a contiguous buffer as they are produced. Avoids memory fragmentation during backward pass. |
| sub_group_size                    | 1e9   |                                                              |
| stage3_max_live_parameters        | 1e9   | The maximum number of parameters resident per GPU before releasing. Smaller values use less memory, but perform more communication. |
| stage3_max_reuse_distance         | 1e9   |                                                              |

| Running Status (est.) | value               |
| --------------------- | ------------------- |
| time per step         | 1m20s               |
| total time            | 22h~25h             |
| GPU average usage     | 22024MiB / 32510MiB |

