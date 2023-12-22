# 测试说明


# 注意
此项目中的example_chat_completion.py已更新可以进行交互式测试
第一个问题随便问，不要记录，因为first-token-time很大，为0.5s左右；随后的问题可以直接问，不用clear掉历史记录，不用重新加载大模型。
以前的代码first token time大，是因为非交互式生成只输出第一次生成时的first-token-time（0.5s左右）
run.sh中的max_batch_size改小的原因是，max_batch_size=6，4，2时会Cuda OOM
## 替换代码
将此仓库中的example_chat_completion.py和generation.py替换到对应位置

## bash run.sh
复制此仓库中的run.sh
