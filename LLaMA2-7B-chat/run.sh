torchrun --nproc_per_node 1 /home/ubuntu/workspace/llama/example_chat_completion.py \
    --ckpt_dir /home/ubuntu/workspace/llama/llama-2-7b-chat/ \
    --tokenizer_path /home/ubuntu/workspace/llama/llama-2-7b-chat/tokenizer.model \
    --max_seq_len 4096 --max_batch_size 1
