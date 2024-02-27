# 计算逻辑
初次前向传播是预填充阶段，接下来的第一次前向传播生成第一个token
所以在forward方法开始处获取开始时间，forward结束时获取结束时间，第一个token的生成时间就是第二次forward结束时的encode_time+decode_time
同时记录已进行过的forward次数(one_sec_tokens)，在每个forward结束时，判断时间是否超过1s，以记录1s内生成的token数（forward数量减1）
```python
        torch.cuda.synchronize()
        ed = time.time()
        #第一个forward是encoder的，第二个forward是第一个token生成，所以first_token_time = encode_time + decode_time1.
        if self.fwd_num == 1:
            self.encode_time += ((ed - st) * 1000)
            self.seq_len = seq_length
            self.one_sec_tokens += 1
        else:
            self.decode_time += ((ed - st) * 1000)
            if self.fwd_num==2:                             #计算first_token_time 
                self.first_token_time = self.encode_time+self.decode_time
            if self.decode_time+self.encode_time<=1000.5:   #判断时间是否超过1s
                self.one_sec_tokens += 1
```
在cli_demo中进行输出即可
```python
        print("Length of input ids(输入长度): %d" % (model.transformer.seq_len))
        print("Tokens generated in 1 sec（第一秒生成token数）: %d" % (model.transformer.one_sec_tokens))
        print("First token time（生成第一个token耗时）: %.4f ms" % (model.transformer.first_token_time))
        print("Generated token count（总生成token数）:%d" % (model.transformer.fwd_num - 1))
        print("Time per token（平均生成每个token用时）:%.4f ms" % ((model.transformer.encode_time+model.transformer.decode_time)/(model.transformer.fwd_num - 1)))
        model.transformer.one_sec_tokens=0
```
