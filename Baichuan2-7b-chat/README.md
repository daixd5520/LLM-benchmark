# 测试方式
## 准备环境
python版本3.10.12或者3.11，pip intall一下/.../Baichuan2/requirements.txt
## 准备模型
在us3://aigc-llm 里面找到Baichuan2-7B-Chat，下载
## 准备代码
### git clone
https://github.com/baichuan-inc/Baichuan2.git
### 替换代码
将项目代码对应替换成此仓库中的cli_demo.py和modeling_baichuan.py
## 运行测试
python cli_demo.py
###注意事项
1.第一次对话， *Tokens generated in 1 sec* 往往很小，甚至为0（若first_token已经用时1s以上）；第一次对话模型生成结束之后进行clear，再输入新的问题可变成正常数量。
2.每次测试完一个用例，若不clear，input_length会包含历史记录，建议使用clear清除历史记录。
