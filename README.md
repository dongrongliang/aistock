# stockai

基于深度学习的证券市场趋势预测

## 目录结构说明

- cnn: 模型库
- main: 预测服务
- train_scipts:训练脚本
- utils: 工具库
- trend_ana: 统计学工具库
- data_source: 数据源及数据获取工具库

## 训练和调试

### 神经网络
1.训练脚本:

`
python3 -W ignore train_xy.py -fm -es -pid adres_ffn2_restrm2_db -bid adres_ffn2_restrm2_db -rlid adres_ffn2_restrm2_db -train_cn fullnewtech2_train -test_cn fullnewtech2_test -epo 100 -dh -relearn -batch 4000 -evalbn -lrate 0.001 -optim adam -wd 0
`

2.样本格式
样本生成脚本:
python3 data_souce/data_tushare.py
样本路径：
- data_source/data/
   - train
   stock_code.npz
   - test
   stock_code.npz

内置参数：
split_mode = 'stock' 按股票切分训练集测试集，‘time'按时间，''默认不切分
sid_file_lst = ['new_tech'] 股票代码文件名，存储路径:main/paras
train_ratio = 0.8 
test_ratio = 0.2
sample_name = 'fullnewtech2'
period = int(2 * 12 * 28) 单个股票取样周期总长度
model_period = 112 单个样本周期数
trend_zone = 5 标签模糊区域，用于数据增强
full_mode = True 42个维度行情数据， False 16个维度行情数据