创建环境：conda env create -f environment.yml
激活环境：conda activate rechorus
针对 Mac 用户：如果安装失败，请删除 .yml 中的 pytorch 相关行，并参考 PyTorch 官网安装适配 Mac 的版本。
请确保你的项目文件夹路径里没有中文。
测试运行可以使用以下测试用代码（要先进入src文件夹）：
python main.py --model_name FuXiAlpha --dataset MovieLens_1M --epoch 5 --batch_size 1024 --emb_size 32 --lr 0.001 --num_layers 1 --num_heads 2 --gpu 0
