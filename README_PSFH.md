## How to run for PS-FH-AOP Challenge

- **准备环境**

  请参考 docs/zh_cn/get_started.md 或 docs/en/get_started.md
- **准备数据集**

  请将数据集组织成如下形式
```none
YourDataRoot
├── images
│   ├── train
│   │   ├── 0001.png
│   │   ├── 0004.png
│   │   ├── ...
│   ├── val
│   │   ├── 0002.png
│   │   ├── 0003.png
│   │   ├── ...
├── labels
│   │   ├── 0001.png
│   │   ├── 0004.png
│   │   ├── ...
│   ├── val
│   │   ├── 0002.png
│   │   ├── 0003.png
│   │   ├── ...
```

  数据集组织可参考 my_projects/FH-PS-AOP/dataset_prepare.py, my_projects/FH-PS-AOP/dataset_split.py

- **准备config**

  修改my_projects/FH-PS-AOP/configs/dataset.py中的*data_root* 到 *YourDataRoot*

- **训练模型**

```shell
  python.exe .\tools\train.py .\my_projects\FH-PS-AOP\configs\upernet_r101_4xb4-aw-dice-3.0-ce-1.0-80k_psfh-256x256.py
```

- **测试结果**
- 推理.png数据

  参考my_projects\FH-PS-AOP\inference.py, 需先修改路径
- 推理.mha数据

  参考my_projects\FH-PS-AOP\inference_mha.py, 需先修改路径