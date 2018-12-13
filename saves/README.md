## `.tflite` Models

> **NOTE**  
> 以下网络的输入、输出均需要经过处理才能得到结果，或作为下一个网络的输入。具体处理方法请见 `mtcnn/mtcnn.py`！  

### PNet - Proposal Network

- 名称：`pnetfixed`
- 输入：图像（OpenCV BGR 次序）
    - 大小：`( None, 1024, 1024, 3 )`
        - 输入图像的长、宽可由脚本 `save.py` 中 `INPUT_BOX_SIZE` 参数调整
- 输出：原始预测框、置信度
    - 大小：`( None, 507, 507, 4 )`, `( None, 507, 507, 2 )`
        - 输出热度图的大小为卷积结果，随输入图像大小改动而改动

### RNet - Refine Network

- 名称：`originalrnet`
- 输入：滑动窗口
    - 大小：`( None, 24, 24, 3 )`
- 输出：预测框、置信度
    - 大小：`( None, 4 )`, `( None, 2 )`

### ONet - Output Network

- 名称：`originalonet`
- 输入：人脸窗口
    - 大小：`( None, 48, 48, 3 )`
- 输出：预测框、人脸关键点（5 点）、置信度
    - 大小：`( None, 4 )`, `( None, 10 )`, `( None, 2 )`

