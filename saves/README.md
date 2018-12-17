## `.tflite` Models

> **NOTE**  
> 以下网络的输入、输出均需要经过处理才能得到下一个网络的输入或可用结果。具体处理过程的移植可参考 [`mtcnn/mtcnn.py`](https://github.com/ipazc/mtcnn/blob/master/mtcnn/mtcnn.py)。  

### PNet - Proposal Network

- 签名：`pnetfixed`
- 输入：图像（OpenCV **BGR** 次序）
    - 大小：`( None, 1024, 1024, 3 )`
        - 由于 TFLite 限制，这里必须固定输入图像的大小，本份 `.tflite` 中设置为 1024 x 1024。输入图像的大小可由脚本 `save.py` 中 `INPUT_BOX_SIZE` 参数设置。
        - 可以考虑对输入图像滑动窗口。
- 输出：原始预测框、置信度
    - 大小：`( None, 507, 507, 4 )`, `( None, 507, 507, 2 )`
        - 输出热度图为卷积结果，其大小与输入图像大小相关，这里为 `507`。

### RNet - Refine Network

- 签名：`rnet`
- 输入：滑动窗口
    - 大小：`( None, 24, 24, 3 )`
- 输出：预测框、置信度
    - 大小：`( None, 4 )`, `( None, 2 )`

### ONet - Output Network

- 签名：`onet`
- 输入：人脸窗口
    - 大小：`( None, 48, 48, 3 )`
- 输出：预测框、人脸关键点（5 点）、置信度
    - 大小：`( None, 4 )`, `( None, 10 )`, `( None, 2 )`

