# Saved Model `.pb`

>   **Note**  
>   这些网络的输入、输出均需要通过额外的处理，详见 [mtcnn](https://github.com/ipazc/mtcnn/blob/master/mtcnn/mtcnn.py#L396) 实现！  

## Signatures

### 输入网络 / Proposal Network / `PNet`

**Inputs**  

-   `pnet/image_in`
    -   输入网络（PNet）的输入 placeholder
    -   shape: `[None, None, None, 3]`
        0.  batch_size
        1.  size
        2.  size
        3.  channels

**Outputs**  

-   `pnet/prob_out`
    -   人脸判别概率值输出
    -   shape: `[None, None, None, 2]`
-   `pnet/window_coord_out`
    -   人脸框坐标回归
    -   shape: `[None, None, None, 4]`
    -   坐标为 CSS 格式（left, top, width, height）

### 中间网络 / Refine Network/ `RNet`

**Inputs**  

-   `rnet/window_in`
    -   微调网络（RNet）的输入 placeholder
    -   shape: `[None, 48, 48, 3]`

**Outputs**  

-   `rnet/prob_out`
    -   人脸判别概率值输出
    -   shape: `[None, 2]`

-   `rnet/window_coord_out`
    -   人脸框坐标回归
    -   shape: `[None, 4]`

### 输出网络 / Output Network / `ONet`

**Inputs**  

-   `onet/window_in`
    -   输出网络（ONet）的输入 placeholder
    -   shape: `[None, 24, 24, 3]`

**Outputs**  

-   `onet/prob_out`
    -   人脸概率输出
    -   shape: `[None, 2]`
        0.  batch_size
        1.  score

-   `onet/window_coord_out`
    -   人脸框坐标回归
    -   shape: `[None, 4]`
        0.  batch_size
        1.  coordinates: `(left, top, width, height)`

-   `onet/five_points_out`
    -   5 点人脸关键点坐标回归
    -   shape: `[None, 10]`
