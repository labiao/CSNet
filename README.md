## CSNet: Change Selection of Activations and Pseudomasks for Image-Level Weakly Supervised Change Detection(TGRS)

<!-- ![](https://github.com/labiao/CSNet/blob/main/figure/CSNet.jpg?raw=true){: .center} -->

<p align="center">
<img src="https://github.com/labiao/CSNet/blob/main/figure/CSNet.jpg?raw=true">
</p>

__Official pytorch implementation of "CSNet: Change Selection of Activations and Pseudomasks for Image-Level Weakly Supervised Change Detection"__


## Updates

14 July, 2025: Initial upload


## Requirement 

- This code is tested on Ubuntu 20.04, with Python 3.6, PyTorch 1.7.1, and CUDA 11.3.

### Dataset & pretrained checkpoint

- Prepare dataset, image-level labels, and pretrained checkpoints
  - Example directory hierarchy
  ```
  CSNet
  |--- dataset
  |    |--- BCD
  |    |        |---A1
  |    |        |---B1
  |    |        |---amn_train.txt  # used for ASPP module
  |    |        |---amn_train_change.txt  # used for ASPP module
  |    |        |---npy5.npy
  |    |        |---train5.txt  # used for classifier training
  |    |        |---val5.txt
  |--- pth
  |    |--- RepVGG-B1g2-train.pth
  |    |--- sam_vit_h_4b8939.pth
  |--- exp_rep
  |    |--- ckpt
  |    |--- runs
  |    |--- exp_rep.log
  |    |--- eval_exp_rep.log
  |    |--- infer.log
  |--- result
  |    |--- amn_ir_label
  |    |--- amn_cam
  |    |--- amn_mask
  |    |--- amn_SAMlabelV1
  |    | ...
  | ...
  ```


## Execution

### Pseudo-mask generation

- Execute the bash file.
    ```bash
    # Please see these files for the detail of execution.
    bash script/train_script.sh
    ```

### Segmentation network
Fort the segmentation network, we experimented with BIT based on [PaddleRS](https://github.com/PaddlePaddle/PaddleRS).


## Acknowledgement
This code is highly borrowed from [IRN](https://github.com/jiwoon-ahn/irn), [SIPE](https://github.com/chenqi1126/SIPE), [CS-WSCDNet](https://github.com/WangLukang/CS-WSCDNet). Thanks to Jiwoon, Ahn, Qi Chen, Lukang Wang.

