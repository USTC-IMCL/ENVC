# ENVC
Offical implementation of [Learning Cross-Scale Prediction for Efficient Neural
Video Compression"](https://arxiv.org/abs/2112.13309v2)


## Installation

We recommend using conda environment for installation.

```bash
conda create --name $ENV_NAME
conda activate $ENV_NAME
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip3 install -r requirements.txt
cd third_party/msda
bash make.sh
```

## Evaluation

Currently, we only provide the evaluation code without practical entropy coding, 
which will be completed in the future.

### Preparing test sequences

Test sequences are required to be in sRGB format. 
The folder structure of test sequences is like this:

	|-- testset
		|-- sequence1
			|-- 000001.png
			|-- 000002.png
			|-- 000003.png
			...
		|-- sequence2
		|-- sequence3
		...

### Downloading pre-trained models
The pretained model can be downloaded [here](https://drive.google.com/drive/folders/1Yj7bKL6xAgtxwm1ycaDp7JL6DLlPWGxJ?usp=share_link).
We currently only provide the models optimized with MSE metric.

| Model    | Name    | config_file                |
|----------|---------|----------------------------|
| ENVCwAR  | mse512  | ./cfg/low_rate_model.yaml  |
| ENVCwAR  | mse1024 | ./cfg/low_rate_model.yaml  |
| ENVCwAR  | mse2048 | ./cfg/low_rate_model.yaml  |
| ENVCwAR  | mse3072 | ./cfg/high_rate_model.yaml |
| ENVCwAR  | mse4096 | ./cfg/high_rate_model.yaml |

### Evaluating a pre-trained model

Run the evaluation script `evaluate.py`:

```bash
python evaluate.py -h
```

This will give you a list of options. 
To evaluate a single sequence in the testset mentioned above, you can run:

```bash
python evaluate.py -i /testset/sequence1 -c ./cfg/low_rate_model.yaml --ckpt_path /ckpt/ENVCwAR/mse512.pth 
```

This will give the logout like:
```
[Frame sequence1 000001] type I bpp 0.175401 psnr 37.8629 msssim 0.980757
[Frame sequence1 000002] type P bpp 0.042709 psnr 36.9020 msssim 0.978645
[Frame sequence1 000003] type P bpp 0.052453 psnr 36.8629 msssim 0.978108
...
[Sequence '/testset/sequence1'] bpp 0.054796 psnr 36.3074 msssim 0.973306
[Dataset '/testset'] bpp 0.054796 psnr 36.3074 msssim 0.973306
```

You can also evaluate multiple test sequences as once using the path in glob 
pattern:

```bash
python evaluate.py -i "/testset/sequence*" -c ./cfg/low_rate_model.yaml --ckpt_path $CKPT_PATH 
```


## Citation

If you use this library for research purposes, please cite:
```
@article{guo2021learning,
  title={Learning cross-scale prediction for efficient neural video compression},
  author={Guo, Zongyu and Feng, Runsen and Zhang, Zhizheng and Jin, Xin and Chen, Zhibo},
  journal={arXiv preprint arXiv:2112.13309},
  year={2021}
}
```