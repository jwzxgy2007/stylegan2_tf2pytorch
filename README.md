# StyleGAN2_tf2pytorch

Convert checkpoint from offical tensorflow [StyleGAN2](https://github.com/NVlabs/stylegan2) to pytorch [StyleGAN2](https://github.com/nv-tlabs/editGAN_release/tree/release_final/models/stylegan2_pytorch)
### Requirements
- Python 3.6
 
- Pytorch >= 1.4.0.

- Tensorflow-gpu==1.14.0 (Offical [StyleGAN2](https://github.com/NVlabs/stylegan2) recommended)

- CUDA 10.0 needed to compile some ops from dnnlib


### Usage
```
python tf2pytorch.py --tf_cp *tensorflow checkpoint path* 
                     --output_path *output pytorch checkpoint path*
```
You can download pytorch checkpoint from  [google drive](https://drive.google.com/drive/folders/1NJWxFdPA21xI6x7PCbRPTMVoIYE_9z8o) provided by [editGAN](https://github.com/nv-tlabs/editGAN_release) and tensorflow checkpoint (stylegan2-car-config-f.pkl) from [google drive](https://drive.google.com/drive/folders/1yanUI9m4b4PWzR0eurKNq6JR1Bbfbh6L) provided by [StyleGAN2](https://github.com/NVlabs/stylegan2), and compare the weight diff (should be zero) between official tensorflow checkpoint and pytorch checkpoint.
```
python tf2pytorch.py --compare true
```
To load the converted checkpoint, you can
```
import pickle
with open(*checkpoint path*, 'rb') as f
    checkpoint_data = pickle.load(f)
net.load_state_dict(checkpoint_data, strict=False)
```
