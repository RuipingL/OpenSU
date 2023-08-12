# Open Scene Understanding (ACVR 2023)[[arxiv](https://arxiv.org/abs/2307.07757)] 
Grounded Situation Recognition Meets Segment Anything for Helping People with Visual Impairments
![My Image](img/Flowchart.png)
<p align="center">
  <img src="https://github.com/RuipingL/OpenSU/blob/main/img/comparison.png" width="600">
</p>

## Environment
```
# Clone this repository and navigate into the repository
git clone https://github.com/RuipingL/OpenSU.git    
cd OpenSU                                          

# Create a conda environment, activate the environment and install PyTorch via conda
conda create --name OpenSU python=3.9              
conda activate OpenSU                             
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge 

# Install requirements via pip
pip install -r requirements.txt

# Install Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```
## Dataset Preparation
Download [images](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip) to the folder `SWiG`, and [json files](https://github.com/jhcho99/CoFormer/tree/master/SWiG/SWiG_jsons) to the folder `SWiG/SWiG_jsons`.
## Model Checkpoints
Download 
[Swin-T](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) to the folder `ckpt/Swin`,
[Segment Anything](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt) to the folder `ckpt/sam`, and [GSR model](https://drive.google.com/file/d/1i44Y5YIJ7ECNq9lYBOd4Qlcp7TVL79zz/view?usp=drive_link) to the folder `ckpt`.
## Training 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py   --batch_size 4 --dataset_file swig --epochs 40 --num_workers 4 --num_glance_enc_layers 3 --num_gaze_s1_dec_layers 3 --num_gaze_s1_enc_layers 3 --num_gaze_s2_dec_layers 3 --dropout 0.15 --hidden_dim 512 --output_dir OpenSU
```
## Evaluation
```
python main.py --saved_model ckpt/OpenSU_Swin.pth --output_dir OpenSU_eva --dev  # Evaluation on develpment set
python main.py --saved_model ckpt/OpenSU_Swin.pth --output_dir OpenSU_eva --test # Evaluation on test set
```
## Demo
```
python demo.py --image_path img/carting_214.jpg --sam sam       # Using vanilla Segment Anything as segmentation map generator
python demo.py --image_path img/carting_214.jpg --sam mobilesam # Using MobileSAM as segmentation map generator
```
Output:
```
# Text information
verb: carting 
role: agent, noun: dog.n.01 
role: item, noun: man.n.01 
role: tool, noun: cart.n.01 
role: place, noun: outdoors.n.01 
the dog cartes the man in a cart at a outdoors.
```
<img src="https://github.com/RuipingL/OpenSU/blob/main/img/carting_214_sam.jpg" width="300">

