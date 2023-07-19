# Open Scene Understanding
Grounded Situation Recognition Meets Segment Anything for Helping People with Visual Impairments
![My Image](img/Flowchart.png)
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
