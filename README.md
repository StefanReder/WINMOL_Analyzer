# WINMOL_Analyzer

Installation:

git clone https://github.com/StefanReder/WINMOL_Analyzer


pip install -r requirements.txt

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

pip install tensorflow

Verify that the GPU is working

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
