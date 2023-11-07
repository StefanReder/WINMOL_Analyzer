# WINMOL_Analyzer

Installation in conda env:

conda create --name WINMOL_Analyzer python==3.9

conda activate WINMOL_Analyzer

git clone https://github.com/StefanReder/WINMOL_Analyzer

python3 -m pip install tensorflow[and-cuda]

Verify that the GPU is working

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

pip install -r requirements.txt

python -m ipykernel install --user --name=WINMOL_Analyzer
