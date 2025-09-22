# make sure current folder is OSPO
# bash script/run.sh

CUDA_VISIBLE_DEVICES=0

python ospo/step1.py --category color1
python ospo/step1.py --category color2
python ospo/step1.py --category texture1
python ospo/step1.py --category texture2
python ospo/step1.py --category shape1
python ospo/step1.py --category shape2
python ospo/step1.py --category 2D_spatial
python ospo/step1.py --category 3D_spatial
python ospo/step1.py --category numeracy1
python ospo/step1.py --category numeracy2
python ospo/step1.py --category non-spatial
python ospo/step1.py --category complex

CUDA_VISIBLE_DEVICES=0,1,2,3

python ospo/step2_1.py
python ospo/step2_2.py 
python ospo/step3_1.py 
python ospo/step3_2.py 
python ospo/step4.py 
