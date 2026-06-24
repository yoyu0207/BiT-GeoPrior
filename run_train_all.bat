@echo off
call D:\develop\miniconda3\Scripts\activate.bat SA
cd /d D:\yoyu\SA_Identification\project
 
echo [1/4] SNUNet
python train.py --model SNUNet --lr 5e-5 --epochs 100
 
echo [2/4] SNUNet_GeoAware
python train.py --model SNUNet_GeoAware --lr 5e-5 --epochs 100
 
echo [3/4] FCSiamDiff
python train.py --model FCSiamDiff --lr 5e-5 --epochs 100
 
echo [4/4] BiT_GWR
python train.py --model BiT_GWR --lr 6e-5 --epochs 200
 
echo Done
pause