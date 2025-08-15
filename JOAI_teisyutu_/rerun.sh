cd /kaggle/working/JOAI_teisyutu
python data/data_transform.py
python data/extract_temp_test.py
python data/add_feature_test.py
python data/extract_temp_train.py
python data/add_feature_train.py
python joai/exp0/exp0.py
rm -r joai/exp0/checkpoints
rm -r joai/exp0/models
python joai/exp1/exp1.py
python joai/exp3/exp3.py
rm -r joai/exp3/checkpoints
rm -r joai/exp3/models
python joai/exp4/exp4.py
python joai/exp5/exp5.py
python joai/exp6/exp6.py
python joai/exp7/exp7.py
python joai/exp8/exp8.py
python joai/exp9/exp9.py
python joai/exp11/exp11.py
python joai/exp13/exp13.py
rm -r joai/exp13/checkpoints
rm -r joai/exp13/models
python joai/exp14/exp14.py
rm -r joai/exp14/checkpoints
rm -r joai/exp14/models
python joai/exp16/exp16.py
rm -r joai/exp16/checkpoints
rm -r joai/exp16/models
python joai/exp21/exp21.py
rm -r joai/exp21/checkpoints
rm -r joai/exp21/models
python joai/exp22/exp22.py
rm -r joai/exp22/checkpoints
rm -r joai/exp22/models
python joai/exp24/exp24.py
rm -r joai/exp24/checkpoints
rm -r joai/exp24/models
python joai/exp26/exp26.py
rm -r joai/exp26/checkpoints
rm -r joai/exp26/models
python joai/exp28/exp28.py