ieee-fraud-detection
==============================

https://www.kaggle.com/c/ieee-fraud-detection

Best score: Public LB 0.958766 (lightgbm + 'magic' feature)
------------
<img width="735" alt="PublicLB_0 958766" src="https://user-images.githubusercontent.com/24473602/71526678-53806280-291b-11ea-9b76-6db07546fc17.png">

Project Organization
------------
    ├── data  
    │   ├── external  
    │   ├── interim  
    │   ├── processed  
    │   │   ├── 0002_submission.csv  
    │   │   ├── features_test.pkl  
    │   │   ├── features_train.pkl  
    │   │   ├── magic_test.pkl  
    │   │   ├── magic_train.pkl  
    │   │   ├── nroman_test.pkl  
    │   │   ├── nroman_train.pkl  
    │   │   ├── raw_test.pkl  
    │   │   └── raw_train.pkl  
    │   ├── raw  
    │   │   ├── sample_submission.csv  
    │   │   ├── test_identity.csv  
    │   │   ├── test_transaction.csv  
    │   │   ├── train_identity.csv  
    │   │   └── train_transaction.csv  
    │   └── submission  
    ├── LICENSE  
    ├── log  
    │   ├── main  
    │   │   ├── 0000.log  
    │   │   ├── 0001.log  
    │   │   ├── 0002.log  
    │   │   └── 0003.log  
    │   └── train  
    │       ├── 0000.tsv  
    │       ├── 0001.tsv  
    │       ├── 0002.tsv  
    │       └── 0003.tsv  
    ├── Makefile  
    ├── models  
    │   ├── 0000_lgb_model.pkl  
    │   ├── 0001_lgb_skl_model.pkl  
    │   ├── 0002_xgb_model.pkl  
    │   └── lgb_model.pkl  
    ├── notebooks  
    │   ├── 00-debug.ipynb  
    │   ├── 01-eda-and-models.ipynb  
    │   └── jupyter.log  
    ├── README.md  
    └── src  
      ├── config  
      │   ├── config_0000.py  
      │   ├── config_0001.py  
      │   ├── config_0002.py  
      │   └── config_0003.py  
      ├── configure.py  
      ├── experiment.py  
      ├── feature_factory.py  
      ├── features  
      │   ├── altgor.py  
      │   ├── feature_base.py  
      │   ├── __init__.py  
      │   ├── magic.py  
      │   ├── nroman.py  
      │   └── raw.py  
      ├── main.py  
      ├── modelapi_factory.py  
      ├── models  
      │   ├── base_modelapi.py  
      │   ├── base_trainer.py  
      │   ├── __init__.py  
      │   ├── lgb_modelapi.py  
      │   ├── lgb_model.py  
      │   ├── lgb_skl_modelapi.py  
      │   ├── lgb_skl_model.py  
      │   ├── lgb_skl_trainer.py  
      │   ├── lgb_trainer.py  
      │   ├── xgb_modelapi.py  
      │   ├── xgb_model.py  
      │   └── xgb_trainer.py  
      ├── tests  
      │   ├── __init.py__  
      │   └── test_data.py  
      ├── trainer_factory.py  
      ├── transformer.py  
      └── utils  
          ├── __init__.py  
          ├── memory_reducer.py  
          ├── mylog.py  
          ├── reduce_mem_usage.py  
          └── seeder.py  

--------
