# Seizure-detect-Yan

![alt text](<Screenshot 2025-08-19 090905.png>)

--/Pattn/main_cls.py: 
    Main function for training Pattn. Including args settings, training, & result visualization.

--/Pattn/train.sh: 
    Script for training.

--/Pattn/resample.py: 
    Function for resample the TUSZ EEG data

--/Pattn/resample.sh: 
    Script for resampling.

--/Pattn/data_provider/data_loader:
    Dataset implementations & I/O for EEG data. Including **data augmentation, normalization/standardazation, etc.**

--/Pattn/models/PAttn.py:
    Structure of Pattn model.

--/Pattn/results:
    Results of all data augmentaion methods.