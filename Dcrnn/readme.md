**The way to train Dcrnn**

Step 1: Run "/Dcrnn/resample.sh" to resample the data.

Step 2: Run "/Dcrnn/filemaker.sh" to generate filemaker files

Step 3: Run "/Dcrnn/standadizatoin.sh" to generate standadization file

Step 4: Run "/Dcrnn/train.sh" to train and test Dcrnn

For most of the parameters you don't need to change them, you only to change the root paths related parameters. 

**Folder structure**

"/Dcrnn/data" All data loading and reprocessing related codes are in this folder

"/Dcrnn/model" All model structures are in this folder (usually no need to the codes in this folder)

**Data augmentation related code**  
Please refer to "/Pattn/data_provider/data_loader.py" for the data augmentation methods, there are specific comments and remarks for you to understand.
