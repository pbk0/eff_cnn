# the datasets are backed up at https://console.cloud.google.com/storage/browser/phd-datasets/eff_cnn

# original dataset for ascad-v1-fk
#   wget https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip
#   Note it is large so we have uploaded small version from eff_cnn here
wget https://storage.googleapis.com/phd-datasets/eff_cnn/ASCAD_dataset.zip
unzip ASCAD_dataset.zip
rm ASCAD_dataset.zip

# original dataset for ascad-v1-vk
wget https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-083349/ascad-variable.h5

# other datasets copied from eff cnn which we might not use
#wget https://storage.googleapis.com/phd-datasets/eff_cnn/AES_HD_dataset.zip
#wget https://storage.googleapis.com/phd-datasets/eff_cnn/AES_RD_dataset.z01
#wget https://storage.googleapis.com/phd-datasets/eff_cnn/AES_RD_dataset.zip
#wget https://storage.googleapis.com/phd-datasets/eff_cnn/DPAv4_dataset.zip
#unzip AES_HD_dataset.zip
#unzip DPAv4_dataset.zip
#cat AES_RD_dataset.z* > AES_RD_dataset_full.zip
#unzip AES_RD_dataset_full.zip