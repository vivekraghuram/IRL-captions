wget "http://cs231n.stanford.edu/coco_captioning.zip"
unzip coco_captioning.zip
rm coco_captioning.zip

wget "http://www-nlp.stanford.edu/data/glove.840B.300d.zip"
unzip glove.840B.300d.zip
mkdir glove
mv glove.840B.300d.txt glove/
rm glove.840B.300d.zip
cd ../
python preprocess_captions.py
python preprocess_captions_glove.py

# get coco image
curl https://sdk.cloud.google.com | bash
mkdir data
cd data
mkdir images
cd images
gsutil -m rsync gs://images.cocodataset.org/train2014 train2014

# process images
cd ../../datasets
mkdir processing
cd ../../
python preprocess_images.py