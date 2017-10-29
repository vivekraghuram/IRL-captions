wget "http://cs231n.stanford.edu/coco_captioning.zip"
unzip coco_captioning.zip
rm coco_captioning.zip

wget "http://www-nlp.stanford.edu/data/glove.840B.300d.zip"
unzip glove.840B.300d.zip
mkdir glove
mv glove.840B.300d.txt glove/
rm glove.840B.300d.zip
python preprocess.py