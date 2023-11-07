# Extract annotation files

cd resources;
tar -xzf OCL_annot.tgz;
cd ..;



# Download and extract image data

echo "Downloading SUN-attribute";
mkdir -p SUN;
cd SUN;
if [ ! -e "SUNAttributeDB_Images.tar.gz" ]; then
    wget https://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB_Images.tar.gz;
fi
if [ ! -e "images" ]; then
    tar -xzf SUNAttributeDB_Images.tar.gz;
fi
cd ..;


echo "Downloading aPY";
mkdir -p aPY;
cd aPY;
if [ ! -e "ayahoo_test_images.tar.gz" ]; then
    wget http://vision.cs.uiuc.edu/attributes/ayahoo_test_images.tar.gz;
fi
if [ ! -e "VOCtrainval_14-Jul-2008.tar" ]; then
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar;
fi
if [ ! -e "ayahoo_test_images" ]; then
tar -xzf ayahoo_test_images.tar.gz;
tar -xf VOCtrainval_14-Jul-2008.tar;
cd ..;

echo "Downloading COCO";
mkdir -p COCO;
cd COCO;
wget http://images.cocodataset.org/zips/train2014.zip;
wget http://images.cocodataset.org/zips/val2014.zip;
unzip train2014.zip;
unzip val2014.zip;
cd ..;


# Prepare web image data
tar -xf data/from_web.tgz
python download_web_images.py