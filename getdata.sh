if [ ! -d "data" ]; then
    mkdir -p data
    cd data
    wget https://bitbucket.org/soegaard/mtl-cnn/raw/bd240abfe4b09176a400c8e2264d7eb3249c4071/naacl16-data/google_com_train.conll
    wget https://bitbucket.org/soegaard/mtl-cnn/raw/bd240abfe4b09176a400c8e2264d7eb3249c4071/naacl16-data/google_com_dev.conll
    wget https://bitbucket.org/soegaard/mtl-cnn/raw/bd240abfe4b09176a400c8e2264d7eb3249c4071/naacl16-data/google_com_test.conll
    cd ../
fi
