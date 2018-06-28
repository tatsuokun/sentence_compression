# Simple Model for Sentence Compression
3-layered BILSTM model for sentence compression, referred as Baseline in [Klerke et al., NAACL 2016](http://aclweb.org/anthology/N/N16/N16-1179.pdf).
## Requirements
### Framework
 - python (<= 3.6)
 - pytorch (<= 0.3.0)
 
### Packages
 - torchtext
 
## How to run
```
./getdata
python main.py
```
To run the scripts with gpu, use this command `python main.py --gpu-id ID`, which ID is the integer from 0 to the number of gpus what you have. 
## Reference

```
@InProceedings{klerke-goldberg-sogaard:2016:N16-1,
  author    = {Klerke, Sigrid  and  Goldberg, Yoav  and  S{\o}gaard, Anders},
  title     = {Improving sentence compression by learning to predict gaze},
  booktitle = {Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2016},
  address   = {San Diego, California},
  publisher = {Association for Computational Linguistics},
  pages     = {1528--1533},
  url       = {http://www.aclweb.org/anthology/N16-1179}
}
```
