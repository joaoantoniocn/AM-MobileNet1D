
## Requirements
For running this experiment we used a Linux environment with Python 3.6.

You can see a list of python dependencies at [requirements.txt](requirements.txt).

To install it on conda virtual environment (`conda install --file requirements.txt`).

To install it on pip virtual environment (`pip install -r requirements.txt`).

## How to Run
To run it on TIMIT dataset we have first to pre-process the data, removing the start and ending silences moments and also normalizing the audio sentences.

``
python TIMIT_preparation.py $TIMIT_FOLDER $OUTPUT_FOLDER data_lists/TIMIT_all.scp
``

where:
- *$TIMIT_FOLDER* is the folder of the original TIMIT corpus
- *$OUTPUT_FOLDER* is the folder in which the normalized TIMIT will be stored
- *data_lists/TIMIT_all.scp* is the list of the TIMIT files used for training/test the speaker id system.

then, we can run the experiment itself by typing.

``
python speaker_id.py --cfg=cfg/$CFG_FILE
``

where:
- *$CFG_FILE* is the name of the cfg configuration file which is located at cfg folder.

We have made available several cfg configuration files for the experiments, if you want to run the experiment with the AM-MobileNet1D  you must use the [*AM_MobileNet1D_TIMIT.cfg*](cfg/AM_MobileNet1D_TIMIT.cfg) file, otherwise you can use the [*AM_MobileNet_XXX.cfg*](cfg/) file where the *XX* refers to the dataset name.


## Results
When training have a look at the *cfg* configuration file, the output paths for the model and the result (*res.res*) files are placed there.

We have also made available some results from our experiments, you can check them at [*exp*](exp/) folder. The resume of the results are saved in the *res.res* files.


## Cite us

If you use this code or part of it, please cite us!

```
@INPROCEEDINGS{9207519,  
  author={J. A. {Chagas Nunes} and D. {Macêdo} and C. {Zanchettin}},  
  booktitle={2020 International Joint Conference on Neural Networks (IJCNN)},   
  title={AM-MobileNet1D: A Portable Model for Speaker Recognition},   
  year={2020},  
  volume={},  
  number={},  
  pages={1-8},  
  doi={10.1109/IJCNN48605.2020.9207519}
 }
```
You can also find the paper at [IEEE](https://ieeexplore.ieee.org/abstract/document/9207519) or the preprint at [arXiv](https://arxiv.org/abs/2004.00132).

