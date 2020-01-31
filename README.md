
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

We have made available several cfg configuration files for the experiments, if you want to run the experiment with the traditional SincNet (with no use of the improved AM-Softmax layer) you must use the [*SincNet_TIMIT.cfg*](cfg/SincNet_TIMIT.cfg) file, otherwise you can use the [*SincNet_TIMIT_m0XX.cfg*](cfg/) file where the *XX* denotes the size of the margin parameter that will be used for the AM-Softmax layer.


## Results
When training have a look at the *cfg* configuration file, the output paths for the model and the result (*res.res*) files are placed there.

We have also made available some results from our experiments, you can check them at [*exp*](exp/) folder. The resume of the results are saved in the *res.res* files.
