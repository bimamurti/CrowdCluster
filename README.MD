# Dense Crowd Trajectory Prediction
--- 

## Installation
to install we suggest to create new conda environment 
``` conda create -n clusterenvironment python==3.9.18```
```    conda activate clusterenvironment ```

## Dependencies
1. matplotlib 3.8.2
2. pandas 2.2.0
3. sklearn 1.4.0
Install the dependecies with pip or conda command
``` pip install matplotlib==3.8.2 pandas==2.2.0 sklearn==1.4.0```

### Run the code
before run main.py please open the file and set the following variable:\
1. directoryGT as the groundtruth directory (just for comparison)\
2. directory as the directory of the tracking data (whether it is ground truth or the tracking result)\
3. start as the frame to start\
4. finish as the frame to finish the process\
5. tdist is the location threshold hyperparameter\
6. tdirect is the direction threshold hyper parameter (0-180 degree)\
7. filename is for saving the cluster output\

gt02,gt03,gt04 are ground truth tracking data
pred2,pred3,pred4 are Trajectory prediction output from headhunter-T
example:
``` directoryGT='gt03.txt'
directory='gt03.txt'
start=75
finish=950
tdist=110
tdirect=50
filename = "MOT21GT02_outputfull.csv"
```

after setup then run the code

``` python main.py ```

the python code output will show several output values and graphs

If you want to process the social vae and trajectron++ result bring the result of this process to socialVAE [[[1]]] and Trajectron++ [[2]](#2).\
use this config file for both algorithm in configSocialVAE.py and configTrajectron.json\

* get the output file and convert the format according to the algorithm:
we used frames 0-600th for training, 
600th-800th for validating
and 800th-980th for evaluating
* execute the algorithm according their instruction with our config file
here the example of running script setting for trajectron++ to run the script:
```python train.py --eval_every 10 --vis_every 1 --train_data_dict mot213_train.pkl --eval_data_dict mot213_val.pkl --offline_scene_graph yes --preprocess_workers 5 --log_dir ../experiments/pedestrians/models --log_tag _eth_vel_ar3 --train_epochs 100 --augment --conf ../experiments/pedestrians/models/eth_vel/config.json --device cuda ```

## References
<a id="1">[1]</a> 
Salzmann, T., Ivanovic, B., Chakravarty, P., & Pavone, M. (2020). Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVIII 16 (pp. 683-700). Springer International Publishing.
<a id="2">[2]</a> 
P. Xu, J.-B. Hayet, and I. Karamouzas. Socialvae: Human trajectory prediction using timewise
latents. pages 511–528, 2022. doi: 10.1007/978-3-031-19772-7_30. URL https://link.
springer.com/10.1007/978-3-031-19772-7-30.




