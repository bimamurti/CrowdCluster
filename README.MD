Dense Crowd Trajectory Prediction:
To get the cluster run main.py
before run please open the file and set the following variable
directoryGT as the groundtruth directory (just for comparison)
directory as the directory of the tracking data (whether it is ground truth or the tracking result)
start as the frame to start
finish as the frame to finish the process
tdist is the location threshold hyperparameter
tdirect is the direction threshold hyper parameter (0-180 degree)
filename is for saving the cluster output

if you want to process the social vae and trajectron++ result bring the result of this process to socialVAE and Trajectron++