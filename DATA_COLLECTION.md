# Data Collection Guide

Update V1.1.1 (250526)

## 1. Preparation Process
a) Mount the GoPro to the UMI, power it on, and check if it has power and an SD card inserted.  

b) Parameter configuration: Use the GoPro to scan the configuration QR code below.
<div style="text-align: center"><img width="30%" src="assets/QR-MHDMI1mV0r27Tp60fWe0hS0sLcFg1dV.png"></div>

c) Time calibration: Access `umi-gripper.github.io/qrlocal` and align the scanner for a few seconds (a "scan successful" prompt will pop up every second).  

d) Connect your computer/phone (with proper ssh tool *e.g.* Terminus) to the same LAN as OrangePi, then establish an SSH connection to the Orange Pi. 

e) Turn on the VR headset and check if the headset and left/right controllers have power (controller battery levels will be displayed in AR).

f) Connect the VR headset to the same LAN as OrangePi ("network restricted" status for Quest is normal).


## 2. Collection Process (Operations to Perform for Each Set of Data)
a) Server side: Use SSH to connect to the Orange Pi from a computer/phone. First, run `cd ARCap && python check_left_right.py` and test the tactile sensor according to the instructions. Then run the command `python data_collection_umi.py`.  

b) Determine boundaries: In VR, open the XYCap program (Github icon) in the lower right corner. If prompted, create a new boundary and try to frame the range where your hand will move next. If the boundary is inappropriate, step out of the circle and then create a new boundary.  

c) Client connection: Enter the Orange Pi IP in the pop-up IP address box and confirm with the A key. (If the VR displays an IP address of 127.0.0.1, the Wi-Fi is not connected.)

d) Determine the coordinate system: When prompted with "Data 
Collection, Y: place robot", use the left joystick to adjust the Z-axis position and rotation, and the right joystick to adjust the XY-plane position. Ensure the X-axis points to the right, the Y-axis points forward, and the origin is at table height in front of your body. Press Y and then X to confirm the coordinate system.  

e) Start recording poses: A green box will appear with the prompt "Not Recording". Press A to start recording (the box will turn red) and maintain this state while recording data. You should see the prompt "successfully saved pose" appear from time to time.

f) Record calibration video: Place the marker (6cm*6cm, `assets/aruco_gripper_4_letter.pdf`) and record a video of moving the UMI quickly left and right in front of the cube.

g) Record data: Follow the sequence: power on → perform operations → power off. Note: Do not let your hand appear in front of the camera, and each video must start with an empty gripper. Objects can be placed in various positions.


## 3. Post-Processing
a) Turn off the VR headset screen and charge it.  

b) Remove the GoPro, read the videos from the SD card. Create a large folder for each batch: place the calibration video in the `latency_calibration` subfolder, and the remaining videos in the `raw_videos` subfolder. Upload the folder to your server or workstation.  

c) Upload the collected tactile data in the OrangePi to your server or workstation. Copy the `tactile_xxxx` folder (e.g., `tactile_20250322_214647`, `tactile_20250322_221154`) to the corresponding video folder. The final format should be similar to the structure below:  
  ```
  batch_1/
  ├─ latency_calibration/
  ├─ raw_videos/
  └─ tactile_20250322_214647/
  batch_2/
  ├─ latency_calibration/
  ├─ raw_videos/
  └─ tactile_20250322_221154/
  ```  
  Meanwhile, check if the number of files in the tactile data folder is reasonable.  


## Notes
1. One calibration corresponds to one data batch; each batch is recommended to contain approximately 50 entries.  
2. Try not to touch the table surface during collection.  
3. Do not remove the VR headset or step out of the boundary during collection.  
4. Pay attention to whether the collection program reports errors or exits during the process. If the VR headset is removed (VR screen turns black) or the Orange Pi restarts, stop the current batch. If the tactile sensor comes unglued, stop all collection.  
5. Lazy trick: If a video needs to be discarded, wait a little longer to continue recording before stopping, making the video longer.  


## Data Processing Process
1. Run `python run_arcap_pipeline.py data/task/batch_x` and check for errors.  
2. Check `data/task/batch_x/latency_calibration/latency_trajectory_final_algo_2.pdf` to see if the two curves are aligned:  
   a) If they are not aligned, set a positive `--init_offset` to shift the AR curve to the left, or a negative offset to shift it to the right.  
3. If everything is normal, `data/task/batch_x/dataset_plan.pkl` will be saved.  
4. Run `python scripts_slam_pipeline/AR_07_generate_replay_buffer.py data/task/batch_* -o data/task/dataset.zarr.zip`.  
