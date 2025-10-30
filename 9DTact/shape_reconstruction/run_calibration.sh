echo "Start calibration. Please make sure that the camera device id in shape_config_[left|right].yaml is correct."

SIDE=$1

echo "Now start calibrating $SIDE sensor"
cp shape_config_$SIDE.yaml shape_config.yaml

echo "Step 1"
python _1_Camera_Calibration.py
echo "Step 2"
python _2_Sensor_Calibration.py
# echo "Step 3"
# python _3_Shape_Reconstruction.py

rm shape_config.yaml
echo "$SIDE sensor is calibrated"
