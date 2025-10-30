import smbus
import time
import json

DEVICE_AS5600 = 0x36
bus = smbus.SMBus(2)

def ReadRawAngle():
    try:
        read_bytes = bus.read_i2c_block_data(DEVICE_AS5600, 0x0C, 2)
        return (read_bytes[0] << 8) | read_bytes[1]
    except Exception as e:
        print(f"Failed to read sensor: {e}")
        return None

def calibrate_gripper():
    calibration_data = {
        "metadata": {
            "unit": "cm",
            "sensor_range": [0, 4095],
        },
        "calibration": []
    }
    
    print("=== Calibration of gripper width and as5600 readings ===")

    
    for width in range(11, 0, -1): # 11cm -> 1cm
        while True:
            user_input = input(f"Adjust your gripper width to {width}cm, and press ENTER to record: ")
            
            # 读取5次取平均
            readings = []
            for _ in range(5):
                if angle := ReadRawAngle():
                    readings.append(angle)
                    time.sleep(0.05)
            
            avg_angle = sum(readings) // len(readings)
            calibration_data["calibration"].append( (width / 100.0, avg_angle) )  # meter

            print(f"Recorded: width {width}cm -> angle value {avg_angle}")
            break

    with open("as5600_calib/gripper_calibration.json", "w") as f:
        json.dump(calibration_data, f, indent=2)
    
    print("\nSaved to gripper_calibration.json")

if __name__ == "__main__":
    calibrate_gripper()