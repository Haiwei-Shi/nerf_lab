import csv, time, cv2, os, sys
import numpy as np

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
from gen3_7dof.tool_box import TCPArguments, euler_to_rotation_matrix
from gen3_7dof.utilities import DeviceConnection


def getRotMtx(raw_pose):
    # Take raw pose from the kinova and convert to rotation matrix
    # Need to convert to radian
    R_0T = euler_to_rotation_matrix(raw_pose.theta_x * np.pi / 180,
                                    raw_pose.theta_y * np.pi / 180,
                                    raw_pose.theta_z * np.pi / 180)

    return R_0T


def get_world_EE_HomoMtx(base):
    # Get the homogeneous matrix of the end effector in the world frame
    current_pose = base.GetMeasuredCartesianPose()
    R = getRotMtx(current_pose)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([current_pose.x, current_pose.y, current_pose.z])
    
    return T


# Compute the camera's homogeneous transformation matrix in the world frame
def get_world_cam_HomoMtx(H_world_EE):
    # Camera's location w.r.t. the end effector in its local frame
    P_cam_EE = np.array([0, -0.045, 0.055, 1])

    # Transform camera position into world frame
    P_cam_world = H_world_EE @ P_cam_EE
    camera_position = P_cam_world[:3]

    """
    We are also able to get roll, pitch, yaw by Thetax, Thetay, Thetaz
    Right now the results align perfectly with the numbers in the website (Kinova End Effector Thetax, Thetay, Thetaz)
    """

    # Camera orientation is the same as the end effector
    R_world_cam = H_world_EE[:3, :3]
    
    # Convert rotation matrix to Euler angles
    yaw = np.arctan2(R_world_cam[1, 0], R_world_cam[0, 0])
    pitch = np.arcsin(-R_world_cam[2, 0])
    roll = np.arctan2(R_world_cam[2, 1], R_world_cam[2, 2])

    return {
        "camera_x": camera_position[0],
        "camera_y": camera_position[1],
        "camera_z": camera_position[2],
        "camera_roll": np.degrees(roll),
        "camera_pitch": np.degrees(pitch),
        "camera_yaw": np.degrees(yaw),
    }


# Define image save directory
base_dir = "data"
image_subdir = "flower_new"  # Change this dynamically if needed
save_dir = os.path.join(base_dir, image_subdir)
os.makedirs(save_dir, exist_ok=True)  # ensure the directory exists

# Define CSV file for saving pose + image info
csv_filename = os.path.join(save_dir, "cam_pose_log.csv")


def get_next_image_filename(directory):
    # Find the next available image filename in sequence (00001.jpg, 00002.jpg, ...)
    existing_files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]) if x.split(".")[0].isdigit() else 0
    )

    if existing_files:
        last_num = int(existing_files[-1].split(".")[0])
        next_num = last_num + 1
    else:
        next_num = 1  # Start from 00001 if no files exist

    return f"{image_subdir}/{next_num:05d}.jpg"


def save_pose_to_csv(pose, img_filename):
    file_exists = os.path.isfile(csv_filename)

    # Attach the image filename to pose data
    pose["image_filename"] = img_filename

    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=pose.keys())

        if not file_exists:
            writer.writeheader()  # Write header only once

        writer.writerow(pose)
        print(f"Pose saved to CSV: {pose}")


def capture_image(pose):
    ret, frame = cap.read()
    if ret:
        image_filename = get_next_image_filename(save_dir)  # Relative path (e.g., flower/00001.jpg)
        image_path = os.path.join(base_dir, image_filename) 
        cv2.imwrite(image_path, frame)
        print(f"Image saved: {image_filename}")
        save_pose_to_csv(pose, image_filename)

    else:
        print("Failed to capture image.")


try:
    print("Move the robot arm to the desired position and press 'c' to capture an image.")
    print("Press 'z' to quit.")

    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Turn on the robot arm
    tcp_args = TCPArguments()
    with DeviceConnection.createTcpConnection(tcp_args) as router:
        base = BaseClient(router)

        while True:
            H_world_EE = get_world_EE_HomoMtx(base)
            camera_pose = get_world_cam_HomoMtx(H_world_EE)
            pose_data = {**camera_pose}

            ret, frame = cap.read()
            if ret:
                cv2.imshow("Camera", frame)

            # Wait for user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # press 'c' to capture an image
                capture_image(pose_data)

            elif key == ord('z'):  # Press 'Q' to exit
                print("Exiting...")
                break

            time.sleep(0.5)  # Adjust frequency of logging


except KeyboardInterrupt:
    print("Process interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
