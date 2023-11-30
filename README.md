
# Real-time Body Measurement using Pose Estimation and Deep Learning

## Project Overview
This project offers a real-time solution for body measurement using advanced pose estimation techniques and deep learning. It leverages the power of MediaPipe and OpenCV to accurately measure various body parts such as hips, shoulders, upper body, and legs. This tool can be used in various domains including fitness, fashion, and health monitoring.

## Features
- Real-time capturing of body measurements using a webcam.
- Utilization of MediaPipe for accurate pose estimation.
- Calculation of distances between various body landmarks.
- Support for multiple persons' measurements.
- Data storage for both input and calculated measurements.

## Installation

### Prerequisites
- Python 3.x
- OpenCV
- MediaPipe
- A webcam or a video input device

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ntthienphuc/Real-time_Body_Measurement_using_Pose_Estimation_and_Deep_Learning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Real-time_Body_Measurement_using_Pose_Estimation_and_Deep_Learning
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To start the application, run the following command in the terminal:
```bash
python main.py
```
Follow the on-screen prompts to enter the number of persons and their measurements.

## How It Works
- The application captures video from a webcam and uses MediaPipe's pose estimation to identify body landmarks.
- Measurements between specific landmarks (hips, shoulders, etc.) are calculated.
- Users are prompted to input real measurements for comparison purposes.
- Data is saved in a local folder for further analysis.

## Contributing
Contributions to this project are welcome. Please follow these steps:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`.
3. Make your changes and commit them: `git commit -m 'commit message'`.
4. Push to the original branch: `git push origin Real-time_Body_Measurement_using_Pose_Estimation_and_Deep_Learning/[location]`.
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- Special thanks to the MediaPipe and OpenCV communities for their invaluable resources.
