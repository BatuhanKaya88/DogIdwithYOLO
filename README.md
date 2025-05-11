# Dog-Id-with-YOLO: A Stray Dog Attack Awareness and Security System

Project Overview:

We have created a system compatible with street cameras to raise awareness about stray dog attacks in Turkey and reduce the response time of security forces to these attacks. This system will notify security forces about incidents of dog attacks, achieving its intended purpose and contributing to the community. The project is still in development, and we aim to optimize the system further with an advanced notification system.

Note:

Currently, there is a small issue with the labeling system where humans are mistakenly labeled as "dog." However, the system can distinguish between humans and dogs correctly when applied properly. This will be addressed in future versions.

---

## Table of Contents

1. [Prerequisites](#prerequisites)

2. [Dataset Preparation](#dataset-preparation)

3. [Running the Project](#running-the-project)

4. [Training Process](#training-process)

5. [Model Configuration](#model-configuration)

6. [Acknowledgements](#acknowledgements)

---

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.6 or higher

- TensorFlow (v2.x)

- OpenCV

- Roboflow account to access the dataset

- Other necessary libraries listed in `quirements.txt`

Install dependencies using:

```bash

pip install -r requirements.txt

````

---

## Dataset Preparation

1. Download the Dataset:

   * Go to [Roboflow](https://roboflow.com), create an account, and select a suitable dataset for dog behavior recognition.

   * Download the dataset in the required format (preferably YOLO format).

2. Split the Dataset:

    Once downloaded, the dataset should be divided into *Train**, Test, and Validation sets.

   * You can use the `roboflow` tools to automate this, or you can manually split the dataset.

3. Directory Structure:

   * Organize the dataset into the following directories:

     * `train/`

     * `test/`

     * `valid/`

   The folder structure should look like this:

   ```

/path/to/your/project/

     ├── /train/

├── /test/

     └── /valid/

   ```

---

## Running the Project

### 1. Set Up Your File Paths

In the code, there are placeholders like `YOURDESKTOP` in several places. Make sure to replace these with the correct paths to your dataset and project files on your local machine.

Example:

```python

dataset_path = "C:/Users/YourName/Desktop/DogIdwithYOLO/dataset/"

```

### 2. Training the Model

* First, you need to train your model. Start by running the `test.py` file. This will initiate the initial training process.

  ```bash

  python test.py

  ```

* You can change the model architecture in the code. In the `model` section, there are predefined models. You can select a different model that fits your system and hardware capabilities.

### 3. Training the Classifier

* After completing the initial training, run the `train_classifier.py` file to continue the training process and fine-tune the classifier.

  ```bash

  python train_classifier.py

  ```

### 4. Finalizing the Model

* Once the classifier training is completed, run the `EfficientNetB0.py` script to finalize the model training and save the final model weights.

  ```bash

  python EfficientNetB0.py

  ```

* The model will be saved as `dog_human_classifier_model.h5`. You can replace this file with your own trained model if desired.

### 5. Running the Analysis

* After completing the training process, you can run the `analyze_dog_behavior.py` script to analyze the behavior of dogs in video or image streams.

  ```bash

  python analyze_dog_behavior.py

  ```

---

## Training Process

Here is a brief description of each step in the training process:

1. test.py:

   * This script handles the initial training of the object detection model using the dataset you provided.

2. train\_classifier.py:

   * After the base model training, this script fine-tunes the classifier, responsible for distinguishing between different behaviors of dogs and humans.

3. EfficientNetB0.py:

   * This script finalizes the model by training the EfficientNetB0 architecture and saving the model.

4. analyze\_dog\_behavior.py:

   * The main script that applies the trained model to analyze video or image streams, detecting dogs and humans, and notifying security forces.

---

## Model Configuration

You can modify the model used in this project by changing the model-related code. The architecture is set to a default model, but you can choose other models (e.g., ResNet, EfficientNet, etc.) by simply replacing the model name in the code.

```python

# Change the model in the code.

model = YOLO('yolov8n.pt')

```

Make sure to adjust the paths and other model-related parameters according to your system's requirements.

---

## Acknowledgements

* Roboflow: For providing the dataset and tools to build and organize the custom dataset.

* TensorFlow: For building the object detection model and training the classifier.

* OpenCV: For handling video and image processing in real-time.

---

## Future Developments

This system is still in development. The goal is to enhance the detection capabilities and implement a more sophisticated notification system for security forces. Additionally, optimization of the algorithm and model architecture will help improve accuracy and performance in real-world applications.

---

### Contact

If you encounter any issues or have any suggestions for improvements, feel free to open an issue or contact the project maintainer.

---

```

### Key Points in the README:

- Project Overview: A brief explanation of what the project does and its goal.

- Dataset Preparation: Instructions on how to get and prepare the dataset for training.

- Running the Project: Details on how to set up your environment, run scripts, and train the model.

- Training Process: Step-by-step explanation of the scripts involved in training the model.

- Model Configuration: How to change the model and adapt it to your needs.

- Future Developments: A note on where the project is headed.

This structure ensures the project is clear, easy to follow, and professional. If you need any further adjustments or additions, feel free to ask! 😊

```

Some photos

(https://github.com/user-attachments/assets/2f5e0af9-7070-4a70-98e8-ddb2e9165f3b)

