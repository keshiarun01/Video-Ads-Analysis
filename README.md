**Video Advertisement Analysis using Machine Learning**

### Description:
This project performs a comprehensive analysis of video advertisements by combining both textual and visual data. The project aims to evaluate various features of advertisements, such as sentiment and visual elements, to assess their effectiveness. It leverages a combination of natural language processing (NLP) techniques and computer vision models, along with traditional machine learning methods, to classify and analyze video ad data.

### Requirements:
The project requires several libraries to function properly:
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **scikit-learn**: For machine learning algorithms and metrics (e.g., RandomForestClassifier, GradientBoostingClassifier, precision, recall, f1_score).
- **transformers**: For NLP tasks (BERT tokenizer and model).
- **torch**: For handling BERT and deep learning tasks.
- **torchvision**: For image transformations and computer vision models.
- **opencv (cv2)**: For working with video files.
- **os**: For interacting with the file system.

### Workflow:

#### 1. **Data Loading**:
   - Load a CSV file (`Sample.csv`) containing textual data (likely metadata or descriptions related to the advertisements).
   - Define a function to retrieve video file paths from a specified directory.
   - Load the ground-truth CSV file (`ground-truth.csv`) to be used as labels for the classification tasks.

#### 2. **Data Preprocessing**:
   - Handle missing values by filling them with a default value (e.g., 'No').
   - Drop columns that contain any missing values.
   - Split the data into training and testing sets for machine learning models.
   - Use the BERT tokenizer and model for text encoding.

#### 3. **Model Training**:
   - The notebook leverages machine learning classifiers such as Random Forest and Gradient Boosting to perform classification tasks.
   - It uses GridSearchCV to tune hyperparameters and improve model performance.

#### 4. **Evaluation**:
   - Various classification metrics are calculated, including precision, recall, F1-score, and agreement percentage for each question or feature being evaluated.
   - For some questions related to the presence of "cute elements" like animals, babies, or animated characters, the notebook evaluates the agreement percentage and other metrics.

#### 5. **Modeling Visual Data**:
   - The notebook uses pretrained models from torchvision (possibly ResNet or a similar architecture) to process video data.
   - OpenCV (cv2) is used to handle video input and transformations.

### Usage:

1. **Setting Up the Environment**:
   - Install the necessary Python libraries:
     ```bash
     pip install pandas numpy scikit-learn transformers torch torchvision opencv-python
     ```

2. **Running the Notebook**:
   - Load the dataset files and video paths.
   - Run each cell sequentially to preprocess the data, train the models, and evaluate their performance.
   - Modify hyperparameters and file paths as needed based on your environment.

3. **Modifying for Other Data**:
   - To use the notebook with other datasets, adjust the file paths in the loading section and ensure the new dataset follows a similar format.

### Results:
The notebook prints precision, recall, F1-score, and agreement percentages for each feature or question it evaluates. These metrics help gauge the performance of the model and the agreement between predicted and actual labels.

### Conclusion:
This project provides an approach to combining textual and visual data analysis in video advertisements. By leveraging both machine learning models and deep learning techniques (e.g., BERT and computer vision models), it provides a robust framework for evaluating ad effectiveness.

