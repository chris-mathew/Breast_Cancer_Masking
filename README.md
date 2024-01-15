# Breast Cancer Screening Project

This project aims to develop an AI-based breast cancer screening system. It includes components for pre-processing, SQL server, APIs, machine learning model trained on DDOS publically available dataset, user interface, and deployment to create a successful Breast Cancer Desnity and Breast Cancer Risk Assessment program.

Link to demo: https://www.crtl-alt-elite.online/

## Folder Structure

- `__pycache__`: Caches created by Python.
- `Cleaning up code`: Folder related to code cleanup.
- `preprocessing`: Scripts for data preprocessing.
- `sql_database`: SQL database related files.
- `userinterface`: User interface components.
- `utility_scripts`: Scripts for utility functions.
- `breastdensitymodel.py`: First Python file for first prototype of breast density model.
- `dataset_DDSM.py`: Python file for handling the DDSM dataset.
- `segmentationandriskmodel.py`: Final Python file for final version of segmentation and risk prediction.

## Recent Updates

- **User Interface**: Updated UI and deployed on heroku
- **Preprocessing**: Finalised `RollingBallAlgorithm.py`


## Dependencies

- Python 3.6 or higher
- pandas 2.1.4
- Pillow 10.2.0
- streamlit 1.30.0
- NumPy 1.18.5
- Matplotlib 3.3.2
- Seaborn 0.11.0
- scikit-learn 0.24.1
- skimage 0.22.0
- torch 2.1.2
- pyodbc

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/breast-cancer-screening.git
    cd breast-cancer-screening
    ```

2. Install dependencies:
   Individually install all missing dependencies detailed above

3. Install the DDSM dataset, as shown in `dataset_DDSM.py`

4. Run the `preprocessing` on the dataset

5. Train the model in `segmentationandriskmodel.py` with the processed data 
   
6. Export the model via pickle or torch
   
7. Run the User interface code
   This is in the main folder due to needing to store it in the root folder so it worked effectively with heroku.

   For the user interface, you need to run `app.py`

   If you want to edit the application and re-run it, use the code below:
   
    ```bash
    streamlit run app.py
    ```

9. Deploy the streamlit app to Heroku

   This is a useful link that walks step by step through deployment: https://gilberttanner.com/blog/deploying-your-streamlit-dashboard-with-heroku/

   For this you can refer to `requirements.txt`, `setup.sh`, and `Procfile`


## Usage

1. Upload mammogram images using the drag-and-drop tile in the user interface linked at https://www.crtl-alt-elite.online/
2. View breast density measurement and estimated cancer risk.

## Future Improvements

- Factor in breast assymetry.
- Expand the dataset to enhance model training.
- Continous model training
- Integrating clinical data with data protection and security as a priority.


