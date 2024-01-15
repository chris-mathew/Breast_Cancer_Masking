# Breast Cancer Screening Project

This project aims to develop an AI-based breast cancer screening system. It includes components for breast density estimation, BIRADS classification, mammogram analysis, and breast cancer risk prediction.

## Folder Structure

- `__pycache__`: Caches created by Python.
- `Cleaning up code`: Folder related to code cleanup.
- `preprocessing`: Scripts for data preprocessing.
- `sql_database`: SQL database related files.
- `userinterface`: User interface components.
- `utility_scripts`: Scripts for utility functions.
- `breastdensitymodel.py`: Python file for breast density model.
- `dataset_DDSM.py`: Python file for handling the DDSM dataset.
- `segmentationandriskmodel.py`: Python file for segmentation and risk prediction.

## Recent Updates

- **Cleaning up code**: Updated code for a cleaner structure.
- **preprocessing**: Updated `RollingBallAlgorithm.py`.

## How to Contribute

If you'd like to contribute to this project, please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

## Dependencies

- Python 3.6 or higher
- Streamlit 0.79.0
- Pandas 1.2.3
- NumPy 1.18.5
- Matplotlib 3.3.2
- Seaborn 0.11.0
- scikit-learn 0.24.1

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/breast-cancer-screening.git
    cd breast-cancer-screening
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```bash
    streamlit run userinterface/app.py
    ```

## Usage

1. Upload mammogram images using the drag-and-drop tile in the user interface.
2. View breast density measurements, BIRADS classification, and estimated cancer risk.
3. Explore the BIRADS classifications explained section for detailed information.

## Future Improvements

- Implement a preprocessing pipeline for improved model accuracy.
- Expand the dataset to enhance model training.
- ...

## License

This project is licensed under the [MIT License](LICENSE).



# AI based breast cancer investigator

To Do Week of 6th Dec: 

- Continue expanding the dataset 
- Debugging of the model and begin to visualise results
- Begin updating model to include asymettry
- Begin literature review around cancer risk


Note: the model also exist here in a colab
https://colab.research.google.com/drive/1UMmGNRCQdcRE-7k9PDzjhRmf2-S9IKoT#scrollTo=Wd27kp_ozh65



