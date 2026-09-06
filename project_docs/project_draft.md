# End-to-End Recommender System

## Project Overview
The **End-to-End Recommender System** is a machine learning-based application designed to recommend books to users based on collaborative filtering. The project encompasses a complete MLOps pipeline, including data ingestion, validation, transformation, model training, and a web-based user interface for interaction.

## Project Structure
The project is structured as a Python package `books_recommender` which contains the core logic, separated into various components to ensure modularity and maintainability.

### Key Directories and Files
- **`books_recommender/`**: The main package containing source code.
    - **`components/`**: Handles the stages of the ML pipeline:
        - `stage_00_data_ingestion.py`: Handles downloading and extracting data.
        - `stage_01_data_validation.py`: Validates the integrity of the data.
        - `stage_02_data_transformation.py`: Transforms data for the model.
        - `stage_03_model_trainer.py`: Trains the recommendation model.
    - **`pipeline/`**: Orchestrates the training process via `training_pipeline.py`.
    - **`config/`**, **`entity/`**, **`constant/`**: Configuration and data definitions.
    - **`logger/`**, **`exception/`**: Logging and error handling utilities.
- **`app.py`**: A Streamlit web application that serves as the user interface for getting recommendations and triggering training.
- **`main.py`**: Entry point to trigger the training pipeline programmatically.
- **`setup.py`**: Script for installing the project as a package.
- **`Dockerfile`**: Defines the container environment for deployment.
- **`requirements.txt`**: Lists python dependencies.

## Architecture & Workflow
The system follows a typical Machine Learning pipeline architecture:

1.  **Data Ingestion**: Raw data is acquired and prepared for processing.
2.  **Data Validation**: Data is checked against schema expectations to ensure quality.
3.  **Data Transformation**: Data is cleaned and transformed (e.g., creating pivot tables) suitable for the model.
4.  **Model Training**: A Nearest Neighbors model (based on `app.py` inspection) is trained on the processed data.
5.  **Inference**: The `app.py` uses the trained artifacts (model, pivot tables) to generate recommendations.

## Tech Stack
- **Language**: Python 3.7+
- **Web Framework**: Streamlit
- **Machine Learning**: Scikit-Learn (NearestNeighbors)
- **Data Processing**: Pandas, NumPy
- **Containerization**: Docker
- **MLOps**: Modular pipeline design, Logging, Exception Handling

## Features
- **Book Recommendations**: Users can select a book and receive recommendations for similar books.
- **Visual Interface**: Displays book covers and titles for recommendations using Streamlit.
- **Training Pipeline**: Capability to re-train the model via the UI or command line.
- **Dockerized**: Ready for deployment on cloud platforms like AWS.

