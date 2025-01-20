# Instructions to Run the Streamlit Dashboard

## Prerequisites:
1. Ensure Python 3.8+ is installed on your system.
2. Install required Python libraries using pip:
    pip install streamlit pandas seaborn matplotlib sqlite3

## Steps to Run the Streamlit Dashboard:
1. Unzip the provided folder `EAS503.zip` to any directory.
2. Open a terminal or command prompt.
3. Navigate to the folder:
    cd path/to/EAS503
4. Run the Streamlit app using the command:
    streamlit run iris_dashboard.py
5. The app will open automatically in your default browser at:
    http://localhost:8501

## Folder Contents:
- `iris_dashboard.py`: Streamlit app script.
- `iris_extended_db.sqlite`: SQLite database file.
- `iris_extended_cleaned.csv`: Cleaned version of the dataset.
- `iris_extended.csv`: Original dataset for reference.
- `readable IRIS-2.ipynb`: Jupyter Notebook with data cleaning and EDA steps.

## Notes:
- Use the sidebar filters in the dashboard to explore the data interactively.
- If any libraries are missing, install them using `pip`.

