# ESG Data Quality Dashboard

A powerful tool to check and improve the quality of environmental (ESG) data for real estate buildings.

![Image](https://github.com/user-attachments/assets/85cce78d-2e13-49e3-bfff-c5534cb15f2c)
![Image](https://github.com/user-attachments/assets/c8e437d8-6d40-449a-81fc-c443717c2338)


## What This Project Does

This dashboard helps you:

- See how complete and accurate your building's environmental data is
- Find missing information and errors in your data
- Get suggestions to improve your data quality
- Track improvements over time

## Features

- **Easy-to-read quality scores** for energy, water, CO2, and waste data
- **Visual charts** showing data completeness and problems
- **Building comparison** to see which buildings need attention
- **Automatic recommendations** to fix data issues
- **Clean, modern design** that's easy to use

## Getting Started

### What You Need

- Python 3.8 or newer
- Basic knowledge of running commands in a terminal

### Setup Steps

1. **Download the project**:
   ```
   git clone https://github.com/yourusername/esg-data-quality-dashboard.git
   cd esg-data-quality-dashboard
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   ```

3. **Activate the environment**:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. **Install required packages**:
   ```
   pip install -r requirements.txt
   ```

5. **Run the dashboard**:
   ```
   streamlit run app.py
   ```

6. **Open in your browser**: The dashboard will open automatically or visit http://localhost:8501

## How to Use

1. **Navigation**: Use the sidebar to move between different pages:
   - Overview: Quick summary of all data quality
   - Data Quality Analysis: Detailed view for each category
   - Trends: See how data quality changes over time
   - Recommendations: Get advice to improve data quality

2. **Filters**: Narrow down your analysis by:
   - Date Range
   - Building Type
   - Data Source
   - Location

3. **Take Action**: Use the recommendations page to:
   - Identify the most important issues
   - Generate action plans
   - Track progress on improvements

## Sample Data

The dashboard comes with sample data so you can test it right away. To use your own data:

1. Replace the CSV files in the `data` folder with your own
2. Make sure your files follow the same format (see the sample files)

## Need Help?

If you have questions or run into problems:

- Check the user guide in the `docs` folder
- Create an issue on GitHub
- Contact: sameerm1421999@gmail.com

## Created By

Sameer M - Data Analyst specializing in ESG data quality
