# VizWhiz - A Visualization Wizard

**VizWhiz** is a powerful and interactive Streamlit-based web application for general-purpose data exploration and visualization. Upload any CSV file and use the intuitive interface to clean your data, explore its structure, and generate a wide variety of customizable plots.

## Features

- Upload any CSV file
- Data Cleaning:
  - Drop or fill missing values (mean, median, or mode)
  - Remove outliers (customizable IQR threshold)
  - Deduplicate rows
- Visualizations:
  - Histograms (with optional KDE)
  - Box plots
  - Violin plots
  - Scatter plots (with regression lines)
  - Line plots
- Customization Options:
  - Color selection for each group
  - Grouping by categorical variables
  - Log scaling, point opacity, and layout adjustments

## Installation

To run VizWhiz locally, make sure you have Python 3.8+ installed. Then follow these steps:

```bash
# Clone the repository
git clone https://github.com/your-username/vizwhiz.git
cd vizwhiz

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install the required packages mentioned below

# Launch the app
streamlit run app.py
```

> Make sure `app.py` is the name of your main Python script. Adjust if different.

## Requirements

All dependencies include:

- streamlit
- pandas
- plotly
- numpy
- scipy
- seaborn
- matplotlib

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

Please ensure your code follows best practices and includes appropriate documentation or comments.