# Table Finder

A simple Python project that detects tables in document images (invoices, bank documents, etc.) using a pre-trained DETR model from Hugging Face.

This project is a learning exercise focused on:
- building a small backend-style class,
- running inference on images,
- writing basic unit tests with pytest.

---

## Features

- Detect tables in document images
- Supports common image formats (JPEG, PNG, etc.)
- Uses a pre-trained DETR model
- Draws bounding boxes around detected tables
- Simple and readable Python code
- Basic unit tests with pytest

---

## Project structure

```text
.
├── table_finder.py
├── test_table_finder.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Requirements

- Python 3.9 or newer
- PyTorch
- Transformers
- Pillow
- Matplotlib
- Pytest

All dependencies are listed in `requirements.txt`.

---

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/FlorianBlazi/EXO_BACKEND.git  
cd table-finder
```

---

### 2) Create a Python virtual environment (recommended)

macOS / Linux  
```bash
python3 -m venv .venv  
source .venv/bin/activate  
```

Windows (PowerShell)  
```bash
python -m venv .venv  
.\.venv\Scripts\Activate.ps1  
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Detect tables in an image

```python
from PIL import Image  
from table_finder import TableFinder  

finder = TableFinder()  

image = Image.open("invoice.jpg")  
results = finder.find_tables(image)  

print(results)
```

### Draw bounding boxes

```python
from PIL import Image  
from table_finder import TableFinder  

finder = TableFinder()  

image = Image.open("invoice.jpg")  
results = finder.find_tables(image)  

boxed_image = finder.identify_tables(image, results)  
boxed_image.save("annotated.png")  
boxed_image.show()
```

---

## Running tests

```bash
pytest -v
```

---

## Model

This project uses the pre-trained model:

- TahaDouaji/detr-doc-table-detection

---

## Author

Florian Blazi
