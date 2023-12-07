# Standard library imports
import argparse
import math
from typing import Union, Dict

# Third-party library imports
import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from ydata_profiling import ProfileReport
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from PIL import Image
import csv
from PyPDF2 import PdfMerger

# Local or intra-package imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import load_dataset

def images_to_pdf(pdf_filename: str, image_files: list) -> None:
    """
    Convert images to a PDF document.

    Args:
        pdf_filename (str): File path of the output PDF.
        image_files (list): List of file paths for images to be included in the PDF.

    Returns:
        None
    """
    # Create a canvas for the PDF
    c = canvas.Canvas(pdf_filename, pagesize=letter)

    for image_file in image_files:
        img = Image.open(image_file)
        img_width, img_height = img.size

        # Set the image size to fit the PDF page
        pdf_width, pdf_height = letter
        width_ratio = pdf_width / float(img_width)
        height_ratio = pdf_height / float(img_height)
        ratio = min(width_ratio, height_ratio)
        img_width *= ratio
        img_height *= ratio

        # Draw the image onto the PDF canvas
        c.drawImage(image_file, 0, 0, width=img_width, height=img_height)
        c.showPage()

    # Save the PDF
    c.save()

def read_csv(csv_filename: str) -> list:
    """
    Read data from a CSV file and return it as a list of rows.

    Args:
        csv_filename (str): File path of the CSV file to read.

    Returns:
        list: Data read from the CSV file as a list of rows.
    """
    # Open the CSV file and read its contents using csv.reader
    with open(csv_filename, 'r') as file:
        csv_reader = csv.reader(file)
        data = [row for row in csv_reader]
    
    return data


def create_pdf_with_table(pdf_filename: str, csv_data: list) -> None:
    """
    Create a PDF file containing a table generated from CSV data.

    Args:
        pdf_filename (str): File path for the generated PDF.
        csv_data (list): Data to be converted into a table in the PDF.

    Returns:
        None: Saves the PDF file with the table.
    """
    # Initialize a SimpleDocTemplate object with the specified filename and page size
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)

    # Create a table from CSV data
    table = Table(csv_data)

    # Define table style settings
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), 'grey'),
        ('TEXTCOLOR', (0, 0), (-1, 0), 'white'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), 'lightgrey'),
    ])

    # Apply the defined style to the table
    table.setStyle(style)

    # Build the PDF document containing the table
    elements = [table]
    doc.build(elements)

def create_data_profile_pdf(output_dir: str) -> None:
    """
    Generate a PDF containing images and a table from CSV data and merge them into a single PDF.

    Args:
        output_dir (str): Directory path where the data files are stored.

    Returns:
        None: Saves the merged PDF file.
    """
    # List all files in the directory
    data_exploration_files = os.listdir(output_dir)
    data_exploration_files = [os.path.join(output_dir, file) for file in data_exploration_files]

    # Filter image files (JPG) and find the CSV file
    image_files = [file for file in data_exploration_files if file.lower().endswith('.jpg')]
    csv_filename = [file for file in data_exploration_files if file.lower().endswith('.csv')][0]

    # Create a PDF from images
    pdf_from_images = os.path.join(output_dir, 'graphs.pdf')
    images_to_pdf(pdf_from_images, image_files)

    # Read CSV file
    csv_data = read_csv(csv_filename)

    # Create a PDF file with a table from CSV data
    pdf_from_csv = os.path.join(output_dir, 'image_distribution_table.pdf')
    create_pdf_with_table(pdf_from_csv, csv_data)

    # Merge PDFs into a single file
    pdf_merger = PdfMerger()
    pdf_merger.append(pdf_from_images)
    pdf_merger.append(pdf_from_csv)
    pdf_merged = os.path.join(output_dir, 'image_distribution.pdf')
    pdf_merger.write(pdf_merged)
    pdf_merger.close()

def line_graph(image_distribution_path: str, output_dir: str, logarithmic: bool) -> None:
    """
    Generate and save a line graph based on image distribution data.

    Args:
        image_distribution_path (str): File path of the image distribution data (CSV format).
        output_dir (str): Directory path where the graph will be saved.
        logarithmic (bool): Flag indicating whether to use logarithmic scale.

    Returns:
        None: Saves the line graph at the specified output directory.
    """
    # Read the image distribution data
    image_distribution = pd.read_csv(image_distribution_path)

    # Create and customize the line graph
    plt.figure(figsize=(8, 6))
    if logarithmic:
        plt.plot(image_distribution['label'], image_distribution['count'].apply(log), linestyle='-', color='b')
        graph_name = 'line_graph_log.jpg'
        ylabel = 'Log Total Images'
        title = 'Image Distribution (Logarithmic)'
    else:
        plt.plot(image_distribution['label'], image_distribution['count'], linestyle='-', color='b')
        graph_name = 'line_graph.jpg'
        ylabel = 'Total Images'
        title = 'Image Distribution'

    # Label axes and set the title
    plt.xlabel('Country')
    plt.ylabel(ylabel)
    plt.title(title)

    # Hide x-axis tick labels
    empty_labels = [''] * len(image_distribution['label'])
    plt.xticks(range(len(image_distribution['label'])), empty_labels)

    # Display and save the plot
    plt.grid(axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, graph_name))
    plt.close()

def log(x: Union[int, float]) -> Union[int, float]:
    """
    Calculates the logarithm of a given value and returns it (minimum 1)

    Args:
        x (Union[int, float]): Value that should be logarithmized

    Returns:
        Union[int, float]: Logarithmic value (minimum 1)
    """
    return max(1, math.log(x))

def world_heat_map(image_distribution_path: str, output_dir: str, logarithmic: bool) -> None:    
    """
    Generates a world heat map based on image distribution data.

    Args:
        image_distribution_path (str): File path of the image distribution data (CSV format).
        output_dir (str): Directory path where the plot will be saved.
        logarithmic (bool): Determines whether to apply logarithm to the data.

    Returns:
        None: Saves the world heat map plot at the specified output directory.
    """
    # Reading the world map data 
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    # Reading image distribution data from CSV
    image_distribution = pd.read_csv(image_distribution_path)

    # Applying logarithm to the 'count' column in image distribution
    if logarithmic:
        image_distribution['count'] = image_distribution['count'].apply(log)
        graph_name = 'world_heat_map_log.jpg'
        title = 'Logarithmic World Heat Map'
    else:
        graph_name = 'world_heat_map.jpg'
        title = 'World Heat Map'
        
    # Merging world and image distribution dataframes
    world['join'] = 1
    image_distribution['join'] = 1
    data_frame_full = world.merge(image_distribution, on='join').drop('join', axis=1)
    image_distribution.drop('join', axis=1, inplace=True)

    # Checking for matches between 'name' and 'label' columns
    data_frame_full['match'] = data_frame_full.apply(lambda x: x["name"].find(x["label"]), axis=1).ge(0)

    # Filtering the dataframe based on matches and plotting the world map
    df = data_frame_full[data_frame_full['match']]
    df.plot(column='count', legend=True)

    # Saving the world heat map at the specified output path
    plt.title(title)
    plt.savefig(os.path.join(output_dir, graph_name))
    plt.close()

def data_profile(dataset_dir: str, REPO_PATH: str, dataset_name: str) -> None:
    """
    Generates a profile report, image distribution CSV, world heat map, and line graph based on a dataset.

    Args:
        dataset_dir (str): File path of the dataset.
        repo_path (str): Root path of the repository.
        dataset_name (str): Name of the dataset.

    Returns:
        None
    """
    output_dir = os.path.join(REPO_PATH, 'data_exploration', dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loading dataset and creating a profile report
    df = load_dataset.load_data(DATA_PATH=dataset_dir, size_constraints=False)
    profile = ProfileReport(df, title=f'{dataset_name} Profile Report')
    profile.to_file(os.path.join(output_dir, 'profile.html'))

    # Generating image distribution and saving as CSV
    image_distribution = df['label'].value_counts()
    image_distribution.to_csv(os.path.join(output_dir, 'image_distribution.csv'))

    # Generating and saving world heat map and line graph based on image distribution (default and logarithmic) and save it in one pdf
    world_heat_map(os.path.join(output_dir, 'image_distribution.csv'), output_dir=output_dir, logarithmic=True)
    world_heat_map(os.path.join(output_dir, 'image_distribution.csv'), output_dir=output_dir, logarithmic=False)
    line_graph(os.path.join(output_dir, 'image_distribution.csv'), output_dir=output_dir, logarithmic=True)
    line_graph(os.path.join(output_dir, 'image_distribution.csv'), output_dir=output_dir, logarithmic=False)
    create_data_profile_pdf(output_dir)

def create_dataset_profile(user: str, yaml_path: str, dataset_dir: str, dataset_name: str) -> None:
    """
    Create a dataset profile including a report, image distribution CSV, and world heat map.

    Args:
        user (str): The user of the gpml group.
        yaml_path (str): The path to the YAML file with the stored paths.
        dataset_dir (str): The path to directory with the dataset.
        dataset_name (str): The name of the dataset.

    Returns:
        None
    """
    with open(yaml_path) as file:
        paths: Dict[str, Dict[str, str]] = yaml.safe_load(file)
        repo_path = paths['repo_path'][user]
        data_profile(dataset_dir, repo_path, dataset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Dataset Profile')
    parser.add_argument('--user', metavar='str', required=True, help='the user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='the path to the yaml file with the stored paths')
    parser.add_argument('--dataset_dir', metavar='str', required=True, help='the path to directory with the dataset')
    parser.add_argument('--dataset_name', metavar='str', required=True, help='the name of the dataset')
    args = parser.parse_args()
    create_dataset_profile(args.user, args.yaml_path, args.dataset_dir, args.dataset_name)
