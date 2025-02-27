# -*- coding: utf-8 -*-
"""
utils.pdf
---------

Module containing methods to create PDF exports
of generated images and CSV tables.
"""
# Imports
# Built-in
import os
import csv
from PIL import Image

# Local

# 3r-party
from PyPDF2 import PdfMerger
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter


def images_to_pdf(pdf_filename: str, image_files: list) -> None:
    """
    Convert images to a PDF document.

    Args:
        pdf_filename (str): File path of the output PDF.
        image_files (list): List of file paths for images to be included
            in the PDF.

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

        # Add title to the page
        c.setFont("Helvetica", 16)
        c.drawString(
            20,
            pdf_height - 40,
            image_file.split('/')[-1].split('.')[0]
        )
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


def create_pdf_with_tables(pdf_filename: str, csv_files: list) -> None:
    """
    Create a PDF file containing tables generated from multiple CSV files.

    Args:
        pdf_filename (str): File path for the generated PDF.
        csv_files (list): List of file paths for CSV files.

    Returns:
        None: Saves the PDF file with the tables.
    """
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []  # Initialize an empty list to store table elements

    for csv_file in csv_files:
        with open(csv_file, 'r', newline='') as file:
            csv_data = list(csv.reader(file))
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

            table.setStyle(style)
            # Add title as a Paragraph above each table
            styles = getSampleStyleSheet()
            title_paragraph = Paragraph(
                f"<b>{csv_file.split('/')[-1].split('.')[0]}</b>",
                styles['Heading1']
            )
            elements.append(title_paragraph)
            elements.append(table)  # Append the table to the elements list

    doc.build(elements)  # Build the PDF document containing all tables


def create_merged_pdf(output_dir: str, pdf_name: str) -> None:
    """
    Generate a PDF containing images and a table from CSV data and merge them
    into a single PDF.

    Args:
        output_dir (str): Directory path where the data files are stored.

    Returns:
        None: Saves the merged PDF file.
    """
    # List all files in the directory
    data_exploration_files = os.listdir(output_dir)
    data_exploration_files = [
        os.path.join(output_dir, file)
        for file in data_exploration_files
    ]

    # Filter image files (JPG) and find the CSV file
    image_files = [
        file
        for file in data_exploration_files
        if file.lower().endswith(('.jpg', '.png'))
    ]
    csv_files = [
        file
        for file in data_exploration_files
        if file.lower().endswith('.csv')
    ]
    images_exists = False
    if len(image_files) > 0:
        images_exists = True
    csv_exists = False
    if len(csv_files) > 0:
        csv_exists = True

    if not (images_exists or csv_exists):
        raise Exception(
            f'No images or csv files found in directory {output_dir}'
        )

    # Create a PDF from images
    if images_exists:
        pdf_from_images = os.path.join(output_dir, 'graphs.pdf')
        images_to_pdf(pdf_from_images, image_files)

    if csv_exists:
        # Create a PDF file with a table from CSV data
        pdf_from_csv = os.path.join(output_dir, 'csv.pdf')
        create_pdf_with_tables(pdf_from_csv, csv_files)

    # Merge PDFs into a single file
    pdf_merger = PdfMerger()
    if images_exists:
        pdf_merger.append(pdf_from_images)
    if csv_exists:
        pdf_merger.append(pdf_from_csv)
    pdf_merged = os.path.join(output_dir, f'{pdf_name}.pdf')
    pdf_merger.write(pdf_merged)
    pdf_merger.close()
