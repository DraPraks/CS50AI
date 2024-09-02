# PageRank Algorithm Implementation

This project is a Python script that implements the PageRank algorithm to rank web pages based on their interlinking structure. It provides both a sampling method and an iterative method to compute the PageRank of each page.

## Features

- **Crawling HTML Pages**: Parse a directory of HTML pages to build a link structure.
- **PageRank Calculation**: Compute PageRank using both sampling and iterative methods.
- **Transition Model**: Model the probability distribution for transitioning between pages.

## Requirements

- Python 3.x

## Usage

1. Prepare a directory containing HTML files. Each file should represent a webpage with links to other pages.

2. Run the script:

   ```bash
   python pagerank.py corpus_directory
