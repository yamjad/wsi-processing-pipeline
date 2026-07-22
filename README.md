# WSI Processing Pipeline

### Overview

<p>A pipeline for WSI processing for use in image classification tasks. WSI data was obtained from the CPTAC database [here](https://gdc.cancer.gov/about-gdc/contributed-genomic-data-cancer-research/clinical-proteomic-tumor-analysis-consortium-cptac). Functions and workflows are adapted from the Panoptes library which can be found [here](https://github.com/rhong3/Panoptes/tree/master).</p>

### Usage

#### download_cptac_slides.py

<p>Downloads the images from the CPTAC database based on patient IDs.</p>

<p>(1) Create a folder containing download_cptac_slides.py and cptac_slide_index.csv.<br>
(2) Edit the download_cptac_slides.py SAMPLE_IDS to desired patient IDs (twenty patients from LUAD and Uterine datasets selcted).<br>
(3) Install pandas (pip install pandas) and idc client (pip install idc-index).<br>
(4) Run via command: python download_cptac_slides.py</p>

Use optional flags: 
    <p>--csv PATH        Path to the slide index CSV (default: cptac_slide_index.csv)<br>
       --outdir PATH      Directory to download slides into (default: ./idc_downloads)<br>
       --dry-run          Print the commands that would run, but don't execute them</p>
