# WSI Processing Pipeline

### Overview

A pipeline for WSI processing for use in image classification tasks. WSI data was obtained from the [CPTAC database](https://gdc.cancer.gov/about-gdc/contributed-genomic-data-cancer-research/clinical-proteomic-tumor-analysis-consortium-cptac). Functions and workflows are adapted from the Panoptes library which can be found [here](https://github.com/rhong3/Panoptes/tree/master).

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

Results should be stored at in the following path structure, with the last two values being examples: /[OUTDIR PATH]/[cancer_type]/[patient_id]/2.25.48791557373299768401597362411459861639/SM_1.3.6.1.4.1.5962.99.1.132039251.338821108.1640809579091.2.0
