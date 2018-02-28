# Project Report in LaTeX

The project typesetting is done in XeLaTeX.

# Output File
Note that the final file will be in the build folder named: **_report.pdf_**

# Requirements
* Up to date TeX distribution like TexLive

# How to Compile
Run the following command in a terminal:

`xelatex -synctex=1 -shell-escape --output-directory=build -interaction=nonstopmode  report.tex`
