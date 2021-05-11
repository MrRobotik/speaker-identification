all: doc.ps
	ps2pdf doc.ps doc.pdf

doc.ps: doc.dvi
	dvips doc

doc.dvi: doc.tex literature.bib
	latex doc
	-bibtex doc
	latex doc
	latex doc

clean:
	rm *.aux *.log *.blg *.bbl *.ps *.dvi doc.pdf
