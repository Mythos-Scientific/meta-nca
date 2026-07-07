# Makefile for the Architecture Scaling Ablation slide deck
#
#   make          build main.pdf
#   make watch    rebuild continuously on save (latexmk -pvc)
#   make view     build, then open the PDF
#   make clean    remove LaTeX aux files (keep the PDF)
#   make distclean remove aux files and the PDF

MAIN    := main
PDF     := $(MAIN).pdf
FIGDIR  := ../figures

LATEXMK := latexmk
LMKOPTS := -pdf -interaction=nonstopmode -halt-on-error

# Rebuild when the source or any figure changes.
FIGURES := $(wildcard $(FIGDIR)/*.pdf $(FIGDIR)/scaling/*.pdf)

.PHONY: all watch view clean distclean

all: $(PDF)

$(PDF): $(MAIN).tex $(FIGURES)
	$(LATEXMK) $(LMKOPTS) $(MAIN).tex

watch:
	$(LATEXMK) $(LMKOPTS) -pvc $(MAIN).tex

view: $(PDF)
	open $(PDF)

clean:
	$(LATEXMK) -c $(MAIN).tex
	rm -f $(MAIN).nav $(MAIN).snm $(MAIN).vrb build.log

distclean: clean
	$(LATEXMK) -C $(MAIN).tex
	rm -f $(PDF)
