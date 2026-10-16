# eyemap_T4
R code for "Eye structure shapes neuron function in Drosophila motion vision"  
Tested with R 4.4.1 and Rstudio 2024.12.1.

R packages are versioned using renv:
https://rstudio.github.io/renv/articles/renv.html  
Suggest start with copying this repo and run `renv::restore()`  
Estimated installation time: 30 min

To reproduce plots in the paper, start with `Fig_0.R` and follow the instruction at the beginning.  
`Fig_*.R` and `ED_fig_*.R` mostly contain code that plots specific figure panel in the paper (search for, eg., "Fig.1C" or "ED Fig.1A"). Extended figures are associated with main figures as follows:
- Fig.1 + ED Fig.1
- Fig.2 + ED Fig.2 + ED Fig.3
- Fig.3 + ED Fig.4 + ED Fig.5
- Fig.4 + ED Fig.6 + ED Fig.7

Note that the plotting code for some extended figures are included in the main figure scripts (check opening comments).  
Supplementary T4 neuron gallery is made by `supp_gallery.R`.  
Most analysis code are in `proc_*.R` files.
