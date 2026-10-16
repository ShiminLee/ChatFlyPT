# Overview ---------------------------------------------------------------

# This script sets the stage (library/func/data/conventions/etc) for plotting.
# It loads processed data. To re-process, see "proc_*"
# proc order, uCT -> eyemap -> T4, ephys, 

# "Fig_*" and "ED_fig_*" scripts carry out further analysis and produce figures.


# Notation and convention
# "## ##" = need to specify a choice, say, between T4b and T4d
# "Mollweide" and "Mercator" refer to different projections of the same plot

# Plotting and saving
# By default, the plot is shown in a plotting device, usually with "nopen()" or "windows()".
# To save to disk, look for commented commands:
# "ggsave()", "rgl.snapshot()", or "pdf()" paired with "dev.off()".


# load library ------------------------------------------------------------

library(natverse)
library(tidyverse)
library(RColorBrewer) #palette
library(readxl)
library(alphashape3d) # ashape3d
library(np) # kernel regression 
library(ggExtra) #ggMarginal
library(alphahull) # ashape
library(reshape2) #melt
library(cowplot)#ggdraw
library(plotrix)

# clean everythign up.
rm(list=ls())
#close any open rgl windows
while (rgl.cur() > 0) { close3d() }
# close opened dev
while (dev.cur() > 1) { dev.off() }

source("eyemap_func.R")

# FAFB CATMAID server ------------------------------------------------------------------

# neurons in this study can be found here
# https://fafb.catmaid.virtualflybrain.org/?pid=1&zp=65720&yp=160350.0517811483&xp=487737.6942783438&tool=tracingtool&sid0=1&s0=3.1999999999999993&help=true&layout=h(XY,%20%7B%20type:%20%22neuron-search%22,%20id:%20%22neuron-search-1%22,%20options:%20%7B%22annotation-name%22:%20%22Published%22%7D%7D,%200.6)
# under "Paper: Zhao et al 2023"

vfbcatmaid("fafb") # https://catmaid.virtualflybrain.org/
print(catmaid_login())
# expected output: 
# Connection to catmaid server:
# https://fafb.catmaid.virtualflybrain.org
# Login active since: Mon, 25 Nov 2024 21:08:52 GMT

# load uCT data ---------------------------------------------------------

load(paste0("data/microCT/20240701", ".RData"))
load(paste0("data/microCT/20240701", "_nb.RData"))
load(paste0("data/microCT/20240701", "_normals.RData"))
load(paste0("data/microCT/20240701", "_roc.RData"))
load(paste0("data/microCT/20240701", "_dia.RData"))

# separate left and right, re-indexing
lens_left <- lens[ind_left_lens,]
lens_2eye <- lens
ucl_rot_2eye <- ucl_rot_sm

lens <- lens[!ind_left_lens,]
ind_Up_lens <- na.omit(match(i_match[ind_Up], rownames(lens)))
ind_Up_lens_left <- na.omit(match(i_match[ind_Up], rownames(lens_left)))
ind_Down_lens <- na.omit(match(i_match[ind_Down], rownames(lens)))
ind_Down_lens_left <- na.omit(match(i_match[ind_Down], rownames(lens_left)))
cone_lens <- i_match[!ind_left_cone]

ucl_rot_left <- ucl_rot_2eye[order(i_match),][ind_left_lens,]
colnames(ucl_rot_left) <- c('x','y','z')
ucl_rot_right <- ucl_rot_2eye[order(i_match),][!ind_left_lens,]
colnames(ucl_rot_right) <- c('x','y','z')
rm(ucl)
i_match_uct <- i_match
rm(i_match)

# H2 ephys data -----------------------------------------------------------------

load('data/H2_tuning_2023.rda')
# tb_v = [theta(base), phi(base), spk, spk, sub, sub, dark=0/bright=1]

# load EM data ---------------------------------------------------------------

# [index, p, q] of lens/medulla points, see Fig.2C,D for [p q] coord
load('data/lens_ixy.RData')
load('data/med_ixy.RData')

# nb_ind: neighboring points' indices
# nb_dist_med: medulla neighbor column distance
# nb_dist_ucl: ommatidia directions neighbor dist
load('data/hexnb_ind_dist.RData') 

# load EM data ---------------------------------------------------------

# load EM neuron H2
load('data/neu_H2.RData')
# Alt. load from database
# anno_H2 <- catmaid_query_by_name("^H2 neuron \\(right", type = 'neuron')
# H2 <- read.neuron.catmaid(anno_H2$skid, .progress = 'text')
# save(anno_H2, H2, file = 'data/neu_H2.RData')

# PR
load('data/neu_PR.RData')

# L1
load('data/neu_L1.RData')

# R7
load('data/neu_R7.RData')

# Tm5a and LO xyz 
load('data/neu_TmY5a.RData') 
load('data/me_lo.RData') 

# Mi1
load("data/neu_Mi1.RData")
# index of special Mi1: _DRA, _PR, _noPR
load('data/Mi1_ind.RData') 

# T4
load('data/neu_T4b.RData')
load("data/neu_T4_dend.RData")
load('data/T4_hs.RData')

# T4 RF
load('data/T4_RF_pred.RData')

# T4 branching pattern
load('data/T4_gallery.RData')

# eyemap = [rownames of med_xyz, rownames of ucl_rot_sm]. 
# Note that all these 3 variables have the same row order/index.
# NB, load this after loading uCT data
load("data/eyemap.RData")

# chiasm
load('data/chiasm.RData')

# # LOP xyz vs ME xyz
# load('data/L2_med.RData') 

# LOP layers, xyz_layer_T4, from p_layerLOP
load("data/LayerData_T4_20210729_7e7_1um.RData")

# neuropil meshes
load("data/JFRC2NP.surf.fafb.rda") # JFRC2010 mesh, from Greg Jefferis
ME_msh <- as.mesh3d(JFRC2NP.surf.fafb, Regions="ME_R")
LO_msh <- as.mesh3d(JFRC2NP.surf.fafb, Regions="LO_R")
load('data/LOP_msh_mod.RData')


# special pts in medulla for alignment ------------------------
pt_100_Mi1 <- 10
pt_up_Mi1 <- 315

# color palette --------------------------------------------------------------

pal_9 <- brewer.pal(9,"Paired")
pal_lr <- c("light blue", "gray50") # left and right eye
pal_axes <- c('#227722', '#89DCB3','black', 'orange') # +p/q/v/h axes
pal_heat1 <- c("blue","#e7d4e8","red")
pal_heat2 <- rev(brewer.pal(8,"PuBu")[c(2,4,6,8)])
pal_so <- c('gray70', '#fdbb84', '#d7301f', 'gray35', 'gray35')
pal_T4 <- c("turquoise", "#A6611A",  "royalblue", "plum")
# [-tx -ty tz -rx ry -rz]
pal_TR <- c("#B48ADC", "#FFC125","#278384","#3D29A3","#A37B29","#66CD00")

# example hex in Fig.3 ------------------------------------------------------

# asp ratio, [DD D M]
ind_eghex <-  c(409, 223, 107)
# shear angle, [D M V]
ind_eghex_2 <- c(210, 348,  559)

# example T4 ------------------------------------------------------------

T4_ind_eg <- c(9880570, 11364711, 11301434, 11347884)

T4_eg <- neuronlist()
for (LL in 1:4) {
  skid <- sapply(T4_dend[[LL]], function(x) x$skid)
  T4_eg <- c(T4_eg, T4_dend[[LL]][na.omit(match(T4_ind_eg, skid))])
}

# example Mi1 -----------------------------------------------------------

Mi1_eg_skid <- c(14858811, 10888497, 10850334, 13726215, 15137473, 14567424, 13975332)
Mi1_eg_ind <- match(Mi1_eg_skid, anno_Mi1$skid)

# guidelines and ommatidia directions in Mollweide projections -----------------

# - ommatidia directions
ucl_rot_Mo <- ucl_rot_sm
colnames(ucl_rot_Mo) <- c('x','y','z')
ucl_rot_Mo %<>% as_tibble() %>%  
  mutate(y = -y) %>%
  mutate(theta = acos(z)) %>%
  mutate(phi = 2*pi*(y < 0) + (-1)^(y < 0)*acos(x/sin(theta))) %>%
  mutate(t = theta / pi * 180, p = phi/pi*180) %>%  
  mutate(p = if_else(p > 180, p - 360, p)) %>% #move to [-pi pi]
  as.data.frame()
ucl_rot_Mo <- Mollweide(ucl_rot_Mo[,c('t', 'p')])
colnames(ucl_rot_Mo) <- c('xM','yM')
rownames(ucl_rot_Mo) <- rownames(ucl_rot_sm)

# full left
ucl_rot_Mo_left <- ucl_rot_left
colnames(ucl_rot_Mo_left) <- c('x','y','z')
ucl_rot_Mo_left %<>% as_tibble() %>%  
  mutate(y = -y) %>%
  mutate(theta = acos(z)) %>%
  mutate(phi = 2*pi*(y < 0) + (-1)^(y < 0)*acos(x/sin(theta))) %>%
  mutate(t = theta / pi * 180, p = phi/pi*180) %>%  
  mutate(p = if_else(p > 180, p - 360, p)) %>% #move to [-pi pi]
  as.data.frame()
ucl_rot_Mo_left <- Mollweide(ucl_rot_Mo_left[,c('t', 'p')])
colnames(ucl_rot_Mo_left) <- c('xM','yM')
rownames(ucl_rot_Mo_left) <- rownames(ucl_rot_left)

# full right
ucl_rot_Mo_right <- ucl_rot_right
colnames(ucl_rot_Mo_right) <- c('x','y','z')
ucl_rot_Mo_right %<>% as_tibble() %>%  
  mutate(y = -y) %>%
  mutate(theta = acos(z)) %>%
  mutate(phi = 2*pi*(y < 0) + (-1)^(y < 0)*acos(x/sin(theta))) %>%
  mutate(t = theta / pi * 180, p = phi/pi*180) %>%  
  mutate(p = if_else(p > 180, p - 360, p)) %>% #move to [-pi pi]
  as.data.frame()
ucl_rot_Mo_right <- Mollweide(ucl_rot_Mo_right[,c('t', 'p')])
colnames(ucl_rot_Mo_right) <- c('xM','yM')
rownames(ucl_rot_Mo_right) <- rownames(ucl_rot_right)

# - Mollweide guidelines
Mollweide_ori <- c(0,0)
Mollweide_mul <- 1
bydeg <- 5

bkgd_eq <- Mollweide(cbind(90, seq(-180, 180, by = bydeg)))
bkgd_eq <- sweep(bkgd_eq*Mollweide_mul, 2, Mollweide_ori, '+')
colnames(bkgd_eq) <- c('xM','yM')
bkgd_eq_p45 <- Mollweide(cbind(45, seq(-180, 180, by = bydeg)))
bkgd_eq_p45 <- sweep(bkgd_eq_p45*Mollweide_mul, 2, Mollweide_ori, '+')
colnames(bkgd_eq_p45) <- c('xM','yM')
bkgd_eq_m45 <- Mollweide(cbind(135, seq(-180, 180, by = bydeg)))
bkgd_eq_m45 <- sweep(bkgd_eq_m45*Mollweide_mul, 2, Mollweide_ori, '+')
colnames(bkgd_eq_m45) <- c('xM','yM')
bkgd_mer_ww <- Mollweide(cbind(seq(0, 180, by = bydeg), rep(-180, 180/bydeg+1)))
bkgd_mer_w <- Mollweide(cbind(seq(180, 0, by = -bydeg), rep(-90, 180/bydeg+1)))
bkgd_mer_c <- Mollweide(cbind(seq(0, 180, by = bydeg), rep(0,180/bydeg+1)))
bkgd_mer_e <- Mollweide(cbind(seq(180, 0, by = -bydeg), rep(90,180/bydeg+1)))
bkgd_mer_ee <- Mollweide(cbind(seq(0, 180, by = bydeg), rep(180,180/bydeg+1)))
bkgd_mer <- rbind(bkgd_mer_ww, bkgd_mer_w, bkgd_mer_c,bkgd_mer_e,bkgd_mer_ee)
bkgd_mer <- sweep(bkgd_mer*Mollweide_mul, 2, Mollweide_ori, '+')
colnames(bkgd_mer) <- c('xM','yM')

# -- whole
plt_Mo <- ggplot() +
  geom_path(data= as.data.frame(bkgd_mer), aes(x=xM, y=yM), colour= 'grey50') +
  geom_path(data= as.data.frame(bkgd_eq_m45), aes(x=xM, y=yM), colour= 'grey50') +
  geom_path(data= as.data.frame(bkgd_eq_p45), aes(x=xM, y=yM), colour= 'grey50') +
  geom_path(data= as.data.frame(bkgd_eq), aes(x=xM, y=yM), colour = 'grey50') +
  scale_x_continuous(limits = c(-sqrt(8), sqrt(8)), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-sqrt(2), sqrt(2)), breaks = c(-1.5,0,1.5), labels = c(-1.5,0,1.5), expand = c(0, 0)) + # set +y as above eq
  theme_void() +
  theme(legend.position="none", panel.background = element_blank()) +
  coord_fixed(ratio = 1)

# -- 3/4
bkgd_eq <- Mollweide(cbind(90, seq(-90, 180, by = bydeg)))
colnames(bkgd_eq) <- c('xM','yM')
bkgd_eq_p45 <- Mollweide(cbind(45, seq(-90, 180, by = bydeg)))
colnames(bkgd_eq_p45) <- c('xM','yM')
bkgd_eq_m45 <- Mollweide(cbind(135, seq(-90, 180, by = bydeg)))
colnames(bkgd_eq_m45) <- c('xM','yM')
bkgd_mer <- rbind(bkgd_mer_w, bkgd_mer_c,bkgd_mer_e,bkgd_mer_ee)
colnames(bkgd_mer) <- c('xM','yM')

plt_Mo34 <- ggplot() +
  geom_path(data= as.data.frame(bkgd_mer), aes(x=xM, y=yM), colour = 'grey50') +
  geom_path(data= as.data.frame(bkgd_eq_m45), aes(x=xM, y=yM), colour = 'grey50') +
  geom_path(data= as.data.frame(bkgd_eq_p45), aes(x=xM, y=yM), colour = 'grey50') +
  geom_path(data= as.data.frame(bkgd_eq), aes(x=xM, y=yM), colour = 'grey50') +
  theme_void() +
  theme(legend.position="none", panel.background = element_blank()) +
  scale_x_continuous(limits = c(- 2*sqrt(2)/2, 2*sqrt(2)), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-sqrt(2), sqrt(2)), expand = c(0, 0)) +
  coord_fixed(ratio = 1)

# -- min
bkgd_eq <- Mollweide(cbind(90, seq(-30, 150, by = bydeg)))
colnames(bkgd_eq) <- c('xM','yM')
bkgd_mer <- rbind(bkgd_mer_c,bkgd_mer_e)
colnames(bkgd_mer) <- c('xM','yM')

plt_Momin <- ggplot() +
  geom_path(data = as.data.frame(bkgd_mer), aes(x=xM, y=yM), colour = 'grey50') +
  geom_path(data = as.data.frame(bkgd_eq), aes(x=xM, y=yM), colour = 'grey50') +
  theme_void() +
  theme(legend.position="none", panel.background = element_blank()) +
  scale_x_continuous(limits = c(- 2*sqrt(2)/2, 2*sqrt(2)), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-sqrt(2), sqrt(2)), expand = c(0, 0)) +
  coord_fixed(ratio = 1)

# equator and meridian strips -----------------------------------------------

str_hw <- 15
bkgd_str_equa <- Mollweide(
  rbind(
    cbind(rep(90- str_hw, 10), seq(-45, 160, length.out=10)),
    cbind(seq(90- str_hw, 90+ str_hw, length.out=5), rep(160, 5)),
    cbind(rep(90+ str_hw, 10), seq(160, -45, length.out=10)),
    cbind(seq(90+ str_hw, 90- str_hw, length.out=5), rep(-45, 5)) ) )
colnames(bkgd_str_equa) <- c('xM','yM')
bkgd_str_meri <- Mollweide(
  rbind(
    cbind(seq(0, 180, length.out=20), rep(45- str_hw,20)),
    cbind(seq(180, 0, length.out=20), rep(45+ str_hw,20)) ) )
colnames(bkgd_str_meri) <- c('xM','yM')

# guidelines in Mercator projection -------------------------------------------

ucl_rot_Merc_right <- cart2Mercator(ucl_rot_right) %>% as.data.frame()
rownames(ucl_rot_Merc_right) <- rownames(ucl_rot_right)

ucl_rot_Merc <- cart2Mercator(ucl_rot_sm) %>% as.data.frame()
rownames(ucl_rot_Merc) <- rownames(ucl_rot_sm)

xlat <- seq(-45,180,by=45)
xtick <- xlat/180*pi
ylat <- c(seq(-75,75,by=15), 85)
ytick <- log(tan(pi/4 + ylat/180*pi/2))

plt_Mer <- ggplot() +
  theme_bw() +
  scale_x_continuous(limits = range(xtick), breaks = xtick, labels = xlat, expand = c(0, 0)) +
  scale_y_continuous(limits = range(ytick), breaks = ytick, labels= ylat, expand = c(0, 0)) + 
  theme(axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        panel.grid.minor.x = element_blank(), 
        panel.grid.minor.y = element_blank(), )+
  coord_fixed(ratio = 1)

# central meridian --------------------------------------------------------

# choose central lines along p-/q-/v-/h-axes
clv <- 0
clh <- 0
clp <- 0
clq <- 0

# indices of points along these lines
ind_axis <- vaxis_gen(0, ixy = lens_ixy)
xy <- data.frame(ucl_rot_Mo_right)[ind_axis,] # full right
cmer_right <- xy[order(xy$yM),]

ind_axis <- vaxis_gen(0, ixy = ind_xy)
xy <- data.frame(ucl_rot_Mo)[ind_axis,] #matched right
cmer <- xy[order(xy$yM),]

xy <- data.frame(ucl_rot_Merc)[ind_axis,]
cmer_Merc <- xy[order(xy$y),]

ind_axis <- vaxis_gen(7, ixy = lens_ixy)
xy <- data.frame(ucl_rot_Mo)[ind_axis,]
mer10_right <- xy[order(xy$yM),]

ind_axis <- vaxis_gen(9, ixy = ind_xy)
xy <- data.frame(ucl_rot_Mo)[ind_axis,]
mer10 <- xy[order(xy$yM),]

