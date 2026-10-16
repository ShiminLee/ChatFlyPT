# map lens to medulla to LOP 
# eyemap = [Mi1 rownames, ucl_rot rownames]

library(natverse)
library(tidyverse)
library(RColorBrewer)
library(np) # kernel regression to find the base pt on the eye
library(alphahull)
library(deldir)# for smoothing uCT data
library(sp) #in.poly

# clean everythign up.
rm(list=ls())
#close any open rgl windows
while (rgl.cur() > 0) { close3d() }
while (dev.cur() > 1) { dev.off() }

source("eyemap_func.R")


# load data ---------------------------------------------------------------

# uCT data
load(paste0("data/microCT/", '20240701', '.RData'))

lens_left <- lens[ind_left_lens,]
lens <- lens[!ind_left_lens,] # match lens to EM
ind_Up_lens <- na.omit(match(i_match[ind_Up], rownames(lens)))
ind_Down_lens <- na.omit(match(i_match[ind_Down], rownames(lens)))
cone_lens <- i_match[!ind_left_cone]

# re-order ucl_rot rows st same as lens and re-define rownames because mapping is betw Mi1 and lens
ucl_rot_sm0 <- ucl_rot_sm[order(i_match),][!ind_left_lens,] # same order as lens now
ucl_rot_sm0_left <- ucl_rot_sm[order(i_match),][ind_left_lens,] # same order as lens now
rownames(ucl_rot_sm0) <- seq(1, nrow(ucl_rot_sm0))
rownames(ucl_rot_sm0_left) <- seq(1, nrow(ucl_rot_sm0_left))
colnames(ucl_rot_sm0) <- c('x','y','z')
colnames(ucl_rot_sm0_left) <- c('x','y','z')
rm(ucl_rot_sm)
rm(ucl)
rm(i_match)

# Mi1
load("data/neu_Mi1.RData")

# chiasm
load('data/chiasm.RData')

# project onto a sphere  -----------------------------------------------------

# - eye lens from microCT
xyz_data <- lens
colnames(xyz_data) <- c('x','y','z')

xyz_data2 <- xyz_data^2
Y <- rowSums(xyz_data2)
X <- cbind(2*xyz_data, rep(1,dim(xyz_data)[1]))
sph <- lsfit(X,Y,intercept = FALSE)
r_fit <- sqrt(sph[[1]][4]+sum(sph[[1]][1:3]^2)) #radius
cosph <- sph[["coefficients"]][c('x','y','z')] #center of the sphere

# put on a unit sphere
utp <- cbind(xyz_data, sqrt(rowSums(sweep(xyz_data, 2, cosph)^2)))
colnames(utp) <- c('x','y','z','r')
utp %<>% as_tibble() %>%
  mutate(theta = acos((z-cosph['z'])/r)) %>%
  mutate(phi = 2*pi*((y-cosph['y']) < 0) + (-1)^((y-cosph['y']) < 0)*acos((x-cosph['x'])/r/sin(theta))) %>%
  mutate(ux = sin(theta)*cos(phi), uy = sin(theta)*sin(phi), uz = cos(theta)) %>%
  as.data.frame()

utp_lens <- utp

nopen3d()
points3d(utp_lens[, c('ux','uy','uz')], size = 6)
spheres3d(0,0,0,1, col='grey', alpha=0.2)
planes3d(0,0,1, 0, alpha = 0.2)
planes3d(0,1,0, 0, alpha = 0.2)


# - EM Mi1
xyz_data <- Mi1_M10_xyz
colnames(xyz_data) <- c('x','y','z')

xyz_data2 <- xyz_data^2
Y <- rowSums(xyz_data2)
X <- cbind(2*xyz_data, rep(1,dim(xyz_data)[1]))
sph <- lsfit(X,Y,intercept = FALSE)
r_fit <- sqrt(sph[[1]][4]+sum(sph[[1]][1:3]^2))
cosph <- sph[["coefficients"]][c('x','y','z')]
# put on a unit sphere
utp <- cbind(xyz_data, sqrt(rowSums(sweep(xyz_data, 2, cosph)^2)))
utp <- as.data.frame(utp)
colnames(utp) <- c('x','y','z','r')
utp %<>% as_tibble() %>%
  mutate(theta = acos((z-cosph['z'])/r)) %>%
  mutate(phi = 2*pi*((y-cosph['y']) < 0) + (-1)^((y-cosph['y']) < 0)*acos((x-cosph['x'])/r/sin(theta))) %>%
  mutate(ux = sin(theta)*cos(phi), uy = sin(theta)*sin(phi), uz = cos(theta)) %>%
  as.data.frame()

utp_Mi1 <- utp

# align point sets -----------------------------------------------------------

# - EM Mi1
pt_100 <- 10 #upper eq 
pt_100_Mi1 <- pt_100
pt_up <- 315
pt_up_Mi1 <- pt_up

# use PR num
ind_up_Mi1 <- c(599,  39, 207, 606, 388,  10,  12, 157, 361, 287, 503, 105)
ind_down_Mi1 <- c(401, 334, 209, 210, 720, 667, 590, 628, 489, 289, 393)

pt_theta90_ID <- c(ind_up_Mi1, ind_down_Mi1)
pt_theta90_ID_Mi1 <- pt_theta90_ID
pt_theta90 <- rbind(c('ux'=0, 'uy'=0, 'uz'=0), utp_Mi1[pt_theta90_ID,c('ux','uy','uz')])
lm_plane <- lm(pt_theta90$ux ~ pt_theta90$uy + pt_theta90$uz + 0)
vn <- c(-1, lm_plane$coefficients[1], lm_plane$coefficients[2]) #norm
vn <- vn/sqrt(sum(vn^2))
if (sum(vn * utp_Mi1[pt_up, c('ux','uy','uz')]) < 0) {
  vn <- -vn
}

# rotation to the new coordinate
vx <- utp_Mi1[pt_100, c('ux','uy','uz')]
utp_Mi1_rot <- rot_zx0(utp_Mi1[,c('ux','uy','uz')], c(vn), c(vx))
pt_theta90_rot <- rot_zx0(pt_theta90, c(vn), c(vx))

# -- chiasm , just need a reflection (enough if a single plane )
utp_Mi1_rot_chiasm <- utp_Mi1_rot
utp_Mi1_rot_chiasm[,'Y'] <- -utp_Mi1_rot_chiasm[,'Y']

# - lens from micro CT
pt_100 <- 395 # 20240701
pt_up <- 659

pt_100_lens <- pt_100
pt_up_lens <- pt_up

pt_theta90_ID_lens <- c(ind_Up_lens, ind_Down_lens)
pt_theta90 <- rbind(c('ux'=0, 'uy'=0, 'uz'=0), utp_lens[pt_theta90_ID_lens,c('ux','uy','uz')])
lm_plane <- lm(pt_theta90$uz ~ pt_theta90$ux + pt_theta90$uy + 0)
vn <- c(lm_plane$coefficients[1], lm_plane$coefficients[2], -1) #norm
vn <- vn/sqrt(sum(vn^2))
# which side is up
if (sum(vn * utp_lens[pt_up, c('ux','uy','uz')]) < 0) {
  vn <- -vn
}

# rotation to the new coordinate
vx <- utp_lens[pt_100, c('ux','uy','uz')]
utp_lens_rot <- rot_zx0(utp_lens[,c('ux','uy','uz')], c(vn), c(vx))
pt_theta90_rot <- rot_zx0(pt_theta90, c(vn), c(vx))

# establish hex in lens -------------------------------------------------------------------

pt <- lens
ls <- reghex(pt, j7 = c(pt_100_lens,419, 430, 406, 372, 362, 385 ) ) 
nb_ind <- ls[[1]]
ind_xy <- ls[[2]]

# manual  matching -------------------------------------------------------------
# EM Mi1 to uCT lens

load('data/med_ixy.RData')
# shifting data
utp_lens_rot_shift <- sweep(utp_lens_rot,2,c(+1,0,0)) #NOTE the small area one should be on the convex side

lens_Mi1 <- matrix(ncol = 2, nrow = nrow(med_ixy))  # [lens index, Mi1 index ]
for (j in 1:nrow(med_ixy)) {
  ii <- which(med_ixy[j,2] == ind_xy[,2] & med_ixy[j,3] == ind_xy[,3])
  if (length(ii) == 1) {
    lens_Mi1[j,] <- c(ind_xy[ii,1], med_ixy[j,1])
  } else {
    lens_Mi1[j,] <- c(NA, med_ixy[j,1])
  }
}
# append un-matched Mi1
lens_Mi1_all <- rbind(lens_Mi1, cbind(ind_xy[!(ind_xy[,1] %in% na.omit(lens_Mi1[,1])), 1], NA))
# only matched
lens_Mi1 <- lens_Mi1[!is.na(lens_Mi1[,1]), ]

# -- eyemap, [med, lens (cone or ucl indeed)]
eyemap <- cbind(lens_Mi1[,2], lens_Mi1[,1])

# cont. re-order st. eyemap entry is neuron index, eyemap index is pts --------

med_xyz <- Mi1_M10_xyz[eyemap[,1],] 
rownames(med_xyz) <- eyemap[,1]
ucl_rot_sm <- ucl_rot_sm0[eyemap[,2], ] 
ind_Up_ucl <- match(ind_Up_lens, eyemap[,2])
ind_Down_ucl <- match(ind_Down_lens, eyemap[,2])

nb_ind <- matrix(match(nb_ind, eyemap[,2]), ncol = 7, byrow = F) # now it's eyemap index
nb_ind <- nb_ind[!is.na(nb_ind[,1]), ]

# save an un-ordered copy, including unmatched points
lens_ixy <- ind_xy

ind_xy[,1] <- match(ind_xy[,1], eyemap[,2])
ind_xy <- ind_xy[!is.na(ind_xy[,1]), ]

ord <- order(nb_ind[,1])
nb_ind <- nb_ind[ord,] 
ind_xy <- ind_xy[ord,]

# ioa
# right
dd <- ucl_rot_sm
nb_dist_ucl <- matrix(ncol = 7, nrow = nrow(dd))
for (j in 1:nrow(dd)) {
  nb_dist_ucl[nb_ind[j,1],] <- c(
    nb_ind[j,1], acos(dd[nb_ind[j,-1], ] %*% dd[nb_ind[j,1],]) )
}

# nb dist med [center, nbco]  -------------------------------------------

nb_dist_med <- matrix(ncol = 7, nrow = nrow(eyemap))
for (j in 1:nrow(eyemap)) {
  nb_dist_med[nb_ind[j,1],] <- c(
    nb_ind[j,1],
    sqrt(rowSums(sweep(med_xyz[nb_ind[j,-1], ], 2, med_xyz[nb_ind[j,1],,drop=F])^2))
  )
}

nbco <- rbind(c(+1, 0), # p-axis
              c(+1,+1), # v-axis
              c(0, +1), # q-axis
              c(-1, 0),
              c(-1,-1),
              c(0, -1)
)

# SAVE
save(nb_ind, ind_xy, nb_dist_med, nb_dist_ucl, nbco, file = "data/hexnb_ind_dist.RData")


#  add aux points beyond the boundary for regression ----------------------

pts <- ucl_rot_sm

# rotate to southern hemisphere
vcom <- colMeans(pts)
vcom <- vcom / sqrt(sum(vcom^2))
R_mat <- quaternion3D( cross3D(vcom, c(0,0,-1)), acos(vcom %*% c(0,0,-1)) /pi*180 )
pts_z <- as.matrix(pts) %*% t(R_mat)

# stereographic projection
x <- pts_z[,1] / (1 - pts_z[,3])
y <- pts_z[,2] / (1 - pts_z[,3])
stereo <- cbind(x, y)

dtdt <- deldir(stereo[,1], stereo[,2])

# -- deal with edge points,
# if the boundary pt, then check the 2 circumcenters sharing this edge,
# remove the one with no corresponding circumcenter
ee_mat <- matrix(c(1,1,1,2,2,1,2,2), ncol = 2, byrow = T) #index matrix

bp <- which(dtdt$dirsgs$bp1 | dtdt$dirsgs$bp2)

delsgs_mod <- dtdt$delsgs # modify
for (j in bp) {
  ind_edge <- dtdt$dirsgs[j, c('ind1', 'ind2')] #edge vertices
  ind1_tri <- c(delsgs_mod[delsgs_mod$ind1 %in% ind_edge[1], ]$ind2,
                delsgs_mod[delsgs_mod$ind2 %in% ind_edge[1], ]$ind1)
  ind2_tri <- c(delsgs_mod[delsgs_mod$ind1 %in% ind_edge[2], ]$ind2,
                delsgs_mod[delsgs_mod$ind2 %in% ind_edge[2], ]$ind1)
  ind_tri <- c(ind1_tri, ind2_tri)
  ind_share <- ind_tri[duplicated(ind_tri)] # two vertices shared by the edge vertices

  if (length(ind_share) == 2) {
    ee <- c() #exist a circumcenter
    for (m in 1:2) {
      for (n in 1:2) {
        b1 <- ind_edge[m] %in% dtdt$dirsgs[dtdt$dirsgs$ind1 %in% ind_share[n], ]$ind2
        b2 <- ind_edge[m] %in% dtdt$dirsgs[dtdt$dirsgs$ind2 %in% ind_share[n], ]$ind1
        ee <- c(ee, b1 | b2)
      }
    }
    if (sum(ee) < 4) {
      for (k in which(!ee)) {
        # to remove
        ind1 <- delsgs_mod$ind1 %in% ind_edge[ee_mat[k,][1]] & delsgs_mod$ind2 %in% ind_share[ee_mat[k,][2]]
        ind2 <- delsgs_mod$ind2 %in% ind_edge[ee_mat[k,][1]] & delsgs_mod$ind1 %in% ind_share[ee_mat[k,][2]]
        ind_rm <- ind1 | ind2
        delsgs_mod <- delsgs_mod[-which(ind_rm), ]
      }
    }
  }
}

# remove long edge
el <- pts[delsgs_mod$ind1,] - pts[delsgs_mod$ind2,]
el <- sqrt(rowSums(el^2))
delsgs_mod <- delsgs_mod[el < 0.27, ]

stereo_as <- ashape(stereo, alpha = 0.2)
xy_edge <- stereo_as$edges[,1:6]
stereo_poly <- mkpoly(xy_edge)[[1]]

ind_bd <- stereo_poly$ind1 # boundary pt index

stereo_bd <- stereo_poly[,3:4]
stereo_bd <- rbind(stereo_bd, stereo_bd[1,])

# -- add aux pt
pt_aux <- matrix(ncol = 3, nrow = 0)
pt_med_aux <- matrix(ncol = 3, nrow = 0)
for (iibd in ind_bd) {
  # neighbors of this point
  ind_nb <- c(delsgs_mod$ind2[delsgs_mod$ind1 %in% iibd], delsgs_mod$ind1[delsgs_mod$ind2 %in% iibd])
  # aux pt for each neighbor
  for (iinb in ind_nb) {
      xyz <- ucl_rot_sm[iibd,] + ucl_rot_sm[iibd,] - ucl_rot_sm[iinb,] 
      pt_aux <- rbind(pt_aux, matrix(xyz/sqrt(sum(xyz^2)),ncol = 3))
      xyz <- med_xyz[iibd,] + med_xyz[iibd,] - med_xyz[iinb,] 
      pt_med_aux <- rbind(pt_med_aux, xyz)
  }
}

# remove pts within the polygon
pt_aux_z <- as.matrix(pt_aux) %*% t(R_mat)
# stereographic projection
x <- pt_aux_z[,1] / (1 - pt_aux_z[,3])
y <- pt_aux_z[,2] / (1 - pt_aux_z[,3])
stereo_aux <- cbind(x, y) 

ii_inpoly <- point.in.polygon(stereo_aux[,1], stereo_aux[,2], stereo_bd[,1], stereo_bd[,2])

pt_aux <- pt_aux[!ii_inpoly,]
pt_med_aux <- pt_med_aux[!ii_inpoly,]

# remove pts close to ucl_rot_sm
ii_dd <- c()
for (ii in 1:nrow(pt_aux)) {
  ii_dd <- c(ii_dd, min(sqrt(rowSums(sweep(ucl_rot_sm, 2, pt_aux[ii,])^2))) > 0.08)
}
pt_aux <- pt_aux[ii_dd,]
pt_med_aux <- pt_med_aux[ii_dd,]

# remove clustered pt
ii_o <- order(sqrt(rowSums(sweep(pt_aux, 2, colMeans(ucl_rot_sm))^2)))
pt_aux <- pt_aux[ii_o,]
pt_med_aux <- pt_med_aux[ii_o,]
ii <- 1
while (ii < nrow(pt_aux)) {
  dd <- sqrt(rowSums(sweep(pt_aux, 2, pt_aux[ii,])^2))
  ii_dd <- dd > 0.1
  ii_dd[ii] <- TRUE
  pt_aux <- pt_aux[ii_dd,]
  pt_med_aux <- pt_med_aux[ii_dd,]
  ii <- ii + 1
}
pt_aux <- sweep(pt_aux,1,sqrt(rowSums(pt_aux^2)),'/') #normalize

# remove pts too far from med_xyz
ii_dd <- c()
for (ii in 1:nrow(pt_med_aux)) {
  ii_dd <- c(ii_dd, min(sqrt(rowSums(sweep(med_xyz, 2, pt_med_aux[ii,])^2))) < 1e4)
}
pt_aux <- pt_aux[ii_dd,]
pt_med_aux <- pt_med_aux[ii_dd,]

# --- re-define
Npt <- nrow(eyemap)
rownames(pt_aux) <- seq(Npt + 1, Npt + nrow(pt_aux))
rownames(pt_med_aux) <- seq(Npt + 1, Npt + nrow(pt_med_aux))

# aux
ucl_rot_aux <- rbind(ucl_rot_sm, pt_aux)
med_xyz_aux <- rbind(med_xyz, pt_med_aux)
eyemap_aux <- rbind(eyemap, cbind(seq(nrow(eyemap)+1, nrow(eyemap)+nrow(pt_aux)),
                              seq(nrow(eyemap)+1, nrow(eyemap)+nrow(pt_aux))) )


# SAVE eyemap -------------------------------------------------------------

save(eyemap, med_xyz, delsgs_mod, ucl_rot_sm, lens_Mi1, lens_ixy,
     utp_lens_rot, utp_lens_rot_shift, utp_Mi1_rot, utp_Mi1_rot_chiasm, ind_Up_ucl, ind_Down_ucl,
     ucl_rot_aux, med_xyz_aux, eyemap_aux, Npt,
     file = "data/eyemap.RData")

# ME to LOP ------------------------------------------------------------------

# NOT run, VFB has no these annotations
# LL <- 2
# anno_str <- paste("T4",letters[LL], " - complete", sep = "")
# anno_T4 <- catmaid_query_by_annotation(anno_str)
# neu_skid_comp <- anno_T4[,"skid"]
# neu_T4_comp <-  read.neurons.catmaid(neu_skid_comp, .progress = 'text')
# T4b <- neu_T4_comp
# anno_T4b <- anno_T4
# 
# save(anno_T4b, T4b, file = "data/neu_T4b.RData")

# eyemap
load('data/eyemap.RData')
# lob layer
load('data/LayerData_T4_20210729_7e7_1um.RData')
# load T4
load("data/neu_T4b.RData")

layer_ls <- xyz_layer_T4
# combine grid points from 4 layers
len_layer <- c() #length of each grid
II <- 0
for (j in 1:4) {
  len_layer[j] <- dim(layer_ls[[j]])[1]
  row.names(layer_ls[[j]]) <- seq(II+1, II+len_layer[j]) # change names to canonical ordering
  II <- II+len_layer[j]
}
xyz_4layer <- do.call(rbind, layer_ls)

# calculate T4 com
T4b_com <- matrix(nrow = length(T4b), ncol = 6) #com, [dendrite, axon]
for (j in 1:length(T4b)) {
  tar <- T4b[[j]]
  ind_T = match(tar$tags$"T-bars start" , tar$d$PointNo)
  mat_T = tar$d[ind_T,]
  xyz_T <- xyzmatrix(mat_T)
  ind_D = match(tar$tags$"dendrite start" , tar$d$PointNo)
  mat_D = tar$d[ind_D,]
  xyz_D <- xyzmatrix(mat_D)
  
  dp_T <- child_node(tar, mat_T) #downstream nodes of mat_T
  dp_T <- xyzmatrix(dp_T)
  dp_D <- child_node(tar, mat_D) 
  dp_D <- xyzmatrix(dp_D)
  
  T4b_com[j,] <- c(colMeans(dp_D), colMeans(dp_T))
}

# map T4 onto L2 grid
T4b_L2_xyz <- matrix(ncol = 3, nrow = length(T4b))
for (j in 1:length(T4b)) {
  ii_L2 <- which.min(rowSums(sweep(xyz_layer_T4[[2]], 2, T4b_com[j, 4:6])^2))
  T4b_L2_xyz[j,] <- unlist(xyz_layer_T4[[2]][ii_L2,])
}

# kernel regression global
np_eval <- med_xyz
lop_pred <- matrix(nrow = nrow(np_eval), ncol = 3)
for (k in 1:3) {
  npdata <- data.frame(mc = T4b_com[,1:3], ec = T4b_L2_xyz[,k])
  bw <- npregbw(formula= ec~mc.1+mc.2+mc.3, data= npdata, bwtype= 'generalized_nn', regtype= 'll') 
  model_np <- npreg(bw)
  for (j in 1:nrow(np_eval)) {
    # med col
    np_eval_one <- data.frame(mc.1 = np_eval[j,1], mc.2 = np_eval[j,2], mc.3 = np_eval[j,3])
    lop_pred[j,k] <- predict(model_np, newdata = np_eval_one)
  }
}
lop_pred_global <- lop_pred

# - combine pt in layer 2 and associate with ii_stack
L2_med <- cbind(lop_pred, med_xyz) #pt in LOP, mapped by T4 from med col
rownames(L2_med) <- rownames(med_xyz)

# SAVE
save(L2_med, T4b_com, file = "data/L2_med.RData")


# TmY5a to LO --------------------------------------------------------------

load('data/neu_Mi1.RData')
load('data/neu_TmY5a.RData')
load('data/eyemap.RData')
load('data/Mi1_ind.RData')

tag_xyz <- matrix(ncol = 9, nrow = length(TmY5a))
for (j in 1:length(TmY5a)) {
  tar <- TmY5a[[j]]
  M10 = match(tar$tags$"M10", tar$d$PointNo)
  LO4 = match(tar$tags$"LO_LC4", tar$d$PointNo)
  LoP2 = match(tar$tags$"LoP2", tar$d$PointNo)
  tag_xyz[j,] <- c(t(xyzmatrix(tar$d[c(M10, LO4, LoP2),])))
}

# matching Mi1 (for length(TmY5a)==63 only)
TmY5a_Mi1 <- c()
for (j in 1:length(TmY5a)) {
  TmY5a_Mi1 <- c(TmY5a_Mi1, sweep(med_xyz, 2, tag_xyz[j,1:3])^2 %>% rowSums() %>% which.min())
}
TmY5a_Mi1[23] <- 326 # manual corr
TmY5a_Mi1[15] <- 650

# - kernel regression global
np_eval <- med_xyz
lo_pred <- matrix(nrow = nrow(np_eval), ncol = 3)
for (k in 1:3) {
  npdata <- data.frame(mc = tag_xyz[,1:3], ec = tag_xyz[,3+k])
  bw <- npregbw(formula= ec~mc.1+mc.2+mc.3, data= npdata, bwtype= 'generalized_nn', regtype= 'll') 
  model_np <- npreg(bw)
  for (j in 1:nrow(np_eval)) {
    # LO col
    np_eval_one <- data.frame(mc.1 = np_eval[j,1], mc.2 = np_eval[j,2], mc.3 = np_eval[j,3])
    lo_pred[j,k] <- predict(model_np, newdata = np_eval_one)
  }
}

# SAVE
save(TmY5a, anno_TmY5a, file = "data/neu_TmY5a.RData")
save(lo_pred, tag_xyz, file = "data/me_lo.RData")
