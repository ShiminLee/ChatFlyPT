
library(tidyverse)
library(clue) # hungarian

# clean everythign up.
rm(list=ls())
 
source("eyemap_func.R")

# main loop -----------------------------------------------------------

# load position data, match lens to cone/tip to compute directions
# smooth positions and directions (almost redundant)

fn <- c('20240701')

for (f in 1:length(fn)) {
  lens <- read.csv(paste("data/microCT/", fn[f], "_position/lens.csv", sep=''), header=T) 
  cone <- read.csv(paste("data/microCT/", fn[f], "_position/cone.csv", sep=''), header=T) 
  
  # extract
  lens <- lens[, c('Position.X','Position.Y','Position.Z')]
  colnames(lens) <- c("x", "y", "z")
  lens <- lens[order(lens$x, lens$y,lens$z),]
  rownames(lens) <- seq(1,length(lens$x), by = 1)
  
  cone <- cone[, c('Position.X','Position.Y','Position.Z','Default.Labels')]
  colnames(cone) <- c("x", "y", "z", "label")
  cone <- cone[order(cone$x, cone$y,cone$z),]
  rownames(cone) <- seq(1,length(cone$x), by = 1)
  ind_Down <- which(cone$label == "DOWN")
  ind_Up <- which(cone$label == "UP")
  cone <- cone[,c("x", "y", "z")]
  
  # - alignment
  # equator
  v_up <- colMeans(cone[ind_Up,]) - colMeans(cone[ind_Down,])
  cone_equator <- cone[c(ind_Up, ind_Down),]
  pt_theta90 <- cone_equator
  vn <- prcomp(pt_theta90)$rotation[,3]
  vn <- vn/sqrt(sum(vn^2))
  # which side is up
  if (sum(vn * v_up) < 0) {
    vn <- -vn
  }
  
  # front via pca, use both eyes
  eye_pca <- prcomp(rbind(lens, cone))
  vf <- eye_pca$rotation[,3]
  vf <- vf - c(vf %*% vn) * vn
  vf <- vf / sqrt(sum(vf^2))
  if (vf %*% (colMeans(lens) - eye_pca$center) < 0) {
    vf <- - vf
  }
  vleft <- cross3D(vn, vf)

  # - separate left and right
  vleft <- cross3D(vn, vf)
  ind_left_lens <- c(t(vleft) %*% t(sweep(lens,2,eye_pca$center,'-')) > 0)
  ind_left_cone <- c(t(vleft) %*% t(sweep(cone,2,eye_pca$center,'-')) > 0)
  
  # - matching, number of points SHOULD be equal
  L <- dim(cone)[1]
  Lr <- L
  mDist <- matrix(ncol = L, nrow = L)
  dr <- cone
  dc <- lens
  for (j in 1:Lr) {
    mDist[j,] <- as.matrix(dist(rbind(dr[j,], dc)))[-1,1]
  }
  i_match <- solve_LSAP(mDist)
  
  # - centering
  mrot <- cbind(vf,vleft,vn)
  lens <- sweep(as.matrix(lens),2,eye_pca$center,'-') %*% mrot
  cone <- sweep(as.matrix(cone),2,eye_pca$center,'-') %*% mrot
  colnames(lens) <- c('x','y','z')
  colnames(cone) <- c('x','y','z')
  vf <- c(1,0,0)
  vleft <- c(0,1,0)
  vn <- c(0,0,1)
  
  # - smoothing  (separately --> not much difference)
  lens_sm <- lens
  cone_sm <- cone

  # left lens
  pts <- lens[ind_left_lens,]
  ls <- reghex(pts)
  nb_ind <- ls[[1]]
  ptsnew <- pts
  for (m in 1:nrow(nb_ind)) {
    if (!any(is.na(nb_ind[m,]))) {
      ptsnew[nb_ind[m,1],] <- pts[nb_ind[m,1],]/2 + colMeans(pts[nb_ind[m,-1],])/2
    }
  }
  lens_sm[ind_left_lens,] <- ptsnew

  # left cone
  pts <- cone[ind_left_cone,]
  ls <- reghex(pts)
  nb_ind <- ls[[1]]
  ptsnew <- pts
  for (m in 1:nrow(nb_ind)) {
    if (!any(is.na(nb_ind[m,]))) {
      ptsnew[nb_ind[m,1],] <- pts[nb_ind[m,1],]/2 + colMeans(pts[nb_ind[m,-1],])/2
    }
  }
  cone_sm[ind_left_cone,] <- ptsnew

  # right lens
  pts <- lens[!ind_left_lens,]
  ls <- reghex(pts)
  nb_ind <- ls[[1]]
  ptsnew <- pts
  for (m in 1:nrow(nb_ind)) {
    if (!any(is.na(nb_ind[m,]))) {
      ptsnew[nb_ind[m,1],] <- pts[nb_ind[m,1],]/2 + colMeans(pts[nb_ind[m,-1],])/2
    }
  }
  lens_sm[!ind_left_lens,] <- ptsnew

  # right cone
  pts <- cone[!ind_left_cone,]
  ls <- reghex(pts)
  nb_ind <- ls[[1]]
  ptsnew <- pts
  for (m in 1:nrow(nb_ind)) {
    if (!any(is.na(nb_ind[m,]))) {
      ptsnew[nb_ind[m,1],] <- pts[nb_ind[m,1],]/2 + colMeans(pts[nb_ind[m,-1],])/2
    }
  }
  cone_sm[!ind_left_cone,] <- ptsnew
  
  lens0 <- lens
  cone0 <- cone
  lens <- lens_sm
  cone <- cone_sm
  
  # - direction unit vec
  ucl <- lens[i_match,] - cone #use cone index
  # ucl <- lens_sm[i_match,] - cone_sm #use cone index
  ucl_norm <- sqrt(rowSums(ucl^2))
  ucl <- ucl / ucl_norm
  
  # - rotation to canonical coordinate
  # vn -> +z, vf -> +x, this way, equatorial ucl not on equatorial plane given lens positions
  mrot <- cbind(vf,vleft,vn)
  ucl_rot <- as.matrix(ucl) %*% mrot
  # -- smoothing (already done earlier)
  ucl_rot_sm <- ucl_rot
  
  # separate left and right ommatidia directions
  ii <- cone[,2] > 0 
  # left
  cc <- cone[ii,]
  ucl_rot_1 <- ucl_rot[ii, ]
  ls <- reghex(cc)
  nb_ind <- ls[[1]]
  ucl_rot_new <- ucl_rot_1
  for (m in 1:nrow(nb_ind)) {
    if (!any(is.na(nb_ind[m,]))) {
      ucl_rot_new[nb_ind[m,1],] <- ucl_rot_1[nb_ind[m,1],]/2 + colMeans(ucl_rot_1[nb_ind[m,-1],])/2
    }
  }
  ucl_rot_sm[ii, ] <- ucl_rot_new
  
  # right
  cc <- cone[!ii,]
  ucl_rot_1 <- ucl_rot[!ii, ]
  ls <- reghex(cc)
  nb_ind <- ls[[1]]
  ucl_rot_new <- ucl_rot_1
  for (m in 1:nrow(nb_ind)) {
    if (!any(is.na(nb_ind[m,]))) {
      ucl_rot_new[nb_ind[m,1],] <- ucl_rot_1[nb_ind[m,1],]/2 + colMeans(ucl_rot_1[nb_ind[m,-1],])/2
    }
  }
  ucl_rot_sm[!ii, ] <- ucl_rot_new
  
  # normalize
  ucl_rot_sm <- sweep(ucl_rot_sm,1,sqrt(rowSums(ucl_rot_sm^2)),'/') #normalize
  
  # SAVE
  save(lens, cone, ind_Up, ind_Down, ind_left_lens, ind_left_cone,
       eye_pca, vn, vf, vleft, ucl_rot, L, i_match, ucl,
       ucl_rot_sm,
       file = paste("data/microCT/", fn[f], ".RData", sep='') )
}

# nb_ind and ioa, 20240701 ------------------------------------------------

load(paste0("data/microCT/20240701.RData"))

# - hex coord and ioa
pt <- lens[ind_left_lens, ]
ls <- reghex(pt, lefteye = T )
nb_ind_left <- ls[[1]]
ind_xy_left <- ls[[2]]

pt <- lens[!ind_left_lens, ]
ls <- reghex(pt, lefteye = F)
nb_ind_right <- ls[[1]]
ind_xy_right <- ls[[2]]

# ioa, left
dd <- ucl_rot_sm[order(i_match),][ind_left_lens,]
nb_ind <- nb_ind_left
nb_dist_ucl <- matrix(ncol = 7, nrow = nrow(dd)) #arc dist
for (j in 1:nrow(dd)) {
  nb_dist_ucl[nb_ind[j,1],] <- c(
    nb_ind[j,1],acos(dd[nb_ind[j,-1], ] %*% dd[nb_ind[j,1],])
  )
}
nb_dist_ucl_left <- nb_dist_ucl

# right
dd <- ucl_rot_sm[order(i_match),][!ind_left_lens,]
nb_ind <- nb_ind_right
nb_dist_ucl <- matrix(ncol = 7, nrow = nrow(dd))
for (j in 1:nrow(dd)) {
  nb_dist_ucl[nb_ind[j,1],] <- c(
    nb_ind[j,1], acos(dd[nb_ind[j,-1], ] %*% dd[nb_ind[j,1],]) 
  )
}
nb_dist_ucl_right <- nb_dist_ucl

# SAVE
save(nb_ind_left, nb_ind_right, ind_xy_left, ind_xy_right,
     nb_dist_ucl_left, nb_dist_ucl_right,
     file = paste0("data/microCT/20240701_nb.RData") )



# direction based on normals, only right side --------------------------------

f <- 1

load(paste0("data/microCT/", fn[f], ".RData"))
load(paste0("data/microCT/", fn[f], "_nb.RData"))

pts <- lens[!ind_left_lens,]
nb_ind <- nb_ind_right
ucl <- ucl_rot_sm[order(i_match),][!ind_left_lens,]

# - calculate normals
nucl <- matrix(nrow = nrow(pts), ncol = 3)
for (j in 1:nrow(nucl)) {
  if (sum(is.na(nb_ind[j,])) == 0) {
    pca <- prcomp(pts[nb_ind[j,],])
    vv <- pca$rotation[,3]
    if (vv %*% ucl[nb_ind[j,1],] > 0) {
      nucl[nb_ind[j,1], ] <- vv
    } else {
      nucl[nb_ind[j,1], ] <- -vv
    }
  }
}
save(nucl, file = paste("data/microCT/", fn[f], "_normals.RData", sep='') )

# curvature and sphericity ------------------------------------------------

for (f in 1:length(fn)) {
  load(paste("data/microCT/", fn[f], ".RData", sep=''))  
  
  for (lr in c(TRUE, FALSE)) {
    if (lr) { #left
      pts <- lens[ind_left_lens,]
    } else { #right
      pts <- lens[!ind_left_lens,]
    }
    
    # hex coord
    ls <- reghex(pts, lefteye = lr)
    ind_nb <- ls[[1]]
    ind_xy <- ls[[2]]
    
    ind_roc <- matrix(ncol = 6, nrow = nrow(pts)) #[ind, p,v,q,h,sph]
    ind_dia <- matrix(ncol = 2, nrow = nrow(pts)) #[ind, d]
    ind_roc[,1] <- ind_xy[,1]
    ind_dia[,1] <- ind_xy[,1]
    
    # - avg dia, iterate over all pts, compute distnace to all 6 nbs, 
    for (j in 1:dim(ind_dia)[1]) {
      if (sum(is.na(ind_nb[j,])) == 0) {
        ind_dia[j,2] <- 
          sweep(pts[ind_nb[j,-1],], 2, pts[ind_nb[j,1],,drop=F])^2 %>%
          rowSums() %>%
          sqrt()%>%
          mean()
      }
    }
    
    # - iterate over p\v\q\h=const and compute radius of cur.
    N <- 2 #no. of nbs on one side, total = 2N+1
    
    # p=const
    for (cc in seq(min(ind_xy[,2]),max(ind_xy[,2]))) {
      ixy <- ind_xy[ind_xy[,2] == cc,,drop=F]
      if (nrow(ixy) >= 2*N+1) {
        ixy <- ixy[order(ixy[,3]),] #order by q
        # iterate over point on the line with sufficient nbs
        for (j in (1+N):(nrow(ixy)-N)) {
          xyz <- pts[ixy[j + (-N:N),1], ]
          ind_roc[match(ixy[j,1],ind_xy[,1]), 4] <- roc_fit(xyz) #along q
        }
      }
    }
    
    # q=const
    for (cc in seq(min(ind_xy[,3]),max(ind_xy[,3]))) {
      ixy <- ind_xy[ind_xy[,3] == cc,,drop=F]
      if (nrow(ixy) >= 2*N+1) {
        ixy <- ixy[order(ixy[,2]),] #order by p
        # iterate over point on the line with sufficient nbs
        for (j in (1+N):(nrow(ixy)-N)) {
          xyz <- pts[ixy[j + (-N:N),1], ]
          ind_roc[match(ixy[j,1],ind_xy[,1]), 2] <- roc_fit(xyz) #along p
        }
      }
    }
    
    # v=const, use N-1
    for (cc in seq(min(ind_xy[,2]+ind_xy[,3]), max(ind_xy[,2]+ind_xy[,3]))) {
      ixy <- ind_xy[ind_xy[,2]+ind_xy[,3] == cc,,drop=F]
      if (nrow(ixy) >= 2*(N-1)+1) {
        ixy <- ixy[order(ixy[,3]-ixy[,2]),] #order by h
        # iterate over point on the line with sufficient nbs
        for (j in (1+(N-1)):(nrow(ixy)-(N-1))) {
          xyz <- pts[ixy[j + (-(N-1):(N-1)),1], ]
          ind_roc[match(ixy[j,1],ind_xy[,1]), 5] <- roc_fit(xyz) #along h
        }
      }
    }
    
    # h=const
    for (cc in seq(min(ind_xy[,3]-ind_xy[,2]),max(ind_xy[,3]-ind_xy[,2]))) {
      ixy <- ind_xy[ind_xy[,3]-ind_xy[,2] == cc,,drop=F]
      if (nrow(ixy) >= 2*N+1) {
        ixy <- ixy[order(ixy[,2]+ixy[,3]),] #order by v
        # iterate over point on the line with sufficient nbs
        for (j in (1+N):(nrow(ixy)-N)) {
          xyz <- pts[ixy[j + (-N:N),1], ]
          ind_roc[match(ixy[j,1],ind_xy[,1]), 3] <- roc_fit(xyz) #along v
        }
      }
    }
    
    # sphere
    NS <- 2 #avg over no. of layers of hex
    for (j in 1:nrow(ind_xy)) {
      ii <- ind_xy[j,1]
      for (k in 1:NS) {
        ii <- unique(c(ind_nb[match(ii,ind_nb[,1]),]))
        ii <- na.omit(ii)
      }
      if (length(ii) >= (19-3)) { #missing one side
        xyz_data <- pts[ii, ]
        colnames(xyz_data) <- c('x','y','z')
        xyz_data2 <- xyz_data^2
        Y <- rowSums(xyz_data2)
        X <- cbind(2*xyz_data, rep(1,dim(xyz_data)[1]))
        sph <- lsfit(X,Y,intercept = FALSE)
        r_fit <- sqrt(sph[[1]][4]+sum(sph[[1]][1:3]^2)) #radius
        ind_roc[j, 6] <- r_fit
      }
    }
    
    if (lr) { #left
      ind_roc_left <- ind_roc
      pts_left <- pts
      ind_xy_left <- ind_xy
      ind_nb_left <- ind_nb
      ind_dia_left <- ind_dia
    } else { #right
      ind_roc_right <- ind_roc
      pts_right <- pts
      ind_xy_right <- ind_xy
      ind_nb_right <- ind_nb
      ind_dia_right <- ind_dia
    }
  }
  # SAVE
  save(ind_roc_left, ind_roc_right, ind_xy_left, ind_xy_right,
       file = paste0("data/microCT/", fn[f], "_roc.RData"))
  save(ind_dia_left, ind_dia_right, ind_nb_left, ind_nb_right,
       file = paste0("data/microCT/", fn[f], "_dia.RData"))
}#loop
