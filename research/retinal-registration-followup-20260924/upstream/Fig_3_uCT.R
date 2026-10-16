# re-run Figure_0 after this script

# set up Mollweide projection
ucl_rot_tp <- cart2sphZ(ucl_rot_2eye)[,2:3] %>%
  as_tibble() %>%
  mutate(theta = theta/pi*180, phi = phi/pi*180) %>%
  mutate(phi = if_else(phi > 180, phi - 360, phi)) %>%
  mutate(phi = - phi) %>% #inside out
  as.matrix()
ucl_rot_Mo2 <- Mollweide(ucl_rot_tp)
ucl_rot_Mo2 <- cbind(ucl_rot_Mo2, 1)
ucl_rot_Mo2[!ind_left_cone, 3] <- 2 # left == 1, right == 2
colnames(ucl_rot_Mo2) <- c('xM', 'yM', 'left')

# ED Fig.4D, Mollweide projection of 2 eyes --------------------------------

xy_ashape <- ashape(ucl_rot_Mo2[ind_left_cone, 1:2] +
                      matrix(runif(sum(ind_left_cone)*2, 1e-9, 2e-9), ncol=2), alpha = 0.3)
xy_edge <- xy_ashape$edges[,1:6]
xy_grid_ahull <- mkpoly(xy_edge)[[1]][,3:4] # hull edge points
xy_grid_ahull <- rbind(xy_grid_ahull, xy_grid_ahull[1,])
ahull_left <- xy_grid_ahull

xy_ashape <- ashape(ucl_rot_Mo2[!ind_left_cone, 1:2] +
                      matrix(runif(sum(!ind_left_cone)*2, 1e-9, 2e-9), ncol=2), alpha = 0.3)
xy_edge <- xy_ashape$edges[,1:6]
xy_grid_ahull <- mkpoly(xy_edge)[[1]][,3:4] # hull edge points
xy_grid_ahull <- rbind(xy_grid_ahull, xy_grid_ahull[1,])
ahull_right <- xy_grid_ahull

# point distr
df <- as.data.frame(ucl_rot_Mo2)
windows(width = 16, height = 8)
# pdf("viewing_angles_bi.pdf")
plt <- plt_Mo + 
  geom_point(data=df, aes(x=xM, y=yM, colour = factor(left)), size=1) +
  scale_colour_manual(values = pal_lr, labels=c('left','right')) +
  geom_path(data = ahull_left, aes(x1,y1), lwd = 1, colour = pal_lr[1], alpha =0.7) +
  geom_path(data = ahull_right, aes(x1,y1), lwd = 1, colour = pal_lr[2], alpha =0.7) +
  theme(legend.position = c(.9, .9) ) +
  labs(title = '2 eyes')
plt
# dev.off()

# Fig.3F, inter-ommatidia angle --------------------------

ioa_hex <- cbind(ucl_rot_Mo_right, rowMeans(nb_dist_ucl_right[,2:7], na.rm = T)/pi*180)
ioa_hex <- as.data.frame(ioa_hex)
colnames(ioa_hex) <- c('xM','yM','ioa')
ioa_hex_right <- ioa_hex

ioa_hex <- cbind(ucl_rot_Mo_left, rowMeans(nb_dist_ucl_left[,2:7], na.rm = T)/pi*180) 
ioa_hex <- as.data.frame(ioa_hex)
colnames(ioa_hex) <- c('xM','yM','ioa')
ioa_hex_left <- ioa_hex

# -- pal and breaks
n_lvl <- 8
breaks_ioa <- seq(1,11,length.out = n_lvl+1)

# -- grid for loessm ioa
bkgd_chull <- rbind(bkgd_mer_ww, bkgd_mer_ee[seq(nrow(bkgd_mer_ee),1,by=-1),])
colnames(bkgd_chull) <- c('xM','yM')
grid_M <- expand.grid(xM = seq(-sqrt(8), sqrt(8), length.out = 100),
                      yM = seq(-sqrt(2), sqrt(2), length.out = 50) )
ii_inpoly <- sp::point.in.polygon(grid_M[,1], grid_M[,2], bkgd_chull[,1], bkgd_chull[,2])

# - loess
# -- left
fit_loess <- loess(ioa ~ xM * yM, data =ioa_hex_left, degree = 2, span = 0.1,
                   control = loess.control(surface = "direct"))
pred_loess <- predict(fit_loess, grid_M, se = T)
df_pred <- grid_M
df_pred$Z <- melt(pred_loess$fit)$value
df_pred$equalSpace <- cut(df_pred$Z, breaks_ioa)
df_pred_left <- df_pred[ii_inpoly == 1, ]
df_pred_left$colSpace <- as.numeric(df_pred_left$equalSpace)
df_pred_left$colSpace[is.na(df_pred_left$colSpace)] <- 0

xy_ashape <- ashape(ucl_rot_Mo2[ind_left_cone, 1:2] +
                      matrix(runif(sum(ind_left_cone)*2, 1e-9, 2e-9), ncol=2), alpha = 0.3)
xy_edge <- xy_ashape$edges[,1:6]
xy_grid_ahull <- mkpoly(xy_edge)[[1]][,3:4] # hull edge points
xy_grid_ahull <- rbind(xy_grid_ahull, xy_grid_ahull[1,])
ahull_left <- xy_grid_ahull
ii_inpoly_left <- sp::point.in.polygon(grid_M[ii_inpoly == 1,1], grid_M[ii_inpoly == 1,2], xy_grid_ahull[,1], xy_grid_ahull[,2])
df_pred_left$Z[!ii_inpoly_left] <- 20 # outside set to large angles
df_pred_left$colSpace[!ii_inpoly_left] <- 0

# -- right
fit_loess <- loess(ioa ~ xM * yM, data =ioa_hex_right, degree = 2, span = 0.1,
                   control = loess.control(surface = "direct"))
pred_loess <- predict(fit_loess, grid_M, se = T)
df_pred <- grid_M
df_pred$Z <- melt(pred_loess$fit)$value
df_pred$equalSpace <- cut(df_pred$Z, breaks_ioa)
df_pred_right <- df_pred[ii_inpoly == 1, ]
df_pred_right$colSpace <- as.numeric(df_pred_right$equalSpace)
df_pred_right$colSpace[is.na(df_pred_right$colSpace)] <- 0

xy_ashape <- ashape(ucl_rot_Mo2[!ind_left_cone, 1:2] +
                      matrix(runif(sum(!ind_left_cone)*2, 1e-9, 2e-9), ncol=2), alpha = 0.3)
xy_edge <- xy_ashape$edges[,1:6]
xy_grid_ahull <- mkpoly(xy_edge)[[1]][,3:4] # hull edge points
xy_grid_ahull <- rbind(xy_grid_ahull, xy_grid_ahull[1,])
ahull_right <- xy_grid_ahull
ii_inpoly_right <- sp::point.in.polygon(grid_M[ii_inpoly == 1,1], grid_M[ii_inpoly == 1,2], xy_grid_ahull[,1], xy_grid_ahull[,2])
df_pred_right$Z[!ii_inpoly_right] <- 20
df_pred_right$colSpace[!ii_inpoly_right] <- 0

# -- plot
breaks_contour <- c(0, 4, 5, 6)
getPalette <- colorRampPalette(c('gray30', 'gray90'))
pal_contour <- getPalette(length(breaks_contour)+1)

# PLOT contour, choose left or right eye
df <- as.data.frame(ucl_rot_Mo_right)
plt <- plt_Mo +
  geom_contour(data=df_pred_right, aes(x=xM,y=yM, z=Z), breaks=breaks_contour,
               colour=pal_lr[2], alpha=0.9, lwd=0.5) +
  geom_point(data = as.data.frame(ucl_rot_Mo2[!ind_left_cone,]), aes(xM, yM), colour ='gray70', size=1, alpha=1) +
  geom_point(data = df[lens_ixy[lens_ixy[,3] == clq,1], ], aes(xM, yM), colour = pal_axes[1], size=2) +
  geom_point(data = df[c(ind_Up_lens, ind_Down_lens), ], aes(xM, yM), colour = pal_axes[4], size=2) +
  geom_point(data = df[lens_ixy[lens_ixy[,2] == clp,1], ], aes(xM, yM), colour = pal_axes[2], size=2) +
  geom_point(data = cmer_right, aes(x=xM, y=yM), colour = pal_axes[3], size=2) +
  labs(title = paste("ioa"))
windows(width = 16, height = 8)
# pdf("eye_right_Mollweide_ioa_hex_4axes.pdf", width = 8.5, height = 4.5)
plt
# dev.off()


# Fig.3E, just 3d ucl ------------------------------------------------------------

nopen3d()
par3d('windowRect' = c(100,100,1000,1000))
spheres3d(0,0,0,0.99, col='grey95', alpha=1, lit=F)
points3d(ucl_rot_sm[ind_xy[ind_xy[,2] == clp, 1], ], size=12, col=pal_axes[2])
points3d(ucl_rot_sm[ind_xy[ind_xy[,3] == clq, 1], ], size=12, col=pal_axes[1])
points3d(ucl_rot_sm[vaxis_gen(clv, ind_xy), ], size=12, col=pal_axes[3])
points3d(ucl_rot_sm[haxis_gen(clh, ind_xy), ], size=12, col=pal_axes[4])
points3d(ucl_rot_sm, size=7, col='gray50')
arrow3d(c(1,0,0), c(1.6, 0, 0), theta = pi / 18, n = 8, col = "gray30", type = "rotation")
arrow3d(c(0,0,1), c(0, 0, 1.5), theta = pi / 12, n = 8, col = "gray30", type = "rotation")
arrow3d(c(0,-1,0), c(0, -1.8, 0), theta = pi / 18, n = 8, col = "gray30", type = "rotation")
rgl.viewpoint(fov=0,zoom=0.8, userMatrix= rotationMatrix(-60/180*pi,1,0,0) %*% rotationMatrix(-35/180*pi,0,0,1))
# rgl.snapshot("ucl_rot.png")

# Fig.3H, ioa_h, ioa_v, in Mollweide ---------------------------------------------------

ioa_h <- matrix(NA, ncol = 1, nrow = nrow(ucl_rot_sm))
for (j in 1:nrow(ucl_rot_sm)) {
  if (sum(is.na(nb_ind[j,])) == 0) {
    dd <- vector(mode='numeric', length=2L)
    dd[1] <- acos(ucl_rot_sm[nb_ind[j,2], ] %*% ucl_rot_sm[nb_ind[j,4],])
    dd[2] <- acos(ucl_rot_sm[nb_ind[j,5], ] %*% ucl_rot_sm[nb_ind[j,7],])
    ioa_h[nb_ind[j,1]] <- mean(dd)/pi*180
  }
}

ioa_v <- matrix(NA, ncol = 1, nrow = nrow(ucl_rot_sm))
for (j in 1:nrow(ucl_rot_sm)) {
  if (sum(is.na(nb_ind[j,])) == 0) {
    ioa_v[nb_ind[j,1]] <- mean(nb_dist_ucl[nb_ind[j,1], c(3,6)], na.rm = T)/pi*180
  }
}

# - Mollweide
df_pos <- data.frame(ucl_rot_Mo)

# -- ioa_h
df_pos$quan<- ioa_h
bin_lim <- c(7, 12) #5-95%

df <- df_pos[, c('xM','yM','quan')]
df$quan[is.na(df$quan)] <- 20
colnames(df) <- c('x','y','z')
grid <- with(df, interp::interp(x, y, z))

plt <- plt_Mo34 + 
  geom_path(data = cmer, aes(x=xM, y=yM), colour = pal_axes[3], lwd=1.1) +
  geom_point(data=df_pos, aes(x = xM, y = yM, colour = quan), size = 2) +
  scale_color_gradient(low = "blue", high= "gray85", limits=bin_lim, oob=scales::squish, trans='log',
                       breaks=bin_lim, labels=bin_lim, guide = guide_colorbar(title = "angle"), na.value = 'yellow') +
  theme(legend.position = c(.9, .9) ) +
  labs(title = "ioa eye along h-axis")
windows(width = 12, height = 8)
# pdf("ioa_h.pdf", width = 8.5, height = 4.5)
plt
# dev.off()

# -- ioa_v
df_pos$quan<- ioa_v
bin_lim <- c(3, 6)

df <- df_pos[, c('xM','yM','quan')]
df$quan[is.na(df$quan)] <- 20
colnames(df) <- c('x','y','z')
grid <- with(df, interp::interp(x, y, z))

plt <- plt_Mo34 + 
  geom_path(data = cmer, aes(x=xM, y=yM), colour = pal_axes[3], lwd=1.1) +
  geom_point(data=df_pos, aes(x = xM, y = yM, colour = quan), size = 2) +
  scale_color_gradient(low = "blue", high= "gray85", limits=bin_lim, oob=scales::squish, trans='log',
                       breaks=bin_lim, labels=bin_lim, guide = guide_colorbar(title = "angle"), na.value = 'yellow') +
  theme(legend.position = c(.9, .9) ) +
  labs(title = "ioa eye along h-axis")
windows(width = 12, height = 8)
# pdf("ioa_v.pdf", width = 8.5, height = 4.5)
plt
# dev.off()

# ED Fig.4F, ED Fig.4G, ioa and shear   -------------------

fn <- c(
  '20240701',
  '20231107',
  '20240530'
)

for (f in 1:length(fn)) {
  load(paste0("data/microCT/", fn[f], ".RData"))
  
  lens_left <- lens[ind_left_lens,]
  lens_mir <- lens_left
  lens_mir[,2] <- -lens_mir[,2]
  lens_2eye <- lens
  lens_right <- lens[!ind_left_lens,]
  ind_Up_lens_right <- na.omit(match(i_match[ind_Up], rownames(lens_right)))
  ind_Up_lens_left <- na.omit(match(i_match[ind_Up], rownames(lens_left)))
  ind_Down_lens_right <- na.omit(match(i_match[ind_Down], rownames(lens_right)))
  ind_Down_lens_left <- na.omit(match(i_match[ind_Down], rownames(lens_left)))
  # re-order ucl_rot rows and re-define rownames because mapping is betw Mi1 and lens
  ucl_rot_left <- ucl_rot_sm[order(i_match),][ind_left_lens,]
  colnames(ucl_rot_left) <- c('x','y','z')
  ucl_rot_right <- ucl_rot_sm[order(i_match),][!ind_left_lens,]
  colnames(ucl_rot_right) <- c('x','y','z')
  
  for (lr in c(1,2)) {
    if (lr == 1) {
      pt <- lens_left
      ind_Up <- ind_Up_lens_left
      ind_Down <- ind_Down_lens_left
      ucl_rot <- ucl_rot_left
    } else {
      pt <- lens_right
      ind_Up <- ind_Up_lens_right
      ind_Down <- ind_Down_lens_right
      ucl_rot <- ucl_rot_right
    }
    
    hexcoord <- reghex(pt)
    ind_nb <- hexcoord[[1]]
    ind_xy <- hexcoord[[2]]
    
    # - Mollweide
    ucl_rot_tp <- cart2sphZ(ucl_rot)[,2:3] %>%
      as_tibble() %>%
      mutate(theta = theta/pi*180, phi = phi/pi*180) %>%
      mutate(phi = if_else(phi > 180, phi - 360, phi)) %>%
      mutate(phi = - phi) %>% #inside out
      as.matrix()
    ucl_rot_Mo <- Mollweide(ucl_rot_tp)
    colnames(ucl_rot_Mo) <- c('xM','yM')
    rownames(ucl_rot_Mo) <- rownames(ucl_rot)
    
    # - use ind_nb to determine nb and ioa, assume  Euclidean dist ~ arclength
    nb_dist_ucl <- matrix(nrow = nrow(ucl_rot), ncol = 7)
    for (j in 1:nrow(ucl_rot)) {
      nb_dist_ucl[ind_nb[j,1],] <- c(ind_nb[j,1],
                                     acos(ucl_rot[ind_nb[j,-1], ] %*% ucl_rot[ind_nb[j,1],]) )
    }
    ioa_hex <- cbind(ucl_rot_Mo, rowMeans(nb_dist_ucl[,2:7], na.rm = T)/pi*180)
    ioa_hex <- as.data.frame(ioa_hex)
    colnames(ioa_hex) <- c('xM','yM','ioa')
    
    # - loessm ioa
    # -- grid
    bkgd_chull <- rbind(bkgd_mer_ww, bkgd_mer_ee[seq(nrow(bkgd_mer_ee),1,by=-1),])
    colnames(bkgd_chull) <- c('xM','yM')
    grid_M <- expand.grid(xM = seq(-sqrt(8), sqrt(8), length.out = 100),
                          yM = seq(-sqrt(2), sqrt(2), length.out = 50) )
    ii_inpoly <- sp::point.in.polygon(grid_M[,1], grid_M[,2], bkgd_chull[,1], bkgd_chull[,2])
    
    # -- loess
    fit_loess <- loess(ioa ~ xM * yM, data = ioa_hex, degree = 2, span = 0.1,
                       control = loess.control(surface = "direct"))
    pred_loess <- predict(fit_loess, grid_M, se = T)
    df_pred <- grid_M
    df_pred$Z <- melt(pred_loess$fit)$value
    df_pred <- df_pred[ii_inpoly == 1, ]
    
    xy_ashape <- ashape(ucl_rot_Mo[, 1:2] +
                          matrix(runif(nrow(ucl_rot)*2, 1e-9, 2e-9), ncol=2), alpha = 0.3)
    xy_edge <- xy_ashape$edges[,1:6]
    xy_grid_ahull <- mkpoly(xy_edge)[[1]][,3:4] # hull edge points
    xy_grid_ahull <- rbind(xy_grid_ahull, xy_grid_ahull[1,])
    ii_inpoly <- sp::point.in.polygon(grid_M[ii_inpoly == 1,1], grid_M[ii_inpoly == 1,2], xy_grid_ahull[,1], xy_grid_ahull[,2])
    df_pred$Z[!ii_inpoly] <- 20 # outside set to large angles
    
    # -- plot
    breaks_contour <- c(0, 4, 5, 6)
    df <- as.data.frame(ioa_hex)
    plt <- plt_Mo +
      geom_contour(data=df_pred, aes(x=xM,y=yM, z=Z), breaks=breaks_contour,
                   colour='gray20', alpha=0.9, lwd=0.5) +
      geom_point(data = df, aes(xM, yM), colour =pal_lr[lr], size=1, alpha=1) +
      theme(legend.position = c(.9, .9) ) +
      labs(title = paste("ioa ", c("left","right")[lr], " eye_", fn[f], sep=''))
    windows(width = 16, height = 8); plt
    # pdf(paste("ioa_", c("left","right")[lr], " eye_", fn[f], ".pdf", sep=''), width = 8.5, height = 4.5)
    print(plt)
    # dev.off()
    
    # - shear
    nb_ang_ucl <- matrix(ncol = 2, nrow = nrow(ucl_rot))
    hex_hvf_eye <- matrix(ncol = 3, nrow = nrow(ucl_rot))
    for (j in 1:nrow(ind_nb)) {
      if (sum(complete.cases(ind_nb[j,])) == 7) {
        bt <- ucl_rot[ind_nb[j,3], ] - ucl_rot[ind_nb[j,6], ]
        bf <- colMeans(ucl_rot[ind_nb[j,4:5],]) - colMeans(ucl_rot[ind_nb[j,c(2,7)],])
        hex_hvf_eye[ind_nb[j,1],] <- bf
        ang <- acos(bt %*% bf / sqrt(sum(bt^2)) / sqrt(sum(bf^2)) ) /pi*180 
        nb_ang_ucl[ind_nb[j,1],] <- c(j, ang)
      }
    }
    df_pos <- data.frame(ucl_rot_Mo)
    df_pos$quan <- nb_ang_ucl[,2]
    quantile(na.omit(df_pos$quan), c(0, 0.01, 0.1, 0.5, 0.9, 0.99, 1))
    
    # -- Mollweide
    rg <- c(60, 90, 120)
    plt <- plt_Mo + 
      # geom_path(data = cmer, aes(x=xM, y=yM), colour = pal_axes[3], lwd=1) +
      geom_point(data=df_pos, aes(x = xM, y = yM, colour = quan), size = 2) +
      scale_color_gradientn(colours = pal_heat1, values = scales::rescale(rg), limits=range(rg), oob=scales::squish,
                            breaks= rg, labels= rg, guide = guide_colorbar(title = "shear") ) +
      theme(legend.position = c(.9, .9) ) +
      labs(title = "skew angle")
    windows(width = 9, height = 6)
    # pdf(paste0("skew_eye_", fn[f], "_", c('left','right')[lr], ".pdf"), width = 8.5, height = 4.5)
    print(plt)
    # dev.off()
  }
}


# ED Fig.4E, ioa along the equator -------------------------------------------

ioa_Mo <- list()
load(paste0("data/microCT/20240701.RData"))

lens_2eye <- lens
lens_left <- lens[ind_left_lens,]
lens_right <- lens[!ind_left_lens,]
ind_Up_lens_right <- na.omit(match(i_match[ind_Up], rownames(lens_right)))
ind_Up_lens_left <- na.omit(match(i_match[ind_Up], rownames(lens_left)))
ind_Down_lens_right <- na.omit(match(i_match[ind_Down], rownames(lens_right)))
ind_Down_lens_left <- na.omit(match(i_match[ind_Down], rownames(lens_left)))
# re-order ucl_rot rows and re-define rownames because mapping is betw Mi1 and lens
ucl_rot_2eye <- ucl_rot
ucl_rot_left <- ucl_rot[order(i_match),][ind_left_lens,]
rownames(ucl_rot_left) <- seq(1, nrow(ucl_rot_left))
colnames(ucl_rot_left) <- c('x','y','z')
ucl_rot_right <- ucl_rot[order(i_match),][!ind_left_lens,]
rownames(ucl_rot_right) <- seq(1, nrow(ucl_rot_right))
colnames(ucl_rot_right) <- c('x','y','z')

for (lr in c(1,2)) {
  if (lr == 1) {
    pt <- lens_left
    ind_Up <- ind_Up_lens_left
    ind_Down <- ind_Down_lens_left
    ucl_rot <- ucl_rot_left
    hexcoord <- reghex(pt, lefteye = T)
  } else {
    pt <- lens_right
    ind_Up <- ind_Up_lens_right
    ind_Down <- ind_Down_lens_right
    ucl_rot <- ucl_rot_right
    hexcoord <- reghex(pt, lefteye = F)
  }
  ind_nb <- hexcoord[[1]]
  ind_xy <- hexcoord[[2]]
  
  # - Mollweide
  ucl_rot_Mo <- ucl_rot
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
  rownames(ucl_rot_Mo) <- rownames(ucl_rot)
  
  # - use ind_nb to determine nb and ioa, assume  Euclidean dist ~ arclength
  nb_dist_ucl <- matrix(nrow = nrow(ucl_rot), ncol = 7)
  for (j in 1:nrow(ucl_rot)) {
    nb_dist_ucl[ind_nb[j,1],] <- c(
      ind_nb[j,1],acos(ucl_rot[ind_nb[j,-1], ] %*% ucl_rot[ind_nb[j,1],])
    )
  }
  ioa_hex <- cbind(ucl_rot_Mo, rowMeans(nb_dist_ucl[,2:7], na.rm = T)/pi*180)
  ioa_hex <- as.data.frame(ioa_hex)
  colnames(ioa_hex) <- c('xM','yM','ioa')
  
  ioa_Mo[[lr]] <- ioa_hex
}

delev <- 15 # +/-15 deg in Mollweide
dMx <-  Mollweide(rbind(c(90,0), c(90,5)))[2,1] # 5 deg in Mollweide
eqbw <-  Mollweide(rbind(c(90,0), c(90,delev)))[2,1] 

lr <- 1
xya <- ioa_Mo[[lr]]
xya <- xya[abs(xya$yM) < eqbw, ] 
df_l <- data.frame(x=xya$xM, y=xya$ioa)

lr <- 2
xya <- ioa_Mo[[lr]]
xya <- xya[abs(xya$yM) < eqbw, ]
df_r <- data.frame(x=xya$xM, y=xya$ioa)

xazim <- seq(0, sqrt(8), by = dMx*12)
xazim <- c(-rev(xazim[-1]), xazim)

windows(width =7.5, height = 6)
# pdf(paste0("ioa_equ_", delev, "deg.pdf"))
ggplot() +
  geom_point(data=df_l, aes(x,y),size=2, col=pal_lr[1]) +
  geom_smooth(data=df_l, aes(x,y), se=F, col='blue', lwd=1) +
  geom_point(data=df_r, aes(x,y),size=2, col=pal_lr[2]) +
  geom_smooth(data=df_r, aes(x,y), se=F, col='black', lwd=1) +
  scale_x_continuous(limits = c(-sqrt(8), sqrt(8)), breaks =xazim, labels =paste0(xazim/dMx/6*30,"°"), expand = c(0, 0)) +
  scale_y_continuous(limits = c(3, 9), breaks = seq(3,10), labels =paste0(seq(3,10),"°"), expand = c(0, 0)) +
  theme_minimal() +
  ylab("inter-ommatidium angle [deg]") +
  xlab("azimuth [deg]") +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14) ) +
  labs(title = paste("ioa +/-", delev, fn)) +
  coord_fixed(ratio = 0.5)
# dev.off()

