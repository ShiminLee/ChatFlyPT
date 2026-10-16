# Fig.1C, T4 eg -------------------------------------------------------------------

# - coord transform via pca
tar <- T4_eg[[2]]
ind_D = match(tar$tags$"dendrite start" , tar$d$PointNo)
node_dend <- child_node(tar, tar$d[ind_D,])
cen <- colMeans(xyzmatrix(node_dend)) 
cen_ii <- rowSums(sweep(Mi1_M10_xyz, 2, cen)^2) %>% order(decreasing=F) %>% head(1+6+12+18)

node_xyz <- Mi1_M10_xyz[cen_ii,]
me_pca <- prcomp(node_xyz)
if (me_pca$rotation[,3] %*% c(326665, 259455, 235136) > 0) {
  me_pca$rotation[,3] <- - me_pca$rotation[,3]
}
if (me_pca$rotation[,1] %*% c(-19607, -44222, 12645) < 0) {
  me_pca$rotation[,1] <- - me_pca$rotation[,1]
}
if (t(cross3D(me_pca$rotation[,3],me_pca$rotation[,1])) %*% me_pca$rotation[,2] < 0 ) {
  me_pca$rotation[,2] <- - me_pca$rotation[,2]
}

# - transform LOP mesh
LOP_msh_xform <- LOP_msh_mod
LOP_msh_xform$vb[1:3,] <- sweep(t(LOP_msh_mod$vb[1:3,]), 2, me_pca$center) %*% me_pca$rotation %>% t()

# - use T4_eg to make new local mesh 
T4_eg_LOP <- nlapply(T4_eg, subset, function(x) pointsinside(x, LOP_msh_mod))

# pc transform
lop_pca <- prcomp(xyzmatrix(T4_eg_LOP))

T4_eg_xformLOP <- xEucl_neu(T4_eg, lop_pca$rotation, lop_pca$center)

# T4 without soma tract
T4_eg_xformLOP_nosoma <- T4_eg_xformLOP
for (j in 1:length(T4_eg_xformLOP_nosoma)) {
  tar <- T4_eg_xformLOP_nosoma[[j]]
  ng <- as.ngraph(tar)
  ii <- match(tar$tags$`SAD junction`, tar$d$PointNo)
  distal_points <- igraph::graph.dfs(ng, root = ii, unreachable=FALSE, neimode='out')$order
  T4_eg_xformLOP_nosoma[[j]]<- subset(tar, distal_points)
}

loplayer <- list()
for (LL in 1:4) {
  xyz <- as.matrix(xyz_layer_T4[[LL]]) # points on lop layer
  xyz <- sweep(xyz,2,lop_pca$center) %*% lop_pca$rotation # transform
  dd <- xyz[,2:3]^2 %>% rowSums() %>% sqrt() #distance to origin
  xyz <-  xyz[dd < 20000, ]
  yz <- data.frame(y = 0, z = seq(-10000, 10000, length.out = 20) )
  x <- xyz[,1]; y <- xyz[,2]; z <- xyz[,3]
  fitlm <- lm(x ~ poly(y, z, degree = 2, raw = T))
  valfit <- predict(fitlm, yz) #generate values from the fit
  loplayer[[LL]] <- cbind(valfit, yz)
}

# collect all points on layers
pp <- list() 
pp[[1]] <- as.matrix(loplayer[[1]] - loplayer[[2]] + loplayer[[1]])
pp[[2]] <- as.matrix((loplayer[[1]] + loplayer[[2]])/2)
pp[[3]] <- as.matrix((loplayer[[2]] + loplayer[[3]])/2)
pp[[4]] <- as.matrix((loplayer[[3]] + loplayer[[4]])/2)
pp[[5]] <- as.matrix(loplayer[[4]] - loplayer[[3]] + loplayer[[4]])

# make layer for plotting
layers_LOP <- matrix(ncol = 3, nrow = 0)
for (j in 1:5) {
  cc <- pp[[j]]
  c2 <- cbind(cc[-nrow(cc),], cc[-1,]) %>%
    t() %>%
    matrix(., ncol = 3, byrow = T)
  layers_LOP <- rbind(layers_LOP, c2)
}
layers_LOP_mepca <- sweep(sweep(layers_LOP %*% t(lop_pca$rotation),2,lop_pca$center,'+'),2,me_pca$center) %*% me_pca$rotation

# - PLOT T4 lop layer
nopen3d()
par3d('windowRect' = c(100,100,1300,1300))
plot3d(T4_eg_xformLOP_nosoma, lwd=2, soma=T, col=pal_T4)
segments3d(layers_LOP, lwd=1)
rgl.viewpoint(fov=0,zoom=0.8, 
              userMatrix= rotationMatrix(-90/180*pi,1,0,0) %*% rotationMatrix(-30/180*pi,0,0,1) )
segments3d(sweep(rbind(
  c(0, 0, 0),
  rotate3d(rotate3d(c(1000, 0, 0), -90/180*pi, 1,0,0), -30/180*pi, 0,0,1)
  ), 2, c(3e4,0e5,0e5), '+'), lwd=2)
# rgl.snapshot(filename = paste("T4_eg.png", sep = ''))


# Fig.1C, plot T4 dendrite with SN ------------------------------------------

# T4 dendrite within ME
T4_eg_ME <- nlapply(T4_eg, subset, function(x) pointsinside(x, ME_msh,rval='distance')>-2000)
# pc transform
me_pca <- prcomp(xyzmatrix(T4_eg_ME))

subtree <- T4_eg
root_xyz <- matrix(ncol = 3, nrow = 4)
for (j in 1:4) {
  tar <- T4_eg[[j]]
  ind_D = match(tar$tags$`dendrite start` , tar$d$PointNo)
  root_xyz[j,] <- xyzmatrix(tar$d[ind_D,])
  # - subtree with root = dendrite start 
  targ <- as.ngraph(tar)
  ii_root <- ind_D
  # subtree 
  sub_points <- igraph::graph.dfs(targ, root = ii_root, unreachable=FALSE, neimode='out')$order
  subtree[[j]] <- subset(tar, sub_points) 
}
subtree_xformME <- xEucl_neu(subtree, me_pca$rotation, me_pca$center)
root_xyz_xformME <- sweep(root_xyz, 2, me_pca$center) %*% me_pca$rotation

## ## choose a type
j <- 1
tar <- subtree_xformME[[j]]
ind_D = match(-1, tar$d$Parent)
df_D <-  tar$d[ind_D,]
root_xyz <- tar$d[ind_D, c("X","Y","Z")]

# - find the subtree with root = dendrite start
targ <- as.ngraph(tar)
ii_root <- ind_D
# subtree and Strahler order
sub_points <- igraph::graph.dfs(targ, root = ii_root, unreachable=FALSE, neimode='out')$order
subtree <- subset(tar, sub_points)
subtree_g <- as.ngraph(subtree, weights = T)
subtree_so <- strahler_order(subtree) # Strahler order
max(subtree_so$segments)

nopen3d()
par3d('windowRect' = c(100,100,1300,1300))
for (i in 1:max(subtree_so$points)) {
  pt_p <- tar$d[,"PointNo"] %in% unique(tar$d[subtree_so$points==i,"Parent"])
  pt_so <- pt_p | subtree_so$points==i
  plot3d(subset(tar, pt_so), col=pal_T4[j], add = i!=1, boundingbox = boundingbox(tar), lwd = i+1)
}
segments3d(rbind(c(-10000, -5000, 0), c(-10000,-4000,0)), lwd=2) #1um
rgl.viewpoint(fov=0,zoom=1, userMatrix= rotationMatrix(-90/180*pi,0,0,1) %*% rotationMatrix(180/180*pi,0,1,0) )
# rgl.snapshot(filename = paste("T4_eg_dend_", j, "_.png", sep = ''))


# Fig.1D, H2 and T4 ---------------------------------------------------------------

tar <- H2
ind_D = match(tar$tags$`LOP start` , tar$d$PointNo)
df_D <-  tar$d[ind_D,]
# - subtree 
targ <- as.ngraph(tar)
ii_root <- ind_D
sub_points <- igraph::graph.dfs(targ, root = ii_root, unreachable=FALSE, neimode='out')$order
subtree <- subset(tar, sub_points) 

pc <- prcomp(xyzmatrix(subtree$d))
vn <- pc$rotation[,1]
vn <- vn/sqrt(sum(vn^2))
vn %*% c(3e5,1e5,2e5)

H2_prune <- subset(H2, vn %*% t(xyzmatrix(H2$d)) < -2.7e5 & vn %*% t(xyzmatrix(H2$d)) > -3.4e5)

# T4
H2T4 <- T4_dend[[2]][c(153, 140, 25, 89, 149, 24, 21, 20)]

# prune two of these
tar <- H2T4[[4]]
ind_D = match(tar$tags$`SAD junction` , tar$d$PointNo)
ii <- distal_to(tar, node.idx = ind_D)
H2T4[[4]] <- subset(tar, ii) 

tar <- H2T4[[7]]
ind_D = match(tar$tags$`SAD junction` , tar$d$PointNo)
ii <- distal_to(tar, node.idx = ind_D)
H2T4[[7]] <- subset(tar, ii)

# PLOT
nopen3d()
par3d('windowRect' = c(100,100,1000,700))
plot3d(H2_prune, col='gray40', soma = T, WithNodes = F, lwd=1, alpha=0.5, lit=F)
plot3d(H2T4, soma=T, lwd=1, col=pal_T4[2])

rgl.viewpoint(fov=0,zoom=0.6, userMatrix= rotationMatrix(+20/180*pi,0,1,0) %*%
                rotationMatrix(-80/180*pi,1,0,0) )
segments3d(sweep(rbind(c(0, 0, 0),
                       rotate3d(rotate3d(c(10000, 0, 0), 20/180*pi, 0,1,0), -80/180*pi, 1,0,0)
                       ), 2, c(3e5,1e5,2e5), '+'), lwd=2)

# Fig.1F, ideal flow field vs H2 -----------------------------------------------------

# - H2
# combine arenaAng and headAnd
uxy <- unique(tb[, c('stimPosX', 'stimPosY')])

# compute PD in eye ref
xyz_add <- matrix(ncol = 3, nrow = 0)
sd_3d <- matrix(ncol = 3, nrow = 0)
df <- matrix(ncol = 6, nrow = 0)
for (j in 1:nrow(uxy)) {
  for (k in c(0,1)) {
    ii <- tb$stimPosX == uxy$stimPosX[j] &
      tb$stimPosY == uxy$stimPosY[j] &
      tb$edgeVal == k
    
    thetaphi <- tb_v[ii, 1:2] #base
    
    if (sum(is.na(thetaphi[,1])) < nrow(thetaphi)) {
      thetaphi[, 1] <- 90 - thetaphi[, 1] # elev to theta
      thetaphi[, 2] <- 360 - thetaphi[, 2] # left to right, need it cos cart2Merc invert y
      thetaphi <- thetaphi / 180*pi
      thetaphi <- cbind(1, thetaphi)
      xyz <- matrix(ncol = 3, nrow = nrow(thetaphi))
      xyz[!is.na(thetaphi[,3]), ] <- sph2cartZ(thetaphi[!is.na(thetaphi[,3]), ])
      
      # stim position
      pxyz <- xyz
      xyz_add <- rbind(xyz_add, colMeans(xyz[!is.na(xyz[,1]),]))
      sd_3d <- rbind(sd_3d, apply(pxyz, 2, sd))
      
      # spk, both dark and bright
      thetaphi <- tb_v[ii,3:4]
      
      thetaphi[, 1] <- 90 - thetaphi[, 1]
      thetaphi[, 2] <- 360 - thetaphi[, 2] # left to right, need it cos cart2Merc invert y
      thetaphi <- thetaphi / 180*pi
      thetaphi <- cbind(1, thetaphi)
      xyz <- matrix(ncol = 3, nrow = nrow(thetaphi))
      xyz[!is.na(thetaphi[,3]), ] <- sph2cartZ(thetaphi[!is.na(thetaphi[,3]), ])
      
      # stim dir
      dxyz <- xyz
      
      # mean position
      pxyz_mean <- colMeans(pxyz)
      pxyz_mean <- pxyz_mean / sqrt(sum(pxyz_mean^2))
      pxyz_mean <- matrix(pxyz_mean, ncol = 3)
      # mean arrow head position
      dxyz_mean <- colMeans(dxyz - pxyz) + pxyz_mean
      dxyz_mean <- dxyz_mean / sqrt(sum(dxyz_mean^2))
      dxyz_mean <- matrix(dxyz_mean, ncol = 3)
      
      df_tmp <- c(unlist(cart2Mercator(pxyz_mean)),
                  unlist(cart2Mercator(dxyz_mean)),
                  as.integer(paste0(uxy$stimPosX[j],uxy$stimPosY[j])),
                  k)
      df <- rbind(df, df_tmp)
    }
  }
}
colnames(df) <- c('x','y','xend','yend','pos', 'edge')

# # variation in position
# sd_3d^2 %>% rowSums() %>% sqrt() %>% mean()

# positions for artificial flow field
xyz_add <- data.frame(xyz_add)
colnames(xyz_add) <- c('x','y','z')
xyz_add <- unique(xyz_add[, c('x', 'y', 'z')])
xyz_add <- as.matrix(xyz_add)

# combine bright/dark
df_arrow <- data.frame(df) %>%
  group_by(pos) %>%
  summarise(x=first(x),y=first(y),xend=mean(xend),yend=mean(yend),edge=max(edge)) %>%
  ungroup() %>%
  as.data.frame()

df_arrow$edge <- as.factor(df_arrow$edge)
df_arrow_H2 <- df_arrow

# - ideal optic flow fields
# initial positions
dA <- 20
pt0 <- matrix(ncol = 3) #pts on screen
for (j in seq(0,90,by = dA)) {
  for (k in seq(0,180*1,length.out = j %/% (dA/2)  *1 + 1)) {
    pt0 <- rbind(pt0, c(sin(j/180*pi)*cos((k-90)/180*pi), sin(j/180*pi)*sin((k-90)/180*pi), cos(j/180*pi)))
  }
}
pt0 <- pt0[-1,]

pt0m <- pt0 #flip the points
pt0m[,3] <- -pt0m[,3]
pt0 <- unique(rbind(pt0, pt0m)) # full hemisphere
pt0 <- pt0 %*% matrix(c(cos(-pi/2), sin(-pi/2), 0,
                        -sin(-pi/2), cos(-pi/2), 0,
                        0, 0, 1), ncol = 3, byrow = T)

# full sphere
pt0b <- pt0
pt0b[,2] <- -pt0b[,2]
pt0 <- unique(rbind(pt0, pt0b))
pt_t0 <- as.matrix(pt0) #pts at t=0 on screen

# -- add H2 locations
pt_t0 <- rbind(pt_t0,xyz_add)

# -- translation
# distance to objects
R <- 1000 
# translation
dx <- -200
dy <- 0
dz <- 0

# new locations
pt_t1 <- pt_t0 * R #proj out
pt_t1 <- sweep(pt_t1, MARGIN = 2, STATS = c(dx, dy, dz), FUN = '-') #move
pt_t1 <- sweep(pt_t1, MARGIN = 1, STATS = sqrt(rowSums(pt_t1^2)), FUN = '/') #proj on unit sphere

# Tangent
for (j in 1:dim(pt_t0)[1]) {
  pt_t1[j,] <- pt_t1[j,] / c(pt_t0[j,] %*% pt_t1[j,])
}
pt_t1 <-  (pt_t1 - pt_t0) + pt_t0

# noise
pt_t0 <- pt_t0 + matrix(runif(dim(pt_t0)[1]*3, 1e-9, 2e-9), ncol = 3) #noise for drawing vectors
pt_t1 <- pt_t1 + matrix(runif(dim(pt_t1)[1]*3, 1e-9, 2e-9), ncol = 3) #noise for drawing vectors


# -- Mercator
ii <- 1 - abs(pt_t0[,3]) > 0.01 & 1 - abs(pt_t1[,3]) > 0.01
tp <- cart2sphZ(pt_t0)[,2:3]
ii <- ii & abs(pi - tp[,2]) > 0.1
pt0_Mer <- cart2Mercator(pt_t0[ii,])
pt1 <- sweep(pt_t1, 1, sqrt(rowSums(pt_t1^2)), '/')
pt1_Mer <- cart2Mercator(pt1[ii,])

df_arrow <- data.frame(cbind(pt0_Mer, pt1_Mer) )
colnames(df_arrow) <- c('x','y','xend','yend')

plt <- plt_Mer +
  geom_segment(data=df_arrow, aes(x = x,y = y, xend = xend,yend = yend), colour='deepskyblue',size =1) +
  geom_segment(data=df_arrow_H2, aes(x = x,y = y, xend = xend,yend = yend), colour='red',size =1) +
  scale_x_continuous(limits = c(-15,95)/180*pi,breaks = seq(-180,180,by=45)/180*pi, labels = paste0(seq(-180,180,by=45), "°"),  expand = c(0, 0)) +
  scale_y_continuous(limits = c(-20,70)/180*pi,breaks = log(tan(pi/4 + seq(-75,75,by=15)/180*pi/2)), labels = paste0(seq(-75,75,by=15),"°"),  expand = c(0, 0)) +
  labs(title = "")
windows(width = 6.5, height = 6.5)
plt
# ggsave(paste("H2_trans_Mercator.pdf",sep=''), width = 6.5, height = 6.5)


# -- yaw
ang <- -5.5*2/180*pi
Rz <- matrix(c(cos(ang), -sin(ang), 0, sin(ang), cos(ang), 0, 0, 0, 1), ncol = 3)
# new locations
pt_t1 <- t(Rz %*% t(pt_t0))

# Tangent
for (j in 1:dim(pt_t0)[1]) {
  pt_t1[j,] <- pt_t1[j,] / c(pt_t0[j,] %*% pt_t1[j,])
}
pt_t1 <-  (pt_t1 - pt_t0) + pt_t0
#noise for drawing vectors
pt_t0 <- pt_t0 + matrix(runif(dim(pt_t0)[1]*3, 1e-9, 2e-9), ncol = 3) 
pt_t1 <- pt_t1 + matrix(runif(dim(pt_t1)[1]*3, 1e-9, 2e-9), ncol = 3) 

# -- Mercator
ii <- 1 - abs(pt_t0[,3]) > 0.01 & 1 - abs(pt_t1[,3]) > 0.01
tp <- cart2sphZ(pt_t0)[,2:3]
ii <- ii & abs(pi - tp[,2]) > 0.1
pt0_Mer <- cart2Mercator(pt_t0[ii,])
pt1 <- sweep(pt_t1, 1, sqrt(rowSums(pt_t1^2)), '/')
pt1_Mer <- cart2Mercator(pt1[ii,])

df_arrow <- data.frame(cbind(pt0_Mer, pt1_Mer) )
colnames(df_arrow) <- c('x','y','xend','yend')

plt <- plt_Mer +
  geom_segment(data=df_arrow, aes(x = x,y = y, xend = xend,yend = yend), colour='deepskyblue',size =1) +
  geom_segment(data=df_arrow_H2, aes(x = x,y = y, xend = xend,yend = yend), colour='red',size =1) +
  scale_x_continuous(limits = c(-15,95)/180*pi,breaks = seq(-180,180,by=45)/180*pi, labels = paste0(seq(-180,180,by=45), "°"),  expand = c(0, 0)) +
  scale_y_continuous(limits = c(-20,70)/180*pi,breaks = log(tan(pi/4 + seq(-75,75,by=15)/180*pi/2)), labels = paste0(seq(-75,75,by=15),"°"),  expand = c(0, 0)) +
  labs(title = "")
windows(width = 6.5, height = 6.5)
plt
# ggsave(paste("H2_rot_Mercator.pdf",sep=''), width = 6.5, height = 6.5)

