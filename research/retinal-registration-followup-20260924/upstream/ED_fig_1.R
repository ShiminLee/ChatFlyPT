# ED Fig.1A, ideal optic flows --------------------------------------------------

pt0 <- matrix(ncol = 3) #pts on screen
for (j in seq(0,90,by = 10)) {
  for (k in seq(0,180*1,length.out = j %/% 10  *1 + 1)) {
    pt0 <- rbind(pt0, c(sin(j/180*pi)*cos((k-90)/180*pi), sin(j/180*pi)*sin((k-90)/180*pi), cos(j/180*pi)))
  }
}
pt0 <- pt0[-1,] #flip the points
pt0m <- pt0 
pt0m[,3] <- -pt0m[,3]
pt0 <- unique(rbind(pt0, pt0m)) # full hemisphere
pt0 <- pt0 %*% matrix(c(cos(-pi/2), sin(-pi/2), 0,
                        -sin(-pi/2), cos(-pi/2), 0,
                        0, 0, 1), ncol = 3, byrow = T)
pt_t0 <- as.matrix(pt0) #pts at t=0 on screen
colnames(pt_t0) <- c("x","y","z")

# distance to objects
R <- 1000 

# translation
dx <- 0
dy <- 0
dz <- 1
# new locations
pt_t1 <- pt_t0 * R #proj out
pt_t1 <- sweep(pt_t1, MARGIN = 2, STATS = c(dx, dy, dz), FUN = '-') #move
pt_t1 <- sweep(pt_t1, MARGIN = 1, STATS = sqrt(rowSums(pt_t1^2)), FUN = '/') #proj on unit sphere
# Tangent
for (j in 1:dim(pt_t0)[1]) {
  pt_t1[j,] <- pt_t1[j,] / c(pt_t0[j,] %*% pt_t1[j,])
}
pt_t1 <-  (pt_t1 - pt_t0) * 100 + pt_t0

# noise
pt_t0 <- pt_t0 + matrix(runif(dim(pt_t0)[1]*3, 1e-9, 2e-9), ncol = 3) #noise for drawing vectors
pt_t1 <- pt_t1 + matrix(runif(dim(pt_t1)[1]*3, 1e-9, 2e-9), ncol = 3) #noise for drawing vectors

# PLOT
nopen3d()
par3d('windowRect' = c(100,100,1100,1100))
arrow3d(c(0,0,0), c(0, 0, 1.2), lit=F, theta=pi/36, n=16, s=0.1,width=0.3, col = "gray50", type = "rotation")
arrow3d(c(0,0,0), c(1.2, 0, 0), lit=F,theta=pi/36, n=16, s=0.1,width=0.3, col = "gray50", type = "rotation")
for (j in 1:dim(pt_t0)[1]) {
  arrow3d(pt_t0[j,], pt_t1[j,], lit=T, width=0.5, theta= pi/6, n= 4, col='blue', type= "rotation")
}

# translation
dx <- 1
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
pt_t1 <-  (pt_t1 - pt_t0) * 100 + pt_t0

# noise
pt_t0 <- pt_t0 + matrix(runif(dim(pt_t0)[1]*3, 1e-9, 2e-9), ncol = 3) #noise for drawing vectors
pt_t1 <- pt_t1 + matrix(runif(dim(pt_t1)[1]*3, 1e-9, 2e-9), ncol = 3) #noise for drawing vectors

# PLOT cont.
for (j in 1:dim(pt_t0)[1]) {
  arrow3d(pt_t0[j,], pt_t1[j,], lit=T, width=0.5, theta= pi/6, n= 4, col='red', type= "rotation")
}
rgl.viewpoint(fov=0,zoom=0.71, userMatrix= rotationMatrix(-90/180*pi,1,0,0) %*%
                rotationMatrix(-17/180*pi,0,0,1) %*%
                rotationMatrix(7/180*pi,1,0,0))
# rgl.snapshot(filename = paste("ideal_flow_tran.png", sep = ''))

# cont. rotation ----------------------------------------------------------------

# - yaw 
ang <- 7/180*pi
Rz <- matrix(c(cos(ang), -sin(ang), 0, sin(ang), cos(ang), 0, 0, 0, 1), ncol = 3)

# new locations
pt_t1 <- t(Rz %*% t(pt_t0))

# Tangent
for (j in 1:dim(pt_t0)[1]) {
  pt_t1[j,] <- pt_t1[j,] / c(pt_t0[j,] %*% pt_t1[j,])
}
pt_t0 <- pt_t0 + matrix(runif(dim(pt_t0)[1]*3, 1e-9, 2e-9), ncol = 3) #noise for drawing vectors
pt_t1 <- pt_t1 + matrix(runif(dim(pt_t1)[1]*3, 1e-9, 2e-9), ncol = 3) #noise for drawing vectors

# PLOT
nopen3d()
par3d('windowRect' = c(100,100,1100,1100))
arrow3d(c(0,0,0), c(0, 0, 1.2), lit=F, theta=pi/36, n=16, s=0.1,width=0.3, col = "gray50", type = "rotation")
arrow3d(c(0,0,0), c(1.2, 0, 0), lit=F,theta=pi/36, n=16, s=0.1,width=0.3, col = "gray50", type = "rotation")
for (j in 1:dim(pt_t0)[1]) {
  arrow3d(pt_t0[j,], pt_t1[j,],lit=T, width=0.5, theta= pi/6, n= 4, col='red', type= "rotation")
}

# - roll
ang <- -7/180*pi
Rx <- matrix(c(1, 0, 0, 0, cos(ang), -sin(ang), 0, sin(ang), cos(ang)), ncol = 3)

# new locations
pt_t1 <- t(Rx %*% t(pt_t0))

# Tangent
for (j in 1:dim(pt_t0)[1]) {
  pt_t1[j,] <- pt_t1[j,] / c(pt_t0[j,] %*% pt_t1[j,])
}

pt_t0 <- pt_t0 + matrix(runif(dim(pt_t0)[1]*3, 1e-9, 2e-9), ncol = 3) #noise for drawing vectors
pt_t1 <- pt_t1 + matrix(runif(dim(pt_t1)[1]*3, 1e-9, 2e-9), ncol = 3) #noise for drawing vectors

# PLOT cont.
for (j in 1:dim(pt_t0)[1]) {
  arrow3d(pt_t0[j,], pt_t1[j,],lit=T, width=0.5, theta= pi/6, n= 4, col='blue', type= "rotation")
}
rgl.viewpoint(fov=0,zoom=0.71, userMatrix= rotationMatrix(-90/180*pi,1,0,0) %*%
                rotationMatrix(-17/180*pi,0,0,1) %*%
                rotationMatrix(7/180*pi,1,0,0))
# rgl.snapshot(filename = paste("ideal_flow_rot.png", sep = ''))

# ED Fig.1B, H2 reconstruction ------------------------------------------------

nopen3d()
par3d('windowRect' = c(100,100,1000,720))
plot3d(H2, col='black', soma = T, WithNodes = F, lwd=2, alpha=0.5, lit=F)
shade3d(LOP_msh_mod, col='gray', alpha= 0.1, lit=F)

rgl.viewpoint(fov=0,zoom=0.6, userMatrix= rotationMatrix(-170/180*pi,1,0,0) %*%
                rotationMatrix(0/180*pi,0,0,1) )
segments3d(sweep(rbind(c(0, 0, 0),
                       rotate3d(rotate3d(c(10000, 0, 0), -170/180*pi, 1,0,0), 0/180*pi,0,0,1)
), 2, c(3e5,3.4e5,2e5), '+'), lwd=2)

rgl.viewpoint(fov=0,zoom=0.6, userMatrix= rotationMatrix(+20/180*pi,0,1,0) %*%
                rotationMatrix(-180/180*pi,1,0,0) )

# ED Fig.1D, combine all grating data, avg ------------------------------------

load('data/H2_tuning.rda' )

uxy <- unique(tb[, c('stimPosX', 'stimPosY')])

df <- matrix(ncol = 5, nrow = 0)
for (j in 1:nrow(uxy)) {
  ii <- tb$stimPosX == uxy$stimPosX[j] &
    tb$stimPosY == uxy$stimPosY[j] &
    !is.na(tb$headAng)
  
  thetaphi <- tb_v[ii, 1:2, drop=F] #base
  
  if (sum(is.na(thetaphi[,1])) < nrow(thetaphi)) {
    thetaphi[, 1] <- 90 - thetaphi[, 1] # elev to theta
    thetaphi[, 2] <- 360 - thetaphi[, 2] # left to right, because cart2Merc invert y
    thetaphi <- thetaphi / 180*pi
    thetaphi <- cbind(1, thetaphi)
    xyz <- matrix(ncol = 3, nrow = nrow(thetaphi))
    xyz[!is.na(thetaphi[,3]), ] <- sph2cartZ(thetaphi[!is.na(thetaphi[,3]), ])
    
    pxyz <- xyz
    
    thetaphi <- tb_v[ii,3:4] #spk
    
    thetaphi[, 1] <- 90 - thetaphi[, 1]
    thetaphi[, 2] <- 360 - thetaphi[, 2] # left to right, need it cos cart2Merc invert y
    thetaphi <- thetaphi / 180*pi
    thetaphi <- cbind(1, thetaphi)
    xyz <- matrix(ncol = 3, nrow = nrow(thetaphi))
    xyz[!is.na(thetaphi[,3]), ] <- sph2cartZ(thetaphi[!is.na(thetaphi[,3]), ])
    
    dxyz <- xyz
    
    df_tmp <- c(colMeans(cart2Mercator(pxyz)), 
                colMeans(cart2Mercator(dxyz)),
                as.integer(paste0(uxy$stimPosX[j],uxy$stimPosY[j]))
    )
  }
  df <- rbind(df, df_tmp)
}
colnames(df) <- c('x','y','xend','yend','pos')
df5 <- df


# - new data
load('data/H2_tuning_2023_grating.rda' )

uxy <- unique(tb[, c('stimPosX', 'stimPosY')])

df <- matrix(ncol = 5, nrow = 0)
for (j in 1:nrow(uxy)) {
  ii <- tb$stimPosX == uxy$stimPosX[j] &
    tb$stimPosY == uxy$stimPosY[j]
  
  thetaphi <- tb_v[ii, 1:2, drop=F] #base
  
  if (sum(is.na(thetaphi[,1])) < nrow(thetaphi)) {
    thetaphi[, 1] <- 90 - thetaphi[, 1] # elev to theta
    thetaphi[, 2] <- 360 - thetaphi[, 2] # left to right
    thetaphi <- thetaphi / 180*pi
    thetaphi <- cbind(1, thetaphi)
    xyz <- matrix(ncol = 3, nrow = nrow(thetaphi))
    xyz[!is.na(thetaphi[,3]), ] <- sph2cartZ(thetaphi[!is.na(thetaphi[,3]), ])
    
    pxyz <- xyz
    
    thetaphi <- tb_v[ii,3:4] #spk
    
    thetaphi[, 1] <- 90 - thetaphi[, 1]
    thetaphi[, 2] <- 360 - thetaphi[, 2] # left to right, need it cos cart2Merc invert y
    thetaphi <- thetaphi / 180*pi
    thetaphi <- cbind(1, thetaphi)
    xyz <- matrix(ncol = 3, nrow = nrow(thetaphi))
    xyz[!is.na(thetaphi[,3]), ] <- sph2cartZ(thetaphi[!is.na(thetaphi[,3]), ])
    
    dxyz <- xyz
    
    df_tmp <- c(colMeans(cart2Mercator(pxyz)),
                colMeans(cart2Mercator(dxyz)),
                as.integer(paste0(uxy$stimPosX[j],uxy$stimPosY[j]))
    )
  }
  df <- rbind(df, df_tmp)
}
colnames(df) <- c('x','y','xend','yend','pos')


# PLOT
df <- rbind(df, df5)
# combine same position
df_arrow <- as.data.frame(df) %>%
  group_by(pos) %>%
  summarise(x=mean(x),
            y=mean(y),
            xend = mean(xend),
            yend = mean(yend),
            pos = mean(pos)) %>%
  as.data.frame()
df_arrow$pos <- as.factor(df_arrow$pos)

# mercator
plt <- plt_Mer +
  geom_segment(data=df_arrow, 
               aes(x = x,y = y, xend = xend,yend = yend,colour=pos),size =1)+
  scale_colour_manual(values = brewer.pal(10,"Paired"), guide="none") +
  scale_x_continuous(limits = c(-15,95)/180*pi,breaks = seq(-180,180,by=45)/180*pi, labels = paste0(seq(-180,180,by=45), "°"),  expand = c(0, 0)) +
  scale_y_continuous(limits = c(-20,70)/180*pi,breaks = log(tan(pi/4 + seq(-75,75,by=15)/180*pi/2)), labels = paste0(seq(-75,75,by=15),"°"),  expand = c(0, 0)) +
  labs(title = "grating - spk - all")
windows(width = 6.5, height = 6.5)
plt
# ggsave(paste("H2_all_grating_avg.pdf",sep=''), width = 9, height = 8)

# ED Fig. 1D, plot all 7 cells, edge ---------------------------------------------------

load('data/H2_tuning_2023.rda' )

pos <- paste0(tb$stimPosX, tb$stimPosY) %>% as.integer()
upos <- unique(pos)

df <- matrix(ncol = 5, nrow = 0)
for (j in 1:7) {
  for (k in c(0,1)) {
    
    ii <- tb$cellNum == j & tb$edgeVal == k
    
    thetaphi <- tb_v[ii, 1:2] #base
    
    if (sum(is.na(thetaphi[,1])) < nrow(thetaphi)) {
      thetaphi[, 1] <- 90 - thetaphi[, 1] # elev to theta
      thetaphi[, 2] <- 360 - thetaphi[, 2] # left to right, need it cos cart2Merc invert y
      thetaphi <- thetaphi / 180*pi
      thetaphi <- cbind(1, thetaphi)
      xyz <- matrix(ncol = 3, nrow = nrow(thetaphi))
      xyz[!is.na(thetaphi[,3]), ] <- sph2cartZ(thetaphi[!is.na(thetaphi[,3]), ])
      
      pxyz <- xyz
      
      thetaphi <- tb_v[ii,3:4] #spk
      
      thetaphi[, 1] <- 90 - thetaphi[, 1]
      thetaphi[, 2] <- 360 - thetaphi[, 2] # left to right, need it cos cart2Merc invert y
      thetaphi <- thetaphi / 180*pi
      thetaphi <- cbind(1, thetaphi)
      xyz <- matrix(ncol = 3, nrow = nrow(thetaphi))
      xyz[!is.na(thetaphi[,3]), ] <- sph2cartZ(thetaphi[!is.na(thetaphi[,3]), ])
      
      dxyz <- xyz
      
      df_tmp <- cbind(cart2Mercator(pxyz), cart2Mercator(dxyz), j, k, pos[ii])
    }
    df <- rbind(df, df_tmp[!is.na(df_tmp[,1]),])
  }
}
colnames(df) <- c('x','y','xend','yend','cell', 'edge', 'Pos')

# avg position
df %<>% group_by(Pos) %>% 
  mutate(x = mean(x), y = mean(y), xend = xend, yend = yend) %>%
  ungroup() %>% 
  arrange(Pos) %>% 
  data.frame()


## ### dark
df_arrow <- as.data.frame(df[df$edge == 0,])
# ### ### bright
df_arrow <- as.data.frame(df[df$edge == 1,])
# ### ### birght x1
# df_arrow <- as.data.frame(df[df$edge == 1 & df$cell == 1,])

df_arrow$cell <- as.factor(df_arrow$cell)

# mercator
plt <- plt_Mer +
  geom_segment(data=df_arrow, aes(x = x,y = y, xend = xend,yend = yend, colour=cell),size =1)+
  # geom_segment(data=df_arrow, aes(x = x,y = y, xend = xend,yend = yend, colour=cell),size =1, arrow = arrow(length = unit(0.01, "npc"), type = "closed")) +
  scale_colour_manual(values = brewer.pal(7,"Set1"), breaks = c("1", "2", "3","4","5","6","7")) +
  scale_x_continuous(limits = c(-45,90)/180*pi,breaks = seq(-45,90,by=45)/180*pi, labels = paste0(seq(-45,90,by=45),"°"),  expand = c(0, 0)) +
  scale_y_continuous(limits = c(-15,75)/180*pi,breaks = log(tan(pi/4 + seq(-15,75,by=15)/180*pi/2)), labels = paste0(seq(-15,75,by=15),"°"),  expand = c(0, 0)) +
  # annotate("text", x = df_arrow$x+0.02, y = df_arrow$y+0.02, label = 1:nrow(df_arrow), size = 3) +
  labs(title = "edge - spk - bright")
windows(width = 9, height = 8)
plt
# ggsave(paste("H2_Mercator_edge_percell_spk_bright.pdf",sep=''), width = 9, height = 8)
# ggsave(paste("H2_Mercator_edge_percell_spk_dark.pdf",sep=''), width = 9, height = 8)


# ED Fig.1E, dark vs. bright vs. grating at each location ---------------------

load('data/H2_tuning_2023.rda' )

# position uniquely determined by  c('stimPosX', 'stimPosY')
# make a factor for plotting
uxy <- unique(tb[, c('edgeVal', 'stimPosX', 'stimPosY')]) %>%
  mutate(stimType = paste0(edgeVal,stimPosX, stimPosY),
         stimTypePos = paste0(stimPosX, stimPosY))

#base
thetaphi <- tb_v[,1:2] 
thetaphi[, 1] <- 90 - thetaphi[, 1]
thetaphi[, 2] <- 360 - thetaphi[, 2] # left to right, need it cos cart2Merc invert y
thetaphi <- thetaphi / 180*pi
thetaphi <- cbind(1, thetaphi)
xyz <- matrix(ncol = 3, nrow = nrow(thetaphi))
xyz <- sph2cartZ(thetaphi) %>% cart2Mercator()
dxyz <- xyz

#spk
thetaphi <- tb_v[,3:4] 
thetaphi[, 1] <- 90 - thetaphi[, 1]
thetaphi[, 2] <- 360 - thetaphi[, 2] # left to right, need it cos cart2Merc invert y
thetaphi <- thetaphi / 180*pi
thetaphi <- cbind(1, thetaphi)
xyz <- matrix(ncol = 3, nrow = nrow(thetaphi))
xyz <- sph2cartZ(thetaphi) %>% cart2Mercator()
pxyz <- xyz

# angle with x-axis
df_ang <- cbind(dxyz, pxyz)
colnames(df_ang) <- c('x0','y0','xsp','ysp')
df_ang <- as.data.frame(df_ang) %>%
  transmute(ang = atan2(ysp-y0, -(xsp-x0)) /pi*180)

df <- cbind(df_ang, tb[,1:5])
df <- merge(df, uxy, by=c('edgeVal','stimPosX', 'stimPosY'))

# -- wilcox test, avg bright/dark bars
# df_test <- df3[df3$edgeVal < 2, ]
df_test <- df
test_wilc <- matrix(nrow = 0, ncol = 2)
for (j in unique(df_test$stimTypePos)) {
  data <- df_test[df_test$stimTypePos == j,]
  test_wilc <- rbind(
    test_wilc,
    c(j,
      wilcox.test(data[,'ang'])$p.value
    )
  )
}
test_wilc[c(3, 5, 4, 6, 2, 1),] # reorder by position

# - add gratings, 2023
load('data/H2_tuning_2023_grating.rda' )

uxy <- unique(tb[, c('stimPosX','stimPosY')]) %>%
  mutate(stimType = paste0(2,stimPosX, stimPosY),
         stimTypePos = paste0(stimPosX, stimPosY))

#base
thetaphi <- tb_v[,1:2] 
thetaphi[, 1] <- 90 - thetaphi[, 1]
thetaphi[, 2] <- 360 - thetaphi[, 2] # left to right
thetaphi <- thetaphi / 180*pi
thetaphi <- cbind(1, thetaphi)
xyz <- matrix(ncol = 3, nrow = nrow(thetaphi))
xyz <- sph2cartZ(thetaphi) %>% cart2Mercator()
dxyz <- xyz

#spk
thetaphi <- tb_v[,3:4] 
thetaphi[, 1] <- 90 - thetaphi[, 1]
thetaphi[, 2] <- 360 - thetaphi[, 2] # left to right
thetaphi <- thetaphi / 180*pi
thetaphi <- cbind(1, thetaphi)
xyz <- matrix(ncol = 3, nrow = nrow(thetaphi))
xyz <- sph2cartZ(thetaphi) %>% cart2Mercator()
pxyz <- xyz

# angle with x-axis
df_ang <- cbind(dxyz, pxyz)
colnames(df_ang) <- c('x0','y0','xsp','ysp')
df_ang <- as.data.frame(df_ang) %>%
  transmute(ang = atan2(ysp-y0, -(xsp-x0)) /pi*180)

df2 <- cbind(df_ang, tb[,1:4])
df2 <- merge(df2, uxy, by=c('stimPosX', 'stimPosY'))
df2 <- cbind(2, df2)
colnames(df2)[1] <- "edgeVal"

# - combine
df3 <- rbind(df, df2)
df3$stimType <- factor(
  df3$stimType,
  levels=c('08713', '18713', '28713', "s1", '08729', '18729', '28729', "s2",
           '08721', '18721', '28721', "s3", '08736', '18736', '28736',"s4",
           '05828', '15828', '25828', "s5", '02913', '12913', '22913') # change order
)
df3$ang <- 90 - df3$ang

# PLOT
windows(width = 10, height = 5)
# pdf(paste("H2_edge_DvsBvsG", '.pdf', sep = ''), width = 10, height = 5)
ggplot(df3, aes(stimType, ang, colour= factor(edgeVal)) ) + 
  geom_jitter(size = 1,height = 0, width = 0.1) +
  stat_summary(fun.min = function(z) { quantile(z,0.25) },
               fun.max = function(z) { quantile(z,0.75) },
               geom = "errorbar", position = position_nudge(x = 0.2, y = 0),
               aes(colour = factor(edgeVal)),
               size = 1, width = .2 ) +
  stat_summary(fun = median, geom = "point",  aes(colour = factor(edgeVal)), size = 1, position = position_nudge(x = 0.2, y = 0)) +
  scale_discrete_manual(values = c('#762A83','#1B7837','#000000'),aesthetics = "colour",guide= guide_legend(title="")) +
  # coord_cartesian(ylim = c(65, 150)) +
  scale_y_continuous(breaks= seq(-75,50,by=25)+90, labels = paste0(seq(-75, 50,by=25)+90, "°")) +
  scale_x_discrete(drop = F,
                   labels=c('02913'='6D', '12913'='6B', '22913'='6G', 's1'='',
                            '05828'='5D', '15828'='5B', '25828'='5G', 's2'='',
                            '08736'='4D', '18736'='4B', '28736'='4G', 's3'='',
                            '08721'='3D', '18721'='3B', '28721'='3G', 's4'='',
                            '08729'='2D', '18729'='2B', '28729'='2G', 's5'='',
                            '08713'='1D', '18713'='1B', '28713'='1G')) +
  theme_minimal() +
  labs(title= paste("H2 tuning", sep = ''),
       x='position', y='Angle wrt +h [deg]') 
# dev.off()


# -- 2-way anova, no interaction
df4 <- df3
df4$stimTypePos <- factor(df4$stimTypePos)
df4$edgeVal <- factor(df4$edgeVal)
resp <- aov(ang ~ stimTypePos + edgeVal, data = df4)
summary(resp)




# ED Fig.1F, deformation,  ------------------------

# make template, center plus 8 directions
NN <- 10 # num of pixel as radius

NXY <- data.frame(dx = c(0,NN,0,-NN,0,NN/sqrt(2),-NN/sqrt(2),-NN/sqrt(2),NN/sqrt(2)),
                  dy = c(0,0,NN,0,-NN,NN/sqrt(2),NN/sqrt(2),-NN/sqrt(2),-NN/sqrt(2)))

# stim center
tb <- read_excel("data_Eyal/H2_2023/movEdgeTabwPY_ThetaCorr_fullResp.xlsx", sheet = 1, col_names = T ) %>% data.frame()
names(tb)[names(tb) == "headAngPitch"] <- "headAng"

stimpos <- tb %>% 
  mutate(pos = as.integer(paste0(stimPosX,stimPosY))) %>%
  group_by(pos) %>%
  summarise(stimPosX = first(stimPosX),
            stimPosY = first(stimPosY),
            arenaAng = mean(arenaAng),
            headAng = mean(headAng)) %>%
  as.data.frame()

pos <- merge(NXY, stimpos, all=T) %>%
  transmute(arenaAng = arenaAng, headAng = headAng,
            px = stimPosX + dx, py = stimPosY + dy)

# area params
A <- (4+10/32 + 4+20/32)/2  #radius
Dx <- (5+6/32) - A  # area center wrt to fly
Dy <- (4+20/32) - A
Dz <- - (1+ 13/32)

# LED size, assume 1.25 at equator
DL <- A * 1.25/180*pi

# - loop
stims <- matrix(ncol = 2, nrow = nrow(pos)) # [elev azim]

for (j in 1:nrow(stims)) {
  arot <- pos$arenaAng[j] /180*pi #arena rotation ang
  afly <- pos$headAng[j] /180*pi #equator below holder
  
  # cylindrical arena, 2d
  # angle from sagittal plane
  ax <- (120 - pos$px[j]*1.25) /180*pi
  #dist from bottom
  ay <- DL * pos$py[j]
  
  # go to lab coord via a translation
  # origin at arena center, xy-plane is level
  x <- A * cos(ax)
  y <- A * sin(ax)
  z <- ay - (4 + 24/32)/2
  
  xyz <- matrix(c(x,y,z), ncol=3)
  
  # arena rotation, 
  # arot is angle below ground, use -arot
  xyz <- matrix(c(cos(-arot), 0, -sin(-arot),
                  0, 1, 0,
                  sin(-arot), 0, cos(-arot)), ncol=3, byrow = T) %*% t(xyz)
  xyz <- t(xyz)
  
  # go to fly's coord
  xyz <- xyz + c(Dx, Dy, Dz)
  
  # fly rotation 
  # afly is angle below ground, use afly since it's ref that's rotating
  xyz <- matrix(c(cos(afly), 0, -sin(afly),
                  0, 1, 0, 
                  sin(afly), 0, cos(afly)), ncol = 3, byrow = T) %*% t(xyz)
  xyz <- t(xyz)
  
  # to spherical coord
  thetaphi <- cart2sphZ(xyz) /pi*180
  
  stims[j, ] <- round(c(90 - thetaphi[2], thetaphi[3]), 2)
}

# - Mercator projection
thetaphi <- stims

thetaphi[, 1] <- 90 - thetaphi[, 1] # elev to theta
thetaphi[, 2] <- 360 - thetaphi[, 2] # left to right
thetaphi <- thetaphi / 180*pi
thetaphi <- cbind(1, thetaphi)

xyz <- sph2cartZ(thetaphi)
xyz_mer <- cart2Mercator(xyz)

# PLOT
df <- as.data.frame(cbind(xyz_mer,
                          factor(rep(seq(1,6), each=9)),
                          rep(c(1,2,rep(1,7)), 6)
))
colnames(df) <- c('x','y','cell','size')

df_arrow <- matrix(ncol = 4, nrow = 0)
for (j in 0:5) {
  df_arrow <- rbind(df_arrow,
                    cbind(
                      matrix(xyz_mer[9*j+1,], nrow=8, ncol = 2, byrow = T),
                      xyz_mer[9*j+2:9,]
                    ))
}
df_arrow <- cbind(df_arrow, rep(seq(1,6), each=8))
df_arrow <- matrix(unlist(df_arrow), ncol = 5) 
df_arrow <- data.frame(df_arrow)
colnames(df_arrow) <- c('x','y','xend','yend','cell')

# mercator
plt <- plt_Mer +
  geom_point(data=df, aes(x = x,y = y,colour=cell,size =size))+
  geom_segment(data=df_arrow, aes(x = x,y = y, xend = xend,yend = yend, colour= as.factor(cell)))+
  scale_colour_manual(values = pal_9[c(5,4,3,6,2,1)],
                      breaks = as.character(seq(1,6))) +
  scale_x_continuous(limits = c(-15,95)/180*pi,breaks = seq(-180,180,by=45)/180*pi, labels = paste0(seq(-180,180,by=45), "°"),  expand = c(0, 0)) +
  scale_y_continuous(limits = c(-20,70)/180*pi,breaks = log(tan(pi/4 + seq(-75,75,by=15)/180*pi/2)), labels = paste0(seq(-75,75,by=15),"°"),  expand = c(0, 0)) +
  theme(legend.position="none")+
  labs(title = "templates ")
windows(width = 6.5, height = 6.5)
plt
# ggsave("arena_deform.pdf", width = 8, height = 8)

# - avg amplitudes 1 vs 4
p1 <- sweep(xyz[2:9,], 2, xyz[1,,drop=F], '-')^2 %>% rowSums() %>% mean() %>% sqrt()
p4 <- sweep(xyz[3*9+2:9,], 2, xyz[3*9+1,,drop=F], '-')^2 %>% rowSums() %>% mean() %>% sqrt()
abs(p1-p4)/p1


