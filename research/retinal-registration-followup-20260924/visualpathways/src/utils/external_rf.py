import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter


def smooth_weighted_hex(df, dx=0.2, sigma=1, pad=1):
    """
    df: pandas DataFrame with columns 'x','y','effective weight' (cartesian coords)
    dx: grid spacing (same units as x,y)
    sigma: gaussian smoothing bandwidth in same units as x,y
    pad: extra padding in grid extents (in number of grid cells)
    returns: (Xgrid, Ygrid, smooth_grid, interp_fn)
    """
    x = df['x'].values
    y = df['y'].values
    w = df['effective weight'].values

    # grid extents
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    # add small margin
    xmin -= pad*dx; xmax += pad*dx
    ymin -= pad*dx; ymax += pad*dx

    # grid coords
    nx = int(np.ceil((xmax - xmin)/dx)) + 1
    ny = int(np.ceil((ymax - ymin)/dx)) + 1
    xedges = np.linspace(xmin, xmax, nx+1)
    yedges = np.linspace(ymin, ymax, ny+1)

    # bin weighted points (note histogram2d expects x,y order)
    H, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=w)

    sigma_pixels = sigma / dx
    H_smooth = gaussian_filter(H, sigma=sigma_pixels, mode='constant')

    # grid centers
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters, indexing='xy')

    smooth_df = pd.DataFrame({
        'x': X.ravel(),
        'y': Y.ravel(),
        'effective weight': H_smooth.T.ravel()  # transpose because histogram2d arr shape is (nx, ny)
    })
    
    return smooth_df


def circular_mean(angles):
    # idx_neg = np.where(angles < 0)[0]
    # angles[idx_neg] += np.pi
    # mean_angle = np.mean(angles)
    # if mean_angle > np.pi/2:
    #     mean_angle -= np.pi
    # phi is orientation confined to [-pi/2, pi/2], so period = pi
    # Double angles to map to full circle, compute mean, then halve
    mean_angle = np.arctan2(np.sin(2 * angles).sum(), np.cos(2 * angles).sum()) / 2
    return mean_angle


def twoD_Gaussian(x, y, x0, y0, sigma_x, sigma_y, theta):
    # shift
    x_c = x - x0
    y_c = y - y0
    
    # rotate coordinates
    x_rot =  x_c * np.cos(theta) + y_c * np.sin(theta)
    y_rot = -x_c * np.sin(theta) + y_c * np.cos(theta)
    
    # Gaussian formula
    g = np.exp(-0.5 * ((x_rot/sigma_x)**2 + (y_rot/sigma_y)**2))

    return g


def find_initial_params(rel_rf_cut_df):
    x = rel_rf_cut_df['x'].values
    y = rel_rf_cut_df['y'].values
    w = rel_rf_cut_df['effective weight'].values
    
    if len(x) == 1:
        return [x[0], y[0], 1, 1, 0]

    # Weighted mean (Gaussian center)
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)

    # Weighted covariance
    x_c = x - xm
    y_c = y - ym
    cov = np.cov(np.vstack([x_c, y_c]), aweights=w)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # enforce sigma_x >= sigma_y
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    sigma = np.sqrt(eigvals)
    sigma_x, sigma_y = sigma[0], sigma[1]
    theta = np.arctan2(eigvecs[1,0], eigvecs[0,0])

    # wrap theta to (-pi/2, pi/2]
    if theta <= -np.pi/2:
        theta += np.pi
    elif theta > np.pi/2:
        theta -= np.pi

    # if somehow sigma_x < sigma_y (edge case), swap and rotate by 90°
    if sigma_x < sigma_y:
        sigma_x, sigma_y = sigma_y, sigma_x
        theta += np.pi/2
        # wrap again
        if theta <= -np.pi/2:
            theta += np.pi
        elif theta > np.pi/2:
            theta -= np.pi

    return [xm, ym, sigma_x, sigma_y, theta]


def fit_rf_gaussian(rf_df, p0=None, sigma=1.0, cumsum_thre=0.7):

    #smoothen and renormalize
    smooth_df = smooth_weighted_hex(rf_df, sigma=sigma)

    #mask low weights 
    if cumsum_thre is not None:   
        ord_weights = np.sort(smooth_df['effective weight'].values)[::-1]
        cumsum_weights = np.cumsum(ord_weights) / np.sum(ord_weights)
        n_weights = np.where(cumsum_weights>cumsum_thre)[0][0]
        thre_weights = ord_weights[n_weights]
        smooth_df = smooth_df[smooth_df['effective weight'] > thre_weights]

    #find initial parameters if not provided
    if p0 is None:
        p0 = find_initial_params(smooth_df)

    #mask original df
    rel_rf_cut_df = rf_df[(rf_df['x'] >= smooth_df['x'].min()) & (rf_df['x'] <= smooth_df['x'].max()) &
                          (rf_df['y'] >= smooth_df['y'].min()) & (rf_df['y'] <= smooth_df['y'].max())]
    rf_fitted = twoD_Gaussian(rel_rf_cut_df['x'], rel_rf_cut_df['y'], *p0)    
    rf_fitted = rf_fitted/np.sum(rf_fitted)

    #goodness of fit (R^2)
    rf_raw = rel_rf_cut_df['effective weight'].values / rel_rf_cut_df['effective weight'].sum() 
    ss_res = np.sum((rf_raw - rf_fitted)**2)
    ss_tot = np.sum((rf_raw - np.mean(rf_raw))**2)
    # R^2 is undefined when the raw RF has zero variance (e.g. a single hex),
    # which makes ss_tot == 0; return nan instead of triggering a divide warning.
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan

    return np.insert(p0,0,r2), rf_fitted