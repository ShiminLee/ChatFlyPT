"""Hardcoded colour palettes used by paper-figure renderers."""
import matplotlib.colors as mcolors
import pandas as pd


def load_colors():
    """Return the four figure-wide colour dataframes.

    Returns:
        (colored_region_df, colored_main_groups_df, colored_sign_df,
         colored_seed_df)
    """
    colored_main_groups_df = pd.DataFrame(
        ['#90f0a6', '#ffc4e2', '#64a0d1', "#cc9e4f", "#616161"],
        index=['visual input', 'VIN', 'VPN', 'VCN', 'other'],
        columns=['color'],
    )
    colored_region_df = pd.DataFrame(
        ["#B6B6B6", "#868686", "#616161", "#383838"],
        index=['right OL', 'left OL', 'CB', 'VNC'],
        columns=['color'],
    )
    colored_sign_df = pd.DataFrame(
        ["#EE672D", "#979DA5", "#1F4695"], index=[1, 0, -1], columns=['color'],
    )
    colored_seed_df = pd.DataFrame.from_dict(
        {
            'L1': '#b4e5cb', 'L2': '#029356', 'L3': '#8fce0a',
            'R7': "#8e56f5", 'R7d': "#4220B4",
            'R8': "#51bfc7", 'R8d': "#099C9C",
            'HBeyelet': '#151515',
        },
        orient='index', columns=['color'],
    )
    return colored_region_df, colored_main_groups_df, colored_sign_df, colored_seed_df


def load_roi_colors():
    """Per-ROI hex/RGBA colours for CB region plots."""
    roi_hex = {
        'SLP':  '#5dc9e1',
        'SMP':  '#4597d2',
        'ICL':  '#652d91',
        'SCL':  '#a15cdb',
        'IB':   '#8351a1',
        'EB':   '#f36f40',
        'PB':   '#fbab3d',
        'BU':   '#f49cc3',
        'LAL':  '#c6197e',
        'LAL(L)': "#e86db3", # added by Arthur Zhao
        'AVLP': "#33cc64", # added by Arthur Zhao
        'AVLP(L)': "#5eed8b", # added by Arthur Zhao
        'PVLP': '#1cb24b',
        'PVLP(L)': "#19dc5d", # added by Arthur Zhao
        'WED':  '#079647',
        'PLP':  '#0d783d',
        'AOTU': '#1c401d',
        'IPS':  '#9dd6c9',
        'SPS':  '#05af8e',
        'GNG':  '#21366b',
    }
    df = pd.DataFrame({'ROI': list(roi_hex.keys()), 'Hex': list(roi_hex.values())})
    df['rgba'] = df['Hex'].apply(mcolors.to_rgba)
    return df
