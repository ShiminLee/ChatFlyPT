from pathlib import Path
import os
from dotenv import find_dotenv

import pandas as pd

def all_hex_df(
) -> pd.DataFrame:
    """
    Return dataframe with all 'hex1_id', 'hex2_id' coordinates
        
    Parameters
    ----------
    None

    Returns
    -------
    rtn : pd.DataFrame
        the columns are 'hex1_id' and 'hex2_id' and their values give all (hex1, hex2) in the ME,
        which is used as a reference in all neuropils
    """
    data_dir = Path(find_dotenv()).parent / "params"
    if os.environ['NEUPRINT_DATASET_NAME'] == 'flywire':
        me_hex_fn = data_dir / "ME_columnar-cells_location_flywire.xlsx"
        me_df = pd.read_excel(me_hex_fn, dtype=str, engine='openpyxl')
        me_df = me_df.astype({'hex1_id': 'int', 'hex2_id': 'int'})
        # me_df = me_df.apply(lambda col: col.str.strip().astype('Int64'))
    else:
        me_hex_fn = data_dir / "ME_columnar-cells_location.xlsx"
        me_df = pd.read_excel(me_hex_fn).convert_dtypes()
    hex_df = me_df[['hex1_id','hex2_id']].drop_duplicates()

    return hex_df.reset_index(drop=True)


def get_hex_df(neuropil:str='ME(R)') -> pd.DataFrame:
    """
    Get access to the full data frame

    This gives you the data frame without needing to know where exactly the file is stored or if
      the columns are directly pulled from neuPrint (future extension). Generally using this
      function is more advisable than using the current pickle file, but less advisable than using
      specialized functions to pull the information you need. If a function doesn't work as
      expected or if you are missing something, get in contact with @floesche or @kitlongden instead.

    Parameter
    ---------
    neuropil : str, default='ME'
        define the neuropil for which you want the data frame.
          (inactive, placeholder for future extension)

    Returns
    -------
    me_df : pd.DataFrame
        data frame with columns ['hex1_id', 'hex2_id'] and one additional column for each cell type
          that is assigned.

    """
    assert neuropil in ['ME(R)', 'ME(L)'],\
        "only Medulla"
    
    data_dir = Path(find_dotenv()).parent / "params"
    if os.environ['NEUPRINT_DATASET_NAME'] == 'flywire':
        me_hex_fn = data_dir / "ME_columnar-cells_location_flywire.xlsx"
        me_df = pd.read_excel(me_hex_fn, dtype=str, engine='openpyxl')
        me_df = me_df.astype({'hex1_id': 'int', 'hex2_id': 'int'})
        # me_df = me_df.apply(lambda col: col.str.strip().astype('Int64'))
    else:
        if neuropil=='ME(R)':
            me_hex_fn = data_dir / "ME_columnar-cells_location.xlsx"
        else:
            me_hex_fn = data_dir / "ME_L_columnar-cells_location.xlsx"
        me_df = pd.read_excel(me_hex_fn).convert_dtypes()
    me_df = me_df.sort_values(['hex1_id', 'hex2_id'])
    
    return me_df