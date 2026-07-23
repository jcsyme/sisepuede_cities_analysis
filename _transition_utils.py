import numpy as np
import os, os.path
import pandas as pd
import pathlib
import sisepuede.utilities._toolbox as sf

from typing import *




##########################
#    GLOBAL VARIABLES    #
##########################

# fields
_FIELD_AREA_M2 = "area_m2"
_FIELD_DIFF_FLAG = "diff_cats"
_FIELD_END = "end"
_FIELD_SSP = "field_ssp"
_FIELD_START = "start"
_FIELD_YEAR = "year"

# some units info
_UNITS_DATA_AREA = "m2"


##############################
#    BUILD BASE FUNCTIONS    #
##############################

def add_ssp_field(
    df: pd.DataFrame,
    model_afolu: 'AFOLU',
) -> pd.DataFrame:
    """
    """

    # filter down to only unique
    df_base = df[[_FIELD_START, _FIELD_END]].drop_duplicates()

    # build vec of fields
    vec_new_field = []
    vec_new_diff_flag = []
    
    for i, row in df_base.iterrows():
        field, flag = build_ssp_field_and_flag(
            row,
            model_afolu
        )

        vec_new_field.append(field[0], )
        vec_new_diff_flag.append(flag, )

    df_base[_FIELD_SSP] = vec_new_field
    df_base[_FIELD_DIFF_FLAG] = vec_new_diff_flag

    df_out = (
        pd.merge(
            df,
            df_base,
            how = "left",
        )
    )
    
    return df_out



def build_ssp_field_and_flag(
    row: pd.Series,
    model_afolu: 'AFOLU',
) -> Tuple[str, float]:
    """Build the SISEPUEDE field associated with the transition.
    """
    # shortcuts
    matt = model_afolu.model_attributes

    # get the land use category
    pycat_lndu = matt.get_subsector_attribute(
        matt.subsec_name_lndu,
        "pycategory_primary_element",
    )

    # get info from row and field build
    cat1 = row[_FIELD_START]
    cat2 = row[_FIELD_END]
    flag = int(cat1 != cat2)
    
    field_out = model_afolu.modvar_lndu_prob_transition.build_fields(
        category_restrictions = {
            f"{pycat_lndu}_dim1": [cat1],
            f"{pycat_lndu}_dim2": [cat2]
        }
    )
    
    # output tuple
    out = (field_out, flag, )

    return out



def get_area(
    df: pd.DataFrame,
) -> float:
    """Get the area associated with the region
    """
    df_out = (
        df
        .get([_FIELD_YEAR, _FIELD_AREA_M2])
        .groupby([_FIELD_YEAR])
        .sum()
        .reset_index()
    )
    
    area = df_out[_FIELD_AREA_M2].mean()

    return area



def get_df_complete_years(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Get dataframe of complete years
    """

    df_years = pd.DataFrame(
        {
            _FIELD_YEAR: range(
                df[_FIELD_YEAR].min(), 
                df[_FIELD_YEAR].max() + 1
            ), 
        }
    )

    return df_years



def get_initial_prevalence_dict(
    df: pd.DataFrame,
) -> Dict[str, float]:
    """Get a dictionary mapping each land use class to its initial area
    """

    dict_area_init = (
        df[
            df[_FIELD_YEAR].isin([int(df[_FIELD_YEAR].min())])
        ]
        .get([_FIELD_START, _FIELD_AREA_M2])
        .groupby([_FIELD_START])
        .sum()
        .to_dict()
        .get(_FIELD_AREA_M2, )
    )

    return dict_area_init



def get_initial_prevalence_vector(
    df: pd.DataFrame,
    attr_lndu: 'AttributeTable',
) -> np.ndarray:
    """Build a prevalence vector from the initial state
    """
    
    # get initial area
    dict_area_init = get_initial_prevalence_dict(df, )
    
    # initialize prevalence in vector
    x = np.zeros(attr_lndu.n_key_values, )
    for k in attr_lndu.key_values:
        ind = attr_lndu.get_key_value_index(k, )
        val = dict_area_init.get(k, )
    
        x[ind] = val

    return x



def get_ssp_vars(
    df: pd.DataFrame,
    model_afolu: 'AFOLU',
    units_input_data: str = _UNITS_DATA_AREA, 
) -> pd.DataFrame:
    """Get SISEPUEDE input variable DataFrame for variables of interest, including

        - Area
        - Initial Land Use Proportion
        - Unadjusted Land Use Transition Probability
    """

    ##  INITIALIZATION

    # shortcuts
    model_attributes = model_afolu.model_attributes
    
    # variables and number of time periods
    modvar_area = model_attributes.get_variable("Area of Region")
    modvar_ilp = model_attributes.get_variable("Initial Land Use Area Proportion")
    n_tp = model_attributes.get_dimensional_attribute_table("time_period").n_key_values


    ##  GET VARS FROM DATA

    area = get_area(df, )

    (
        arr_transitions,
        vec_prev_0,
        vec_years,
        df_split,
    ) = get_transition_matrices(df, model_attributes, )


    ##  GET TRANSITION MATRIX INPUTS
    
    # use AFOLU method
    df_overwrite = [
        model_afolu.format_transition_matrix_as_input_dataframe(arr_transitions[0:n_tp], )
    ]
    
    
    ##  INITIAL LAND USE PROPORTION
    
    # noramlize vector of initial proportions
    arr_init_prev = np.ones((n_tp, vec_prev_0.shape[0], ))
    arr_init_prev *= vec_prev_0/vec_prev_0.sum()
    
    df_overwrite.append(
        model_attributes.array_to_df(
            arr_init_prev,
            modvar_ilp,
        )
    )
    
    
    ##  AREA 
    
    # convert to variable units
    factor = (
        model_attributes
        .get_unit("area")
        .convert(
            units_input_data, 
            modvar_area.attribute("unit_area"), 
        )
    )
    vec_area = area*factor*np.ones(n_tp, )

    # add area to output
    df_overwrite.append(
        model_attributes.array_to_df(
            vec_area,
            modvar_area,
        )
    )

    # concatenate and return
    df_overwrite = pd.concat(df_overwrite, axis = 1, )

    return df_overwrite



def get_transition_matrices(
    df_base: pd.DataFrame,
    model_attributes: 'ModelAttributes',
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Get transition matrices, prevalence vector, and indexing years. Returns
        a tuple of the form:

        (
            arr_matrices,
            vec_initial_prevalence_fracs,
            vec_years,
            df_split,
        )
        
    """
    ##  TURN INTO TRANSITION MATRICES
    
    attr_lndu = model_attributes.get_attribute_table(
        model_attributes.subsec_name_lndu,
    )
    dict_lndu_cat_to_ind = dict(
        (x, i) for i, x in enumerate(attr_lndu.key_values)
    )

    # split out transitions and then get all years
    df_split = split_by_year(df_base, ) 
    all_years = np.array(
        sorted(
            df_split[_FIELD_YEAR].unique()
        )
    )

    # initialize output matrices
    matrices = np.zeros(
        (
            all_years.shape[0], 
            attr_lndu.n_key_values, 
            attr_lndu.n_key_values, 
        )
    )
    
    # format data frame
    df_split_inds = df_split.copy()
    for field in [_FIELD_END, _FIELD_START]:
        df_split_inds[field] = df_split_inds[field].replace(dict_lndu_cat_to_ind, )
    
    dict_split_inds_g = sf.group_df_as_dict(
        df_split_inds,
        [_FIELD_YEAR],
    )
    
    
    ##  GET PREVALENCE AND ITERATE TO UPDATE MATRICES

    # initial prevalence
    x_0 = get_initial_prevalence_vector(df_base, attr_lndu, )
    x = x_0.copy()
    
    # ensures ordering
    for i, y in enumerate(all_years):# all_years:
        
        df = dict_split_inds_g.get(y)
    
        # info for filling matrix
        ind1 = df[_FIELD_START].to_numpy().astype(int)
        ind2 = df[_FIELD_END].to_numpy().astype(int)
        inds_diag = np.arange(attr_lndu.n_key_values)
        
        y = df[_FIELD_AREA_M2].to_numpy()
    
        # 
        arr = matrices[i]
        arr[ind1, ind2] = y
    
        # 
        mass_accounted = arr.sum(axis = 1, )
        arr[inds_diag, inds_diag] = x - mass_accounted
        vec_norm = arr.sum(axis = 1)

        # if no mass is found in cat, set diag to 1
        w = np.where(vec_norm == 0, )[0]
        if len(w) > 0:
            arr[w, w] = 1.0
    
        # update matrices
        vec_norm = arr.sum(axis = 1)
        arr_norm = (arr.transpose()/vec_norm).transpose()
        matrices[i] = arr_norm
    
        # update prevalence x
        x = np.dot(x, arr_norm)
        

    # set output
    out = (
        matrices,
        x_0,
        all_years,
        df_split,
    )

    return out

        

def split_by_year(
    df_fields: pd.DataFrame,
) -> pd.DataFrame:
    """Split out multi-year transitions to annual
    """

    dfg = df_fields.groupby(
        [
            _FIELD_START,
            _FIELD_END
        ]
    )


    ##  ITERATE OVER GROUPS
    
    df_out = []
    df_years = get_df_complete_years(df_fields, )
    global dfa 
    for i, df in dfg:
        
        if i[0] == i[1]: 
            #df_out.append(df)
            continue

        df_append = split_transition_by_year_interpolate(df, )
        df_out.append(df_append, )

        dfa = df_append.copy()
        
    df_out = (
        sf._concat_df(df_out, )
        .groupby(
            [
                _FIELD_START,
                _FIELD_END,
                _FIELD_YEAR,
            ]
        )
        .sum()
        .reset_index()
    )

    return df_out



def split_transition_by_year(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Split out transitions for non-equal categories by year.
    """

    df_out = []
    years_sorted = sorted(list(df[_FIELD_YEAR].unique()))

    # used to set differences
    dict_years_to_next = dict(
        (years_sorted[i], years_sorted[i + 1])
        for i in range(len(years_sorted) - 1)
    )

    dfg = df.groupby([_FIELD_YEAR], )

    
    for i, df_cur in dfg:
        year = i[0]
        year_next = dict_years_to_next.get(year, )
        if year_next is None: continue
            
        year_diff = year_next - year

        row_base = df_cur.iloc[0].copy()
        
        for k, y in enumerate(range(year_diff)):

            val_base = row_base[_FIELD_AREA_M2]
            row_new = row_base.copy()           # 
            
            # set the year
            row_new[_FIELD_YEAR] = year + k
            row_new = pd.DataFrame(row_new, ).transpose()

            extendage = [row_new]
            
            # update the 
            if row_base[_FIELD_START] != row_base[_FIELD_END]:

                # adjust value to reduce uniformly each year
                val_transition = val_base/year_diff
                val_stay = val_base*(year_diff - k - 1)/year_diff
            
                row_stable_new = row_base.copy()   # area preserved
                
                row_new[_FIELD_AREA_M2] = val_transition
                row_stable_new[_FIELD_AREA_M2] = val_stay

                # set year and end state to start 
                row_stable_new[_FIELD_YEAR] = year + k
                row_stable_new[_FIELD_END] = row_stable_new[_FIELD_START]

                # turn to dataframe
                row_stable_new = pd.DataFrame(row_stable_new, ).transpose()

                extendage = [row_new, row_stable_new]

            
            df_out.extend(extendage, )

    
    df_out = sf._concat_df(df_out, )

    return df_out



def split_transition_by_year_interpolate(
    df: pd.DataFrame,
    field_norm: str = "_NORM",
    field_year_base: str = "_YEAR_BASE",
) -> pd.DataFrame:
    """Split out transitions for non-equal categories by year.
        Assume that cat_source != cat_target
    """

    df_out = []
    years_sorted = sorted(list(df[_FIELD_YEAR].unique()))

    # used to set differences
    dict_years_to_next = dict(
        (years_sorted[i], years_sorted[i + 1])
        for i in range(len(years_sorted) - 1)
    )

    dfg = df.groupby([_FIELD_YEAR], )
    df_years = get_df_complete_years(df, )


    ##  BUILD DF OF TRANSITIONS OUT TO SMOOTH

    # initialize data frame
    df_interp = {_FIELD_YEAR: [], _FIELD_AREA_M2: [], }
    dict_year_to_total = {}

    diff_next = None
    
    for i, df_cur in dfg:

        # row and associated info
        row_base = df_cur.iloc[0].copy()
        val_base = row_base[_FIELD_AREA_M2]
        year = i[0]

        # update dict
        dict_year_to_total.update({year: val_base, })
        
        # get years--skip if next year is not defined
        year_next = dict_years_to_next.get(year, )

        """
        if year_next is None:
            if diff_next is None:
                raise RuntimeError(f"No successive years found.")
            
            val_next = val_base/
            
        year_diff = year_next - year
        diff_next = year_diff
        year_midpoint = int(year + np.floor((year_diff - 1)/2))
        
        # get values we need to work with
        val_transition_avg = val_base/year_diff

        
        df_interp.get(_FIELD_YEAR).append(year)
        df_interp.get(_FIELD_AREA_M2).append(val_transition_avg)
        """
    
    # merge, then 
    df_interp = (
        pd.merge(
            df_years,
            df.get([_FIELD_YEAR, _FIELD_AREA_M2], ),
            #pd.DataFrame(df_interp, ),
            how = "left",
        )
        .interpolate(method = "spline", order = 2, )
    )   
    df_interp[_FIELD_AREA_M2] = np.clip(
        df_interp[_FIELD_AREA_M2],
        0, 
        np.inf, 
    )

    # set a df to merge in 
    df_merge = df.get([_FIELD_YEAR]).copy()
    df_merge[field_year_base] = df_merge[_FIELD_YEAR].copy()
    
    df_interp = (
        pd.merge(
            df_interp,
            df_merge,
            how = "left",
        )
        .sort_values(by = [_FIELD_YEAR], )
        .reset_index(drop = True, )
    )

    # fill forward
    df_interp[field_year_base] = df_interp[field_year_base].ffill().astype(int)
    #return df_interp

    ##  GROUP AGAIN TO NORMALIZE

    df_out = []
    dfg = df_interp.groupby(field_year_base)

    for i, df_cur in dfg:
        total = df_cur[_FIELD_AREA_M2].sum()
        
        targ = dict_year_to_total.get(i, )
        df_cur[_FIELD_AREA_M2] *= (
            targ/total
            if total > 0
            else 0
        )

        df_out.append(df_cur, )

    ##  COMPILE OUTPUT
    
    df_out = sf._concat_df(df_out, )
    df_out[_FIELD_END] = df[_FIELD_END].iloc[0]
    df_out[_FIELD_START] = df[_FIELD_START].iloc[0]

    df_out = df_out.get(df.columns, )
    
    return df_out
