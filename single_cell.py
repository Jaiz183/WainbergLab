from __future__ import annotations
import h5py
import numpy as np
import os
import polars as pl
import re
import sys
from decimal import Decimal
from datetime import date, datetime, time, timedelta
from itertools import chain, islice
from packaging import version
from pathlib import Path
from scipy.sparse import csr_array, csc_array, csr_matrix, csc_matrix, \
    hstack, vstack
from scipy.stats import mode, rankdata
from textwrap import fill
from typing import Any, Callable, Dict, ItemsView, Iterable, KeysView, \
    Literal, Mapping, Sequence, ValuesView, Union
from utils import bonferroni, check_bounds, check_dtype, check_type, \
    check_types, cython_inline, cython_type, fdr, filter_columns, \
    generate_palette, getnnz, is_integer, plural, prange, \
    sparse_matrix_vector_op, Timer, to_tuple
pl.enable_string_cache()

Color = Union[str, float, np.floating,
              tuple[Union[int, np.integer], Union[int, np.integer],
                    Union[int, np.integer]],
              tuple[Union[int, np.integer], Union[int, np.integer],
                    Union[int, np.integer], Union[int, np.integer]],
              tuple[Union[float, np.floating], Union[float, np.floating],
                    Union[float, np.floating]],
              tuple[Union[float, np.floating], Union[float, np.floating],
                    Union[float, np.floating], Union[float, np.floating]]]
Indexer = Union[int, np.integer, str, slice,
                np.ndarray[1, Union[np.integer, np.bool_]], pl.Series,
                list[Union[int, np.integer, str, bool, np.bool_]]]
Scalar = Union[str, int, float, Decimal, date, time, datetime, timedelta, bool,
               bytes]
NestedScalarOrArrayDict = \
    Dict[str, Union[str, int, np.integer, float, np.floating, bool, np.bool_,
         np.ndarray[Any, Any], 'NestedScalarOrArrayDict']]
SingleCellColumn = \
    Union[str, pl.Expr, pl.Series, np.ndarray,
          Callable[['SingleCell'], Union[pl.Series, np.ndarray]]]
PseudobulkColumn = \
    Union[str, pl.Expr, pl.Series, np.ndarray,
          Callable[['Pseudobulk', str], Union[pl.Series, np.ndarray]]]


class SingleCell:
    """
    A lightweight alternative to AnnData for representing single-cell data.
    
    Has slots for:
    - X: a scipy sparse array of counts per cell and gene
    - obs: a polars DataFrame of cell metadata
    - var: a polars DataFrame of gene metadata
    - obsm: a dictionary of NumPy arrays of cell metadata
    - varm: a dictionary of NumPy arrays of gene metadata
    - uns: a dictionary of scalars (strings, numbers or Booleans) or NumPy
           arrays, or nested dictionaries thereof
    as well as `obs_names` and `var_names`, aliases for obs[:, 0] and
    var[:, 0].
    
    Why is X a sparse array rather than matrix? Aside from being more modern
    and consistent with np.array, it's also faster, as explained at
    github.com/scipy/scipy/blob/2aee5efcbe3720f41fe55f336f492ae0acbecdee/scipy/
    sparse/_base.py#L1333-L1337.
    """
    # noinspection PyUnresolvedReferences
    def __init__(self,
                 X: csr_array | csc_array | csr_matrix | csc_matrix |
                    'AnnData' | str | Path,
                 obs: pl.DataFrame | None = None,
                 var: pl.DataFrame | None = None,
                 obsm: dict[str, np.ndarray[2, Any]] | None = None,
                 varm: dict[str, np.ndarray[2, Any]] | None = None,
                 uns: NestedScalarOrArrayDict | None = None,
                 *,
                 X_key: str | None = None,
                 assay: str | None = None,
                 obs_columns: str | Iterable[str] = None,
                 var_columns: str | Iterable[str] = None,
                 num_threads: int | np.integer | None = 1) -> None:
        """
        Load a SingleCell dataset from an AnnData, Seurat or 10x file, or
        create one from an in-memory AnnData object or count matrix + metadata.
        To create a SingleCell dataset from an in-memory Seurat object, use
        `SingleCell.from_seurat()`.
        
        By default, when an AnnData or Seurat file/object contains both raw and
        normalized counts, only the raw counts will be loaded. To load
        normalized counts instead, use the `X_key` argument.
        
        Loading Seurat objects requires the ryp Python-R bridge.
        
        Args:
            X: the data as a sparse array or matrix (with rows = cells,
               columns = genes), AnnData object, AnnData .h5ad file,
               Seurat .rds file (version 3, not version 5), or 10x .mtx.gz file
               (with barcodes.tsv.gz and features.tsv.gz assumed to be in the
               same directory, unless custom paths to these files are specified
               via the obs and/or var arguments)
            obs: a polars DataFrame of metadata for each cell (row of X), or
                 if X is a 10x .mtx.gz file, an optional filename for
                 cell-level metadata (which is otherwise assumed to be at
                 barcodes.tsv.gz in the same directory as the .mtx.gz file)
            var: a polars DataFrame of metadata for each gene (column of X), or
                 if X is a 10x .mtx.gz file, an optional filename for
                 gene-level metadata (which is otherwise assumed to be at
                 features.tsv.gz in the same directory as the .mtx.gz file)
            obsm: a dictionary of NumPy arrays of metadata for each cell. Keys
                  must be strings.
            varm: a dictionary of NumPy arrays of metadata for each gene. Keys
                  must be strings.
            uns: a dictionary of unstructured metadata. Keys must be strings;
                 values can be scalars (strings, numbers or Booleans), NumPy
                 arrays, or nested dictionaries thereof.
            X_key: if X is an AnnData object, the location within the object to
                   use as X. If None, defaults to `layers['UMIs']` or `raw/X`
                   if present, otherwise `X`.
                   If X is an .h5ad filename, the name of the key in the .h5ad
                   file to use as X. If None, defaults to `'layers/UMIs'` or
                   `'raw/X'` if present, otherwise `'X'`.
                   If X is a Seurat .rds filename, the slot within the active
                   assay (or the assay specified by the `assay` argument, if
                   not None) to use as X. If None, defaults to `'counts'`.
                   Set to `'data'` to load the normalized counts, or
                   `'scale.data'` to load the normalized and scaled counts.
                   If dense, will be automatically converted to a sparse array.
            assay: if X is a Seurat .rds filename, the name of the assay within
                   the Seurat object to load data from. Defaults to
                   `seurat_object@active_assay` (usually `'RNA'`).
            obs_columns: if X is a filename, the columns of obs to load
            var_columns: if X is a filename, the columns of var to load
            num_threads: the number of threads to use when reading .h5ad files;
                         set `num_threads=None` to use all available cores
                         (as determined by `os.cpu_count()`)
        
        Note:
            When initializing from an AnnData object or .h5ad file, both
            ordered and unordered categorical columns of obs and var will be
            loaded as polars Enums rather than polars Categoricals. This is
            because polars Categoricals use a shared numerical encoding across
            columns, so their codes are not [0, 1, 2, ...] like pandas
            categoricals and polars Enums are. Using Categoricals would lead to
            a large overhead (~25% for a typical dataset when loading obs).
        """
        type_string = str(type(X))
        is_anndata = type_string.startswith("<class 'anndata")
        if is_anndata:
            from anndata import AnnData
            if not isinstance(X, AnnData):
                is_anndata = False
        is_filename = isinstance(X, (str, Path))
        if not is_filename:
            if obs_columns is not None:
                error_message = \
                    'when X is not a filename, obs_columns must be None'
                raise ValueError(error_message)
            if var_columns is not None:
                error_message = \
                    'when X is not a filename, var_columns must be None'
                raise ValueError(error_message)
        else:
            if obs_columns is not None:
                obs_columns = to_tuple(obs_columns)
                if len(obs_columns) == 0:
                    error_message = 'obs_columns is empty'
                    raise ValueError(error_message)
                check_types(obs_columns, 'obs_columns', str, 'strings')
            if var_columns is not None:
                var_columns = to_tuple(var_columns)
                if len(var_columns) == 0:
                    error_message = 'var_columns is empty'
                    raise ValueError(error_message)
                check_types(var_columns, 'var_columns', str, 'strings')
        is_h5ad = is_filename and X.endswith('.h5ad')
        if is_h5ad:
            if num_threads is None:
                num_threads = os.cpu_count()
            else:
                check_type(num_threads, 'num_threads', int,
                           'a positive integer')
                check_bounds(num_threads, 'num_threads', 1)
        elif num_threads != 1:
            error_message = \
                'num_threads can only be specified when loading an .h5ad file'
            raise ValueError(error_message)
        if isinstance(X, (csr_array, csc_array, csr_matrix, csc_matrix)):
            for prop, prop_name in (X_key, 'X_key'), (assay, 'assay'):
                if prop is not None:
                    error_message = (
                        f'when X is a sparse array or matrix, {prop_name} '
                        f'must be None')
                    raise ValueError(error_message)
            check_type(obs, 'obs', pl.DataFrame, 'a polars DataFrame')
            check_type(var, 'var', pl.DataFrame, 'a polars DataFrame')
            if obsm is None:
                obsm = {}
            if varm is None:
                varm = {}
            if uns is None:
                uns = {}
            for field, field_name in (obsm, 'obsm'), (varm, 'varm'):
                for key, value in field.items():
                    if not isinstance(key, str):
                        error_message = (
                            f'all keys of {field_name} must be strings, but '
                            f'it contains a key of type '
                            f'{type(key).__name__!r}')
                        raise TypeError(error_message)
                    if not isinstance(value, np.ndarray):
                        error_message = (
                            f'all values of {field_name} must be NumPy '
                            f'arrays, but {field_name}[{key!r}] has type '
                            f'{type(value).__name__!r}')
                        raise TypeError(error_message)
                    if value.ndim != 2:
                        error_message = (
                            f'all values of {field_name} must be 2D NumPy '
                            f'arrays, but {field_name}[{key!r}] is '
                            f'{value.ndim:,}D')
                        raise ValueError(error_message)
                valid_uns_types = str, int, np.integer, float, np.floating, \
                    bool, np.bool_, np.ndarray
                for description, value in SingleCell._iter_uns(uns):
                    if not isinstance(value, valid_uns_types):
                        error_message = (
                            f'all values of uns must be scalars (strings, '
                            f'numbers or Booleans) or NumPy arrays, or nested '
                            f'dictionaries thereof, but {description} has '
                            f'type {type(value).__name__!r}')
                        raise TypeError(error_message)
            if isinstance(X, csr_matrix):
                X = csr_array(X)
            if isinstance(X, csc_matrix):
                X = csc_array(X)
            self._X = X
            self._obs = obs
            self._var = var
            self._obsm = obsm
            self._varm = varm
            self._uns = uns
        elif is_filename:
            X = str(X)
            if X.endswith('.h5ad'):
                if not os.path.exists(X):
                    error_message = f'h5ad file {X!r} does not exist'
                    raise FileNotFoundError(error_message)
                for prop, prop_name in (obs, 'obs'), (var, 'var'), \
                        (obsm, 'obsm'), (varm, 'varm'), (uns, 'uns'), \
                        (assay, 'assay'):
                    if prop is not None:
                        error_message = (
                            f'when loading an .h5ad file, {prop_name} must be '
                            f'None')
                        raise ValueError(error_message)
                # See anndata.readthedocs.io/en/latest/fileformat-prose.html
                # for the AnnData on-disk format specification
                with h5py.File(X) as h5ad_file:
                    # Load obs and var
                    self._obs = self._read_h5ad_dataframe(
                        h5ad_file, 'obs', columns=obs_columns,
                        num_threads=num_threads)
                    self._var = self._read_h5ad_dataframe(
                        h5ad_file, 'var', columns=var_columns,
                        num_threads=num_threads)
                    # Load obsm
                    if 'obsm' in h5ad_file:
                        obsm = h5ad_file['obsm']
                        self._obsm = {key: obsm[key][:] for key in obsm}
                    else:
                        self._obsm = {}
                    # Load varm
                    if 'varm' in h5ad_file:
                        varm = h5ad_file['varm']
                        self._varm = {key: varm[key][:] for key in varm}
                    else:
                        self._varm = {}
                    # Load uns
                    if 'uns' in h5ad_file:
                        self._uns = SingleCell._read_uns(h5ad_file['uns'])
                    else:
                        self._uns = {}
                    # Load X
                    if X_key is None:
                        has_layers_UMIs = 'layers/UMIs' in h5ad_file
                        has_raw_X = 'raw/X' in h5ad_file
                        if has_layers_UMIs and has_raw_X:
                            error_message = (
                                "both layers['UMIs'] and raw.X are present; "
                                "this should never happen in well-formed "
                                "AnnData files")
                            raise ValueError(error_message)
                        X_key = 'layers/UMIs' if has_layers_UMIs else \
                            'raw/X' if has_raw_X else 'X'
                    else:
                        check_type(X_key, 'X_key', str, 'a string')
                        if X_key not in h5ad_file:
                            error_message = (
                                f'X_key {X_key!r} is not present in the .h5ad '
                                f'file')
                            raise ValueError(error_message)
                    X = h5ad_file[X_key]
                    matrix_class = X.attrs['encoding-type'] \
                        if 'encoding-type' in X.attrs else \
                        X.attrs['h5sparse_format'] + '_matrix'
                    if matrix_class == 'csr_matrix':
                        array_class = csr_array
                    elif matrix_class == 'csc_matrix':
                        array_class = csc_array
                    else:
                        error_message = (
                            f"X has unsupported encoding-type "
                            f"{matrix_class!r}, but should be 'csr_matrix' or "
                            f"'csc_matrix' (csr is preferable for speed)")
                        raise ValueError(error_message)
                    self._X = array_class((
                        self._read_dataset(X['data'], num_threads),
                        self._read_dataset(X['indices'], num_threads),
                        self._read_dataset(X['indptr'], num_threads)),
                        shape=X.attrs['shape'] if 'shape' in X.attrs else
                              X.attrs['h5sparse_shape'])
            else:
                if X.endswith('.rds'):
                    if not os.path.exists(X):
                        error_message = f'Seurat file {X} does not exist'
                        raise FileNotFoundError(error_message)
                    for prop, prop_name in (obs, 'obs'), (var, 'var'), \
                            (obsm, 'obsm'), (varm, 'varm'), (uns, 'uns'):
                        if prop is not None:
                            error_message = (
                                f'when loading an .rds file, {prop_name} '
                                f'must be None')
                            raise ValueError(error_message)
                    from ryp import r, to_py, to_r
                    r('suppressPackageStartupMessages(library(SeuratObject))')
                    r(f'.SingleCell.seurat_object = readRDS({X!r})')
                    try:
                        if not to_py('inherits(.SingleCell.seurat_object, '
                                     '"Seurat")'):
                            classes = to_py('class(.SingleCell.seurat_object)',
                                            squeeze=False)
                            if len(classes) == 0:
                                error_message = (
                                    f'the R object loaded from {X} must be a '
                                    f'Seurat object, but has no class')
                            elif len(classes) == 1:
                                error_message = (
                                    f'the R object loaded from {X} must be a '
                                    f'Seurat object, but has class '
                                    f'{classes[0]!r}')
                            else:
                                classes_string = \
                                    ', '.join(f'{c!r}' for c in classes[:-1])
                                error_message = (
                                    f'the R object loaded from {X} must be a '
                                    f'Seurat object, but has classes '
                                    f'{classes_string} and {classes[-1]!r}')
                            raise TypeError(error_message)
                        self._X, self._obs, self._var, self._obsm, \
                            self._uns = SingleCell._from_seurat(
                                '.SingleCell.seurat_object',
                                assay=assay, slot=X_key)
                        self._varm = {}
                    finally:
                        r('rm(.SingleCell.seurat_object)')
                elif X.endswith('.mtx.gz'):
                    if not os.path.exists(X):
                        error_message = f'10x file {X} does not exist'
                        raise FileNotFoundError(error_message)
                    for prop, prop_name, prop_description in (
                            (obs, 'obs',
                             'a barcodes.tsv.gz file of cell-level metadata'),
                            (var, 'var',
                             'a features.tsv.gz file of gene-level metadata')):
                        if prop is not None and not \
                                isinstance(prop, (str, Path)):
                            error_message = (
                                f'when loading a 10x .mtx.gz file, '
                                f'{prop_name} must be None or the path to '
                                f'{prop_description}')
                            raise TypeError(error_message)
                    for prop, prop_name in \
                            (obsm, 'obsm'), (varm, 'varm'), (uns, 'uns'), \
                            (X_key, 'X_key'), (assay, 'assay'):
                        if prop is not None:
                            error_message = (
                                f'when loading an .h5ad file, {prop_name} '
                                f'must be None')
                            raise ValueError(error_message)
                    from scipy.io import mmread
                    self._X = csr_array(mmread(X).T.tocsr())
                    self._obs = pl.read_csv(
                        f'{os.path.dirname(X)}/barcodes.tsv.gz'
                        if obs is None else obs,
                        has_header=False, new_columns=['cell'])
                    self._var = pl.read_csv(
                        f'{os.path.dirname(X)}/features.tsv.gz'
                        if var is None else var,
                        has_header=False, new_columns=['gene'])
                    self._obsm = {}
                    self._varm = {}
                    self._uns = {}
                else:
                    error_message = (
                        f'X is a filename with unknown extension '
                        f'{".".join(X.split(".")[1:])}; it must be .h5ad '
                        f'(AnnData), .rds (Seurat) or .mtx.gz (10x)')
                    raise ValueError(error_message)
                if obs_columns is not None:
                    self._obs = self._obs.select(obs_columns)
                if var_columns is not None:
                    self._var = self._var.select(var_columns)
        elif is_anndata:
            for prop, prop_name in (obs, 'obs'), (var, 'var'), \
                    (obsm, 'obsm'), (varm, 'varm'), (uns, 'uns'), \
                    (assay, 'assay'):
                if prop is not None:
                    error_message = (
                        f'when initializing a SingleCell dataset from an '
                        f'AnnData object, {prop_name} must be None')
                    raise ValueError(error_message)
            if X_key is None:
                has_layers_UMIs = 'UMIs' in X._layers
                has_raw_X = hasattr(X._raw, '_X')
                if has_layers_UMIs and has_raw_X:
                    error_message = (
                        "both layers['UMIs'] and raw.X are present; this "
                        "should never happen in well-formed AnnData objects")
                    raise ValueError(error_message)
                counts = X._layers['UMIs'] if has_layers_UMIs else \
                    X._raw._X if has_raw_X else X._X
                if not isinstance(counts, (
                        csr_array, csc_array, csr_matrix, csc_matrix)):
                    error_message = (
                        f'to initialize a SingleCell dataset from an AnnData '
                        f'object, its X must be a csr_array, csc_array, '
                        f'csr_matrix, or csc_matrix, but it has type '
                        f'{type(counts).__name__!r}. Either convert X to a '
                        f'csr_array or csc_array, or specify a custom X via '
                        f'the X_key argument')
                    raise TypeError(error_message)
            else:
                check_type(X_key, 'X_key',
                           (csr_array, csc_array, csr_matrix, csc_matrix),
                           'a csr_array, csc_array, csr_matrix, or csc_matrix')
                counts = X_key
            self._X = counts if isinstance(counts, (csr_array, csc_array)) \
                else csr_array(counts) if isinstance(counts, csr_matrix) else \
                    csc_matrix(counts)
            for attr in '_obs', '_var':
                df = getattr(X, attr)
                if df.index.name is None:
                    df = df.rename_axis('_index')  # for consistency with .h5ad
                setattr(self, attr, pl.from_pandas(df, schema_overrides={
                    column: pl.Enum(dtype.categories)
                    for column, dtype in
                    df.dtypes[df.dtypes == 'category'].items()
                }, include_index=True))
            self._obsm: dict[str, np.ndarray] = dict(X._obsm)
            self._varm: dict[str, np.ndarray] = dict(X._varm)
            self._uns = dict(X._uns)
        else:
            error_message = (
                f'X must be a csc_array, csr_array, csc_matrix, or '
                f'csr_matrix, an AnnData object, or an .h5ad (AnnData), .rds '
                f'(Seurat) or .mtx.gz (10x) filename, but has type '
                f'{type(X).__name__!r}')
            raise TypeError(error_message)
        # Check that shapes match
        num_cells, num_genes = self._X.shape
        if len(self._obs) == 0:
            error_message = 'len(obs) is 0: no cells remain'
            raise ValueError(error_message)
        if len(self._var) == 0:
            error_message = 'len(var) is 0: no genes remain'
            raise ValueError(error_message)
        if len(self._obs) != num_cells:
            error_message = (
                f'len(obs) is {len(self._obs):,}, but X.shape[0] is '
                f'{num_cells:,}')
            raise ValueError(error_message)
        if len(self._var) != num_genes:
            error_message = (
                f'len(var) is {len(self._var):,}, but X.shape[1] is '
                f'{num_genes:,}')
            raise ValueError(error_message)
        for key, value in self._obsm.items():
            if len(value) != num_cells:
                error_message = (
                    f'len(obsm[{key!r}]) is {len(value):,}, but X.shape[0] is '
                    f'{num_cells:,}')
                raise ValueError(error_message)
        for key, value in self._varm.items():
            if len(value) != num_genes:
                error_message = (
                    f'len(varm[{key!r}]) is {len(value):,}, but X.shape[0] is '
                    f'{num_genes:,}')
                raise ValueError(error_message)
        # Set `uns['normalized']` and `uns['QCed']` to False if not set yet;
        # if set and not a Boolean, move to `uns['_normalized']`/`uns['_QCed']`
        for key in 'normalized', 'QCed':
            if key in self._uns:
                if not isinstance(self._uns[key], bool):
                    import warnings
                    new_key = f'_{key}'
                    warning_message = (
                        f'uns[{key!r}] already exists and is not Boolean; '
                        f'moving it to uns[{new_key!r}]')
                    warnings.warn(warning_message)
                    self._uns[new_key] = self._uns[key]
                    self._uns[key] = False
            else:
                self._uns[key] = False
    
    @property
    def X(self) -> csr_array | csc_array:
        return self._X
    
    @X.setter
    def X(self, X: csr_array | csc_array | csr_matrix | csc_matrix) -> None:
        if isinstance(X, (csr_array, csc_array)):
            pass
        elif isinstance(X, csr_matrix):
            X = csr_array(X)
        elif isinstance(X, csc_matrix):
            X = csc_array(X)
        else:
            error_message = (
                f'new X must be a csr_array, csc_array, csr_matrix, or '
                f'csc_matrix, but has type {type(X).__name__!r}')
            raise TypeError(error_message)
        if X.shape != self._X.shape:
            error_message = (
                f'new X is {X.shape[0]:,} × {X.shape[1]:,}, but old X is '
                f'{self._X.shape[0]:,} × {self._X.shape[1]:,}')
            raise ValueError(error_message)
        self._X = X
    
    @property
    def obs(self) -> pl.DataFrame:
        return self._obs
    
    @obs.setter
    def obs(self, obs: pl.DataFrame) -> None:
        check_type(obs, 'obs', pl.DataFrame, 'a polars DataFrame')
        if len(obs) != len(self._obs):
            error_message = (
                f'new obs has length {len(obs):,}, but old obs has length '
                f'{len(self._obs):,}')
            raise ValueError(error_message)
        self._obs = obs

    @property
    def var(self) -> pl.DataFrame:
        return self._var
    
    @var.setter
    def var(self, var: pl.DataFrame) -> None:
        check_type(var, 'var', pl.DataFrame, 'a polars DataFrame')
        if len(var) != len(self._var):
            error_message = (
                f'new var has length {len(var):,}, but old var has length '
                f'{len(self._var):,}')
            raise ValueError(error_message)
        self._var = var
    
    @property
    def obsm(self) -> dict[str, np.ndarray[2, Any]]:
        return self._obsm
    
    @obsm.setter
    def obsm(self, obsm: dict[str, np.ndarray[2, Any]]) -> None:
        num_cells = len(self)
        for key, value in obsm.items():
            if not isinstance(key, str):
                error_message = (
                    f'all keys of obsm must be strings, but new obsm contains '
                    f'a key of type {type(key).__name__!r}')
                raise TypeError(error_message)
            if not isinstance(value, np.ndarray):
                error_message = (
                    f'all values of obsm must be NumPy arrays, but new '
                    f'obsm[{key!r}] has type {type(value).__name__!r}')
                raise TypeError(error_message)
            if value.ndim != 2:
                error_message = (
                    f'all values of obsm must be 2D NumPy arrays, but new '
                    f'obsm[{key!r}] is {value.ndim:,}D')
                raise ValueError(error_message)
            if len(obsm) != num_cells:
                error_message = (
                    f'the length of new obsm[{key!r}] is {len(value):,}, but '
                    f'X.shape[0] is {self._X.shape[0]:,}')
                raise ValueError(error_message)
        self._obsm = obsm
    
    @property
    def varm(self) -> dict[str, np.ndarray[2, Any]]:
        return self._varm
    
    @varm.setter
    def varm(self, varm: dict[str, np.ndarray[2, Any]]) -> None:
        num_cells = len(self)
        for key, value in varm.items():
            if not isinstance(key, str):
                error_message = (
                    f'all keys of varm must be strings, but new varm '
                    f'contains a key of type {type(key).__name__!r}')
                raise TypeError(error_message)
            if not isinstance(value, np.ndarray):
                error_message = (
                    f'all values of varm must be NumPy arrays, but new '
                    f'varm[{key!r}] has type {type(value).__name__!r}')
                raise TypeError(error_message)
            if value.ndim != 2:
                error_message = (
                    f'all values of varm must be 2D NumPy arrays, but new '
                    f'varm[{key!r}] is {value.ndim:,}D')
                raise ValueError(error_message)
            if len(varm) != num_cells:
                error_message = (
                    f'the length of new varm[{key!r}] is {len(value):,}, but '
                    f'X.shape[0] is {self._X.shape[0]:,}')
                raise ValueError(error_message)
        self._varm = varm
    
    @property
    def uns(self) -> NestedScalarOrArrayDict:
        return self._uns
    
    @uns.setter
    def uns(self, uns: NestedScalarOrArrayDict) -> None:
        valid_uns_types = str, int, np.integer, float, np.floating, \
            bool, np.bool_, np.ndarray
        for description, value in SingleCell._iter_uns(uns):
            if not isinstance(value, valid_uns_types):
                error_message = (
                    f'all values of uns must be scalars (strings, numbers or '
                    f'Booleans) or NumPy arrays, or nested dictionaries '
                    f'thereof, but {description} has type '
                    f'{type(value).__name__!r}')
                raise TypeError(error_message)
        self._uns = uns
    
    @staticmethod
    def _iter_uns(uns: NestedScalarOrArrayDict, *, prefix: str = 'uns') -> \
            Iterable[tuple[str, str | int | np.integer | float | np.floating |
                                bool | np.bool_ | np.ndarray[Any, Any]]]:
        """
        Recurse through uns, yielding tuples of a string describing each key
        (e.g. "uns['a']['b']") and the corresponding value.
        
        Args:
            uns: an uns dictionary

        Yields:
            Length-2 tuples where the first element is a string describing each
            key, and the second element is the corresponding value.
        """
        for key, value in uns.items():
            key = f'{prefix}[{key!r}]'
            if isinstance(value, dict):
                SingleCell._iter_uns(value, prefix=key)
            else:
                yield key, value
    
    @staticmethod
    def _read_uns(uns_group: h5py.Group) -> NestedScalarOrArrayDict:
        """
        Recursively load uns from an .h5ad file.
        
        Args:
            uns_group: uns as an h5py.Group

        Returns:
            The loaded uns.
        """
        return {key: SingleCell._read_uns(value)
                     if isinstance(value, h5py.Group) else
                     pl.Series(value[()]).cast(pl.String).to_numpy()
                     if value.attrs['encoding-type'] == 'string-array' else
                     value[()].decode('utf-8')
                     if value.attrs['encoding-type'] == 'string' else
                     value[()].item()
                for key, value in uns_group.items()}
    
    @staticmethod
    def _save_uns(uns: NestedScalarOrArrayDict,
                  uns_group: h5py.Group,
                  h5ad_file: h5py.File) -> None:
        """
        Recursively save uns to an .h5ad file.
        
        Args:
            uns: an uns dictionary
            uns_group: uns as an h5py.Group
            h5ad_file: an `h5py.File` open in write mode
        """
        uns_group.attrs['encoding-type'] = 'dict'
        uns_group.attrs['encoding-version'] = '0.1.0'
        for key, value in uns.items():
            if isinstance(value, dict):
                SingleCell._save_uns(value, uns_group.create_group(key),
                                     h5ad_file)
            else:
                dataset = uns_group.create_dataset(key, data=value)
                dataset.attrs['encoding-type'] = \
                    ('string-array' if value.dtype == object else 'array') \
                    if isinstance(value, np.ndarray) else \
                    'string' if isinstance(value, str) else 'numeric-scalar'
                dataset.attrs['encoding-version'] = '0.2.0'
    
    @property
    def obs_names(self) -> pl.Series:
        return self._obs[:, 0]
    
    @property
    def var_names(self) -> pl.Series:
        return self._var[:, 0]
    
    def set_obs_names(self, column: str) -> SingleCell:
        """
        Sets a column as the new first column of obs, i.e. the obs_names.
        
        Args:
            column: the column name in obs; must have String, Categorical, or
                    Enum data type

        Returns:
            A new SingleCell dataset with `column` as the first column of obs.
            If `column` is already the first column, return this dataset
            unchanged.
        """
        obs = self._obs
        check_type(column, 'column', str, 'a string')
        if column == obs.columns[0]:
            return self
        if column not in obs:
            error_message = f'{column!r} is not a column of obs'
            raise ValueError(error_message)
        check_dtype(obs[column], f'obs[{column!r}]',
                    (pl.String, pl.Categorical, pl.Enum))
        # noinspection PyTypeChecker
        return SingleCell(X=self._X,
                          obs=obs.select(column, pl.exclude(column)),
                          var=self._var, obsm=self._obsm, varm=self._varm,
                          uns=self._uns)
    
    def set_var_names(self, column: str) -> SingleCell:
        """
        Sets a column as the new first column of var, i.e. the var_names.
        
        Args:
            column: the column name in var; must have String, Categorical, or
                    Enum data type

        Returns:
            A new SingleCell dataset with `column` as the first column of var.
            If `column` is already the first column, return this dataset
            unchanged.
        """
        var = self._var
        check_type(column, 'column', str, 'a string')
        if column == var.columns[0]:
            return self
        if column not in var:
            error_message = f'{column!r} is not a column of var'
            raise ValueError(error_message)
        check_dtype(self._var[column], f'var[{column!r}]',
                    (pl.String, pl.Categorical, pl.Enum))
        # noinspection PyTypeChecker
        return SingleCell(X=self._X, obs=self._obs,
                          var=var.select(column, pl.exclude(column)),
                          obsm=self._obsm, varm=self._varm, uns=self._uns)
    
    @staticmethod
    def _read_datasets(datasets: Sequence[h5py.Dataset],
                       num_threads: int | np.integer) -> \
            dict[str, np.ndarray[1, Any]] | None:
        """
        Read a sequence of HDF5 datasets into a dictionary of 1D NumPy arrays.
        Assume all are from the same file (this is not checked).
        
        Args:
            datasets: a sequence of h5py.Datasets to read
            num_threads: the number of threads to use when reading; if >1,
                         spawn multiple processes and read into shared memory

        Returns:
            A dictionary of 1D NumPy arrays with the contents of the datasets;
            the keys are taken from each dataset's dataset.name.
            If len(datasets) == 0, return None.
        """
        if len(datasets) == 0:
            return
        import multiprocessing
        # noinspection PyUnresolvedReferences
        from multiprocessing.sharedctypes import _new_value
        # Allocate shared memory for each dataset. Use _new_value() instead of
        # multiprocessing.Array() to avoid the memset at github.com/python/
        # cpython/blob/main/Lib/multiprocessing/sharedctypes.py#L62.
        # noinspection PyTypeChecker
        buffers = {dataset.name: _new_value(
            len(dataset) * np.ctypeslib.as_ctypes_type(dataset.dtype))
            for dataset in datasets}
        # Spawn num_threads processes: the first loads the first len(dataset) /
        # num_threads elements of each dataset, the second loads the next
        # len(dataset) / num_threads, etc. Because the chunks loaded by each
        # process are non-overlapping, there's no need to lock.
        filename = datasets[0].file.filename
        
        def read_dataset_chunks(thread_index: int) -> None:
            try:
                with h5py.File(filename) as h5ad_file:
                    for dataset_name, buffer in buffers.items():
                        chunk_size = len(buffer) // num_threads
                        start = thread_index * chunk_size
                        end = len(buffer) \
                            if thread_index == num_threads - 1 else \
                            start + chunk_size
                        chunk = np.s_[start:end]
                        dataset = h5ad_file[dataset_name]
                        dataset.read_direct(np.frombuffer(
                            buffer, dtype=dataset),
                            source_sel=chunk, dest_sel=chunk)
            except KeyboardInterrupt:
                pass  # do not print KeyboardInterrupt tracebacks, just return
        
        processes = []
        for thread_index in range(num_threads):
            process = multiprocessing.Process(
                target=read_dataset_chunks, args=(thread_index,))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        # Wrap the shared memory in NumPy arrays
        arrays = {
            dataset_name: np.frombuffer(buffer, dtype=dataset.dtype)
            for (dataset_name, buffer), dataset in
            zip(buffers.items(), datasets)}
        return arrays
    
    @staticmethod
    def _read_dataset(dataset: h5py.Dataset,
                      num_threads: int | np.integer,
                      preloaded_datasets: dict[str, h5py.Dataset] |
                                          None = None) -> np.ndarray[1, Any]:
        """
        Read an HDF5 dataset into a 1D NumPy array.
        
        Args:
            dataset: the h5py.Dataset to read
            num_threads: the number of threads to use for reading; if >1, spawn
                         multiple processes and read into shared memory, unless
                         - the array is small enough (heuristically, under 10k
                           elements) that it's probably faster to read serially
                         - the array is so small that there's less than one
                           element to read per thread
                         - the array has `dtype=object` (not compatible with
                           shared memory)
            preloaded_datasets: a dictionary of preloaded datasets; if
                                dataset.name is in preloaded_dataset; it will
                                just return preloaded_datasets[dataset.name]
        
        Returns:
            A 1D NumPy array with the contents of the dataset.
        """
        if preloaded_datasets is not None and \
                dataset.name in preloaded_datasets:
            return preloaded_datasets[dataset.name]
        dtype = dataset.dtype
        if num_threads == 1 or dtype == object or \
                len(dataset) < 10_000 or len(dataset) < num_threads:
            return dataset[:]
        else:
            return SingleCell._read_datasets([dataset], num_threads)\
                [dataset.name]
    
    @staticmethod
    def _read_h5ad_dataframe(h5ad_file: h5py.File,
                             key: Literal['obs', 'var'],
                             columns: str | Sequence[str] | None = None,
                             num_threads: int | np.integer = 1) -> \
            pl.DataFrame:
        """
        Load obs or var from an .h5ad file as a polars DataFrame.
        
        Args:
            h5ad_file: an `h5py.File` open in read mode
            key: the key to load as a DataFrame; must be `'obs'` or `'var'`
            columns: the column(s) of the DataFrame to load; the index column
                     is always loaded as the first column, regardless of
                     whether it is specified here, and then the remaining
                     columns are loaded in the order specified
            num_threads: the number of threads to use when reading
        
        Returns:
            A polars DataFrame of the data in h5ad_file[key].
        """
        group = h5ad_file[key]
        # Special case: the entire obs or var may rarely be a single NumPy
        # structured array (dtype=void)
        if isinstance(group, h5py.Dataset) and \
                np.issubdtype(group.dtype, np.void):
            data = pl.from_numpy(group[:])
            data = data.with_columns(pl.col(pl.Binary).cast(pl.String))
            return data
        check_type(group, 'h5ad_file[key]', h5py.Group, 'an h5py.Group')
        # If reading in parallel, preload non-string datasets in parallel
        if num_threads > 1:
            datasets = []
            h5ad_file[key].visititems(
                lambda name, node: datasets.append(node)
                if isinstance(node, h5py.Dataset) and node.dtype != object
                and len(node) >= 10_000 and len(node) >= num_threads else None)
            preloaded_datasets = \
                SingleCell._read_datasets(datasets, num_threads)
        else:
            preloaded_datasets = None
        data = {}
        if columns is None:
            columns = group.attrs['column-order']
        else:
            columns = [column for column in to_tuple(columns)
                       if column != group.attrs['_index']]
            for column in columns:
                if column not in group.attrs['column-order']:
                    error_message = f'{column!r} is not a column of {key}'
                    raise ValueError(error_message)
        for column in chain((group.attrs['_index'],), columns):
            value = group[column]
            encoding_type = value.attrs.get('encoding-type')
            if encoding_type == 'categorical' or (
                    isinstance(value, h5py.Group) and all(
                    key == 'categories' or key == 'codes'
                    for key in value.keys())) or 'categories' in value.attrs:
                # Sometimes, the categories are stored in a different place
                # which is pointed to by value.attrs['categories']
                if 'categories' in value.attrs:
                    category_object = h5ad_file[value.attrs['categories']]
                    category_encoding_type = None
                    # noinspection PyTypeChecker
                    codes = SingleCell._read_dataset(
                        value, num_threads, preloaded_datasets)
                else:
                    category_object = value['categories']
                    category_encoding_type = \
                        category_object.attrs.get('encoding-type')
                    codes = SingleCell._read_dataset(
                        value['codes'], num_threads, preloaded_datasets)
                # Sometimes, the categories are themselves nullable
                # integer or Boolean arrays
                if category_encoding_type == 'nullable-integer' or \
                        category_encoding_type == 'nullable-boolean' or (
                        isinstance(category_object, h5py.Group) and all(
                        key == 'values' or key == 'mask'
                        for key in category_object.keys())):
                    data[column] = pl.Series(SingleCell._read_dataset(
                        category_object['values'], num_threads,
                        preloaded_datasets)[codes])
                    mask = pl.Series(SingleCell._read_dataset(
                        category_object['mask'], num_threads,
                        preloaded_datasets)[codes] | (codes == -1))
                    has_missing = mask.any()
                    if has_missing:
                        data[column] = data[column].set(mask, None)
                    continue
                # noinspection PyTypeChecker
                categories = SingleCell._read_dataset(
                    category_object, num_threads, preloaded_datasets)
                mask = pl.Series(codes == -1)
                has_missing = mask.any()
                # polars does not (as of version 0.20.2) support Categoricals
                # or Enums with non-string categories, so if the categories are
                # not strings, just map the codes to the categories.
                if category_encoding_type == 'array' or (
                        isinstance(category_object, h5py.Dataset) and
                        category_object.dtype != object):
                    data[column] = pl.Series(categories[codes],
                                             nan_to_null=True)
                    if has_missing:
                        data[column] = data[column].set(mask, None)
                elif category_encoding_type == 'string-array' or (
                        isinstance(category_object, h5py.Dataset) and
                        category_object.dtype == object):
                    if has_missing:
                        codes[mask] = 0
                    data[column] = pl.Series(codes, dtype=pl.UInt32)
                    if has_missing:
                        data[column] = data[column].set(mask, None)
                    # noinspection PyUnresolvedReferences
                    data[column] = data[column].cast(
                        pl.Enum(pl.Series(categories).cast(pl.String)))
                else:
                    encoding = \
                        f'encoding-type {category_encoding_type!r}' \
                        if category_encoding_type is not None else \
                            'encoding'
                    error_message = (
                        f'{column!r} column of {key!r} is a categorical '
                        f'with unsupported {encoding}')
                    raise ValueError(error_message)
            elif encoding_type == 'nullable-integer' or \
                    encoding_type == 'nullable-boolean' or (
                    isinstance(value, h5py.Group) and all(
                    key == 'values' or key == 'mask' for key in value.keys())):
                values = SingleCell._read_dataset(
                    value['values'], num_threads, preloaded_datasets)
                mask = SingleCell._read_dataset(
                    value['mask'], num_threads, preloaded_datasets)
                data[column] = pl.Series(values).set(pl.Series(mask), None)
            elif encoding_type == 'array' or (
                    isinstance(value, h5py.Dataset) and value.dtype != object):
                data[column] = pl.Series(SingleCell._read_dataset(
                    value, num_threads, preloaded_datasets), nan_to_null=True)
            elif encoding_type == 'string-array' or (
                    isinstance(value, h5py.Dataset) and value.dtype == object):
                data[column] = SingleCell._read_dataset(
                    value, num_threads, preloaded_datasets)
            else:
                encoding = f'encoding-type {encoding_type!r}' \
                    if encoding_type is not None else 'encoding'
                error_message = \
                    f'{column!r} column of {key!r} has unsupported {encoding}'
                raise ValueError(error_message)
        # NumPy doesn't support encoding object-dtyped string arrays as UTF-8,
        # so do the conversion in polars instead
        data = pl.DataFrame(data)\
            .with_columns(pl.col(pl.Binary).cast(pl.String))
        return data
    
    @staticmethod
    def read_obs(h5ad_file: h5py.File | str | Path,
                 columns: str | Iterable[str] | None = None,
                 num_threads: int | np.integer | None = 1) -> pl.DataFrame:
        """
        Load just obs from an .h5ad file as a polars DataFrame.
        
        Args:
            h5ad_file: an .h5ad filename
            columns: the column(s) of obs to load; if None, load all columns
            num_threads: the number of threads to use when reading; set
                         `num_threads=None` to use all available cores (as
                         determined by `os.cpu_count()`)
    
        Returns:
            A polars DataFrame of the data in obs.
        """
        check_type(h5ad_file, 'h5ad_file', (str, Path),
                   'a string or pathlib.Path')
        if columns is not None:
            columns = to_tuple(columns)
            if len(columns) == 0:
                error_message = 'no columns were specified'
                raise ValueError(error_message)
            check_types(columns, 'columns', str, 'strings')
        if num_threads is None:
            num_threads = os.cpu_count()
        else:
            check_type(num_threads, 'num_threads', int, 'a positive integer')
            check_bounds(num_threads, 'num_threads', 1)
        with h5py.File(h5ad_file) as f:
            return SingleCell._read_h5ad_dataframe(
                f, 'obs', columns=columns, num_threads=num_threads)
    
    @staticmethod
    def read_var(h5ad_file: str | Path,
                 columns: str | Iterable[str] | None = None,
                 num_threads: int | np.integer | None = 1) -> pl.DataFrame:
        """
        Load just var from an .h5ad file as a polars DataFrame.
        
        Args:
            h5ad_file: an .h5ad filename
            columns: the column(s) of var to load; if None, load all columns
            num_threads: the number of threads to use when reading; set
                         `num_threads=None` to use all available cores (as
                         determined by `os.cpu_count()`)
    
        Returns:
            A polars DataFrame of the data in var.
        """
        check_type(h5ad_file, 'h5ad_file', (str, Path),
                   'a string or pathlib.Path')
        if columns is not None:
            columns = to_tuple(columns)
            if len(columns) == 0:
                error_message = 'no columns were specified'
                raise ValueError(error_message)
            check_types(columns, 'columns', str, 'strings')
        if num_threads is None:
            num_threads = os.cpu_count()
        else:
            check_type(num_threads, 'num_threads', int, 'a positive integer')
            check_bounds(num_threads, 'num_threads', 1)
        with h5py.File(h5ad_file) as f:
            return SingleCell._read_h5ad_dataframe(
                f, 'var', columns=columns, num_threads=num_threads)
    
    @staticmethod
    def read_obsm(h5ad_file: str | Path,
                  keys: str | Iterable[str] | None = None) -> \
            np.ndarray[2, Any]:
        """
        Load just obsm from an .h5ad file as a polars DataFrame.
        
        Args:
            h5ad_file: an .h5ad filename
            keys: the keys(s) of obsm to load; if None, load all keys
        
        Returns:
            A dictionary of NumPy arrays of the data in obsm.
        """
        check_type(h5ad_file, 'h5ad_file', (str, Path),
                   'a string or pathlib.Path')
        if keys is not None:
            keys = to_tuple(keys)
            if len(keys) == 0:
                error_message = 'no keys were specified'
                raise ValueError(error_message)
            check_types(keys, 'keys', str, 'strings')
        with h5py.File(h5ad_file) as f:
            if 'obsm' in f:
                obsm = f['obsm']
                if keys is None:
                    return {key: value[:] for key, value in obsm.items()}
                else:
                    for key_index, key in keys.items():
                        if key not in obsm:
                            error_message = (
                                f'keys[{key_index}] is {key!r}, which is not '
                                f'a key of obsm')
                            raise ValueError(error_message)
                    return {key: obsm[key][:] for key in keys}
            else:
                if keys is not None:
                    error_message = 'keys was specified, but obsm is empty'
                    raise ValueError(error_message)
                return {}
         
    @staticmethod
    def read_varm(h5ad_file: str | Path,
                  keys: str | Iterable[str] | None = None) -> \
            np.ndarray[2, Any]:
        """
        Load just varm from an .h5ad file as a polars DataFrame.
        
        Args:
            h5ad_file: an .h5ad filename
            keys: the keys(s) of varm to load; if None, load all keys
        
        Returns:
            A dictionary of NumPy arrays of the data in varm.
        """
        check_type(h5ad_file, 'h5ad_file', (str, Path),
                   'a string or pathlib.Path')
        if keys is not None:
            keys = to_tuple(keys)
            if len(keys) == 0:
                error_message = 'no keys were specified'
                raise ValueError(error_message)
            check_types(keys, 'keys', str, 'strings')
        with h5py.File(h5ad_file) as f:
            if 'varm' in f:
                varm = f['varm']
                if keys is None:
                    return {key: value[:] for key, value in varm.items()}
                else:
                    for key_index, key in keys.items():
                        if key not in varm:
                            error_message = (
                                f'keys[{key_index}] is {key!r}, which is not '
                                f'a key of varm')
                            raise ValueError(error_message)
                    return {key: varm[key][:] for key in keys}
            else:
                if keys is not None:
                    error_message = 'keys was specified, but varm is empty'
                    raise ValueError(error_message)
                return {}                
    
    @staticmethod
    def read_uns(h5ad_file: str | Path) -> NestedScalarOrArrayDict:
        """
        Load just uns from an .h5ad file as a dictionary.
        
        Args:
            h5ad_file: an .h5ad filename
        
        Returns:
            A dictionary of the data in uns.
        """
        check_type(h5ad_file, 'h5ad_file', (str, Path),
                   'a string or pathlib.Path')
        with h5py.File(h5ad_file) as f:
            if 'uns' in f:
                return SingleCell._read_uns(f['uns'])
            else:
                return {}
    
    @staticmethod
    def _print_matrix_info(X: h5py.Group | h5py.Dataset, X_name: str) -> None:
        """
        Given a key of an .h5ad file representing a sparse or dense matrix,
        print its shape, data type and (if sparse) numbr of non-zero elements.
        
        Args:
            X: the key in the .h5ad file representing the matrix, as a Group or
               Dataset object
            X_name: the name of the key
        """
        is_sparse = isinstance(X, h5py.Group)
        if is_sparse:
            data = X['data']
            shape = X.attrs['shape'] if 'shape' in X.attrs else \
                X.attrs['h5sparse_shape']
            dtype = str(data.dtype)
            nnz = data.shape[0]
            print(f'{X_name}: {shape[0]:,} × {shape[1]:,} sparse matrix with '
                  f'{nnz:,} non-zero elements, data type {dtype!r}, and '
                  f'first non-zero element = {data[0]:.6g}')
        else:
            shape = X.shape
            dtype = str(X.dtype)
            print(f'{X_name}: {shape[0]:,} × {shape[1]:,} dense matrix with '
                  f'data type {dtype!r} and first non-zero element = '
                  f'{X[0, 0]:.6g}')
    
    @staticmethod
    def ls(h5ad_file: str | Path) -> None:
        """
        Print the fields in an .h5ad file. This can be useful e.g. when
        deciding which count matrix to load via the `X_key` argument to
        `SingleCell()`.
        
        Args:
            h5ad_file: an .h5ad filename
        """
        check_type(h5ad_file, 'h5ad_file', (str, Path),
                   'a string or pathlib.Path')
        h5ad_file = str(h5ad_file)
        if not h5ad_file.endswith('.h5ad'):
            error_message = f"h5ad file {h5ad_file!r} must end with '.h5ad'"
            raise ValueError(error_message)
        if not os.path.exists(h5ad_file):
            error_message = f'h5ad file {h5ad_file!r} does not exist'
            raise FileNotFoundError(error_message)
        terminal_width = os.get_terminal_size().columns
        attrs = 'obs', 'var', 'obsm', 'varm', 'obsp', 'varp', 'uns'
        with h5py.File(h5ad_file) as h5ad_file:
            # X
            SingleCell._print_matrix_info(h5ad_file['X'], 'X')
            # layers
            if 'layers' in h5ad_file:
                layers = h5ad_file['layers']
                if len(layers) > 0:
                    for layer_name, layer in layers.items():
                        SingleCell._print_matrix_info(
                            layer, f'layers[{layer_name!r}]')
            # obs, var, obsm, varm, obsp, varp, uns
            for attr in attrs:
                if attr in h5ad_file:
                    entries = h5ad_file[attr]
                    if (attr == 'obs' or attr == 'var') and \
                            isinstance(entries, h5py.Dataset) and \
                            np.issubdtype(entries.dtype, np.void):
                        entries = entries.dtype.fields
                    if len(entries) > 0:
                        print(fill(f'{attr}: {", ".join(entries)}',
                                   width=terminal_width,
                                   subsequent_indent=' ' * (len(attr) + 2)))
            # raw
            if 'raw' in h5ad_file:
                raw = h5ad_file['raw']
                if len(raw) > 0:
                    print('raw:')
                    if 'X' in raw:
                        SingleCell._print_matrix_info(raw['X'], '    X')
                    if 'layers' in raw:
                        layers = raw['layers']
                        if len(layers) > 0:
                            for layer_name, layer in layers.items():
                                SingleCell._print_matrix_info(
                                    layer, f'    layers[{layer_name!r}]')
                    for attr in attrs:
                        if attr in raw:
                            entries = raw[attr]
                            if (attr == 'obs' or attr == 'var') and \
                                    isinstance(entries, h5py.Dataset) and \
                                    np.issubdtype(entries.dtype, np.void):
                                entries = entries.dtype.fields
                            if len(entries) > 0:
                                print(fill(f'    {attr}: {", ".join(entries)}',
                                           width=terminal_width,
                                           subsequent_indent=' ' * (
                                                   len(attr) + 6)))
    
    def __eq__(self, other: SingleCell) -> bool:
        """
        Test for equality with another SingleCell dataset.
        
        Args:
            other: the other SingleCell dataset to test for equality with

        Returns:
            Whether the two SingleCell datasets are identical.
        """
        if not isinstance(other, SingleCell):
            error_message = (
                f'the left-hand operand of `==` is a SingleCell dataset, but '
                f'the right-hand operand has type {type(other).__name__!r}')
            raise TypeError(error_message)
        # noinspection PyUnresolvedReferences
        return self._obs.equals(other._obs) and \
               self._var.equals(other._var) and \
               self._obsm.keys() == other._obsm.keys() and \
               self._varm.keys() == other._varm.keys() and \
            all((self._obsm[key] == other._obsm[key]).all()
                for key in self._obsm) and \
            all((self._varm[key] == other._varm[key]).all()
                for key in self._varm) and \
            SingleCell._eq_uns(self._uns, other._uns) and \
            self._X.nnz == other._X.nnz and not (self._X != other._X).nnz
    
    @staticmethod
    def _eq_uns(uns: NestedScalarOrArrayDict,
                other_uns: NestedScalarOrArrayDict) -> bool:
        """
        Test whether two uns are equal.
        
        Args:
            uns: an uns
            other_uns: another uns

        Returns:
            Whether the two uns are equal.
        """
        return uns.keys() == other_uns.keys() and all(
            isinstance(value, dict) and isinstance(other_value, dict) and
            SingleCell._eq_uns(value, other_value) or
            isinstance(value, np.ndarray) and
            isinstance(other_value, np.ndarray) and
            (value == other_value).all() or
            not isinstance(other_value, (dict, np.ndarray)) and
            value == other_value
            for (key, value), (other_key, other_value) in
            zip(uns.items(), other_uns.items()))
    
    @staticmethod
    def _getitem_error(item: Indexer) -> None:
        """
        Raise an error if the indexer is invalid.
        
        Args:
            item: the indexer
        """
        types = tuple(type(elem).__name__ for elem in to_tuple(item))
        if len(types) == 1:
            types = types[0]
        error_message = (
            f'SingleCell indices must be cells, a length-1 tuple of (cells,), '
            f'or a length-2 tuple of (cells, genes). Cells and genes must '
            f'each be a string or integer; a slice of strings or integers; or '
            f'a list, NumPy array, or polars Series of strings, integers, or '
            f'Booleans. You indexed with: {types}.')
        raise ValueError(error_message)
    
    @staticmethod
    def _getitem_by_string(df: pl.DataFrame, string: str) -> int:
        """
        Get the index where df[:, 0] == string, raising an error if no rows or
        multiple rows match.
        
        Args:
            df: a DataFrame (obs or var)
            string: the string to find the index of in the first column of df

        Returns:
            The integer index of the string within the first column of df.
        """
        first_column = df.columns[0]
        try:
            return df\
                .select(first_column)\
                .with_row_index('_SingleCell_getitem')\
                .row(by_predicate=pl.col(first_column) == string)\
                [0]
        except pl.exceptions.NoRowsReturnedError:
            raise KeyError(string)
    
    @staticmethod
    def _getitem_process(item: Indexer, index: int, df: pl.DataFrame) -> \
            list[int] | slice | pl.Series:
        """
        Process an element of an item passed to __getitem__().
        
        Args:
            item: the item
            index: the index of the element to process
            df: the DataFrame (obs or var) to process the element with respect
                to

        Returns:
            A new indexer indicating the rows/columns to index.
        """
        subitem = item[index]
        if is_integer(subitem):
            return [subitem]
        elif isinstance(subitem, str):
            return [SingleCell._getitem_by_string(df, subitem)]
        elif isinstance(subitem, slice):
            start = subitem.start
            stop = subitem.stop
            step = subitem.step
            if isinstance(start, str):
                start = SingleCell._getitem_by_string(df, start)
            elif start is not None and not is_integer(start):
                SingleCell._getitem_error(item)
            if isinstance(stop, str):
                stop = SingleCell._getitem_by_string(df, stop)
            elif stop is not None and not is_integer(stop):
                SingleCell._getitem_error(item)
            if step is not None and not is_integer(step):
                SingleCell._getitem_error(item)
            return slice(start, stop, step)
        elif isinstance(subitem, (list, np.ndarray, pl.Series)):
            if not isinstance(subitem, pl.Series):
                subitem = pl.Series(subitem)
            if subitem.is_null().any():
                error_message = 'your indexer contains missing values'
                raise ValueError(error_message)
            if subitem.dtype == pl.String or subitem.dtype == \
                    pl.Categorical or subitem.dtype == pl.Enum:
                indices = subitem\
                    .to_frame(df.columns[0])\
                    .join(df.with_row_index('_SingleCell_index'),
                          on=df.columns[0], how='left')\
                    ['_SingleCell_index']
                if indices.null_count():
                    error_message = subitem.filter(indices.is_null())[0]
                    raise KeyError(error_message)
                return indices
            elif subitem.dtype.is_integer() or subitem.dtype == pl.Boolean:
                return subitem
            else:
                SingleCell._getitem_error(item)
        else:
            SingleCell._getitem_error(item)
            
    def __getitem__(self, item: Indexer | tuple[Indexer, Indexer]) -> \
            SingleCell:
        """
        Subset to specific cell(s) and/or gene(s).
        
        Index with a tuple of `(cells, genes)`. If `cells` and `genes` are
        integers, arrays/lists/slices of integers, or arrays/lists of Booleans,
        the result will be a SingleCell dataset subset to `X[cells, genes]`,
        `obs[cells]`, `var[genes]`, `obsm[cells]`, and `varm[genes]`. However,
        `cells` and/or `genes` can instead be strings (or arrays or slices of
        strings), in which case they refer to the first column of obs
        (`obs_names`) and/or var (`var_names`), respectively.
        
        Examples:
        - Subset to one cell, for all genes:
          sc['CGAATTGGTGACAGGT-L8TX_210916_01_B05-1131590416']
          sc[2]
        - Subset to one gene, for all cells:
          sc[:, 'APOE']
          sc[:, 13196]
        - Subset to one cell and one gene:
          sc['CGAATTGGTGACAGGT-L8TX_210916_01_B05-1131590416', 'APOE']
          sc[2, 13196]
        - Subset to a range of cells and genes:
          sc['CGAATTGGTGACAGGT-L8TX_210916_01_B05-1131590416':
             'CCCTCTCAGCAGCCTC-L8TX_211007_01_A09-1135034522',
             'APOE':'TREM2']
          sc[2:6, 13196:34268]
        - Subset to specific cells and genes:
          sc[['CGAATTGGTGACAGGT-L8TX_210916_01_B05-1131590416',
              'CCCTCTCAGCAGCCTC-L8TX_211007_01_A09-1135034522']]
          sc[:, pl.Series(['APOE', 'TREM2'])]
          sc[['CGAATTGGTGACAGGT-L8TX_210916_01_B05-1131590416',
              'CCCTCTCAGCAGCCTC-L8TX_211007_01_A09-1135034522'],
              np.array(['APOE', 'TREM2'])]
        
        Args:
            item: the item to index with
        
        Returns:
            A new SingleCell dataset subset to the specified cells and/or
            genes.
        """
        if not isinstance(item, (int, str, slice, tuple, list,
                                 np.ndarray, pl.Series)):
            error_message = (
                f'SingleCell datasets must be indexed with an integer, '
                f'string, slice, tuple, list, NumPy array, or polars Series, '
                f'but you tried to index with an object of type '
                f'{type(item).__name__!r}')
            raise TypeError(error_message)
        if isinstance(item, tuple):
            if not 1 <= len(item) <= 2:
                self._getitem_error(item)
        else:
            item = item,
        rows = self._getitem_process(item, 0, self._obs)
        if isinstance(rows, pl.Series):
            obs = self._obs.filter(rows) \
                if rows.dtype == pl.Boolean else self._obs[rows]
            rows = rows.to_numpy()
        else:
            obs = self._obs[rows]
        obsm = {key: value[rows] for key, value in self._obsm.items()}
        if len(item) == 1:
            return SingleCell(X=self._X[rows], obs=obs, var=self._var,
                              obsm=obsm, varm=self._varm, uns=self._uns)
        columns = self._getitem_process(item, 1, self._var)
        if isinstance(columns, pl.Series):
            var = self._var.filter(columns) \
                if columns.dtype == pl.Boolean else self._var[columns]
            columns = columns.to_numpy()
        else:
            var = self._var[columns]
        varm = {key: value[rows] for key, value in self._varm.items()}
        X = self._X[rows, columns] \
            if isinstance(rows, slice) or isinstance(columns, slice) else \
            self._X[np.ix_(rows, columns)]
        return SingleCell(X=X, obs=obs, var=var, obsm=obsm, varm=varm,
                          uns=self._uns)
    
    def __len__(self) -> int:
        """
        Get the number of cells in this SingleCell dataset.
        
        Returns:
            The number of cells.
        """
        return self._X.shape[0]
       
    def __repr__(self) -> str:
        """
        Get a string representation of this SingleCell dataset.
        
        Returns:
            A string summarizing the dataset.
        """
        descr = (
            f'SingleCell dataset with {len(self._obs):,} '
            f'{plural("cell", len(self._obs))} (obs), {len(self._var):,} '
            f'{plural("gene", len(self._var))} (var), and {self._X.nnz:,} '
            f'non-zero {"entries" if self._X.nnz != 1 else "entry"} (X)')
        terminal_width = os.get_terminal_size().columns
        for attr in 'obs', 'var', 'obsm', 'varm', 'uns':
            entries = getattr(self, attr).columns \
                if attr == 'obs' or attr == 'var' else getattr(self, attr)
            if len(entries) > 0:
                descr += '\n' + fill(
                    f'    {attr}: {", ".join(entries)}',
                    width=terminal_width,
                    subsequent_indent=' ' * (len(attr) + 6))
        return descr
    
    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of this SingleCell dataset.
        
        Returns:
            A length-2 tuple where the first element is the number of cells,
            and the second is the number of genes.
        """
        return self._X.shape
        
    @staticmethod
    def _write_h5ad_dataframe(h5ad_file: h5py.File,
                              df: pl.DataFrame,
                              key: Literal['obs', 'var'],
                              preserve_strings: bool) -> None:
        """
        Write obs or var to an .h5ad file.
        
        Args:
            h5ad_file: an `h5py.File` open in write mode
            df: the DataFrame to write (obs or var)
            key: the name of the DataFrame; must be `'obs'` or `'var'`
            preserve_strings: if False, encode string columns with duplicate
                              values as Enums to save space; if True, preserve
                              these columns as string columns
        """
        # Create a group for the data frame and add top-level metadata
        group = h5ad_file.create_group(key)
        group.attrs['_index'] = df.columns[0]
        group.attrs['column-order'] = df.columns[1:]
        group.attrs['encoding-type'] = 'dataframe'
        group.attrs['encoding-version'] = '0.2.0'
        for column in df:
            dtype = column.dtype
            if dtype == pl.String:
                if column.null_count() or \
                        not preserve_strings and column.is_duplicated().any():
                    column = column\
                        .cast(pl.Enum(column.unique(maintain_order=True)
                                      .drop_nulls()))
                    dtype = column.dtype
                else:
                    dataset = group.create_dataset(column.name,
                                                   data=column.to_numpy())
                    dataset.attrs['encoding-type'] = 'string-array'
                    dataset.attrs['encoding-version'] = '0.2.0'
                    continue
            if dtype == pl.Enum or dtype == pl.Categorical:
                is_Enum = dtype == pl.Enum
                subgroup = group.create_group(column.name)
                subgroup.attrs['encoding-type'] = 'categorical'
                subgroup.attrs['encoding-version'] = '0.2.0'
                subgroup.attrs['ordered'] = is_Enum
                categories = column.cat.get_categories()
                if not is_Enum:
                    column = column.cast(pl.Enum(categories))
                codes = column.to_physical().fill_null(-1)
                subgroup.create_dataset('codes', data=codes.to_numpy())
                if len(categories) == 0:
                    subgroup.create_dataset('categories', shape=(0,),
                                            dtype=h5py.special_dtype(vlen=str))
                else:
                    subgroup.create_dataset('categories',
                                            data=categories.to_numpy())
            elif dtype.is_float():
                # Nullable floats are not supported, so convert null to NaN
                dataset = group.create_dataset(
                    column.name, data=column.fill_null(np.nan).to_numpy())
                dataset.attrs['encoding-type'] = 'array'
                dataset.attrs['encoding-version'] = '0.2.0'
            elif dtype == pl.Boolean or dtype.is_integer():
                is_boolean = dtype == pl.Boolean
                if column.null_count():
                    # Store as nullable integer/Boolean
                    subgroup = group.create_group(column.name)
                    subgroup.attrs['encoding-type'] = \
                        f'nullable-{"boolean" if is_boolean else "integer"}'
                    subgroup.attrs['encoding-version'] = '0.1.0'
                    subgroup.create_dataset(
                        'values',
                        data=column.fill_null(False if is_boolean else 1)
                        .to_numpy())
                    subgroup.create_dataset(
                        'mask', data=column.is_null().to_numpy())
                else:
                    # Store as regular integer/Boolean
                    dataset = group.create_dataset(column.name,
                                                   data=column.to_numpy())
                    dataset.attrs['encoding-type'] = 'array'
                    dataset.attrs['encoding-version'] = '0.2.0'
            else:
                error_message = \
                    f'internal error: unsupported data type {dtype!r}'
                raise TypeError(error_message)
    
    def save(self, filename: str | Path, *, overwrite: bool = False,
             preserve_strings: bool = False) -> None:
        """
        Save this SingleCell dataset to an AnnData .h5ad file, Seurat .rds
        file, or 10x .mtx.gz file.
        
        Args:
            filename: an AnnData .h5ad file, Seurat .rds file, or 10x .mtx.gz
                      file to save to. File format will be inferred from the
                      extension.
                      - When saving to Seurat, to match the requirements of
                        Seurat objects, the `'X_'` prefix (often used by
                        Scanpy) will be removed from each key of obsm where it
                        is present (e.g. `'X_umap'` will become `'umap'`).
                        Seurat will also add `'orig.ident'`, `'nCount_RNA'` and
                        `'nFeature_RNA'` as gene-level metadata.
                      - When saving to 10x, barcodes.tsv.gz and features.tsv.gz
                        will be created in the same directory.
            overwrite: if False, raises an error if (any of) the file(s) exist;
                       if True, overwrites them
            preserve_strings: if False, encode string columns with duplicate
                              values as Enums to save space, when saving to
                              AnnData .h5ad or Seurat .rds; if True, preserve
                              these columns as string columns. (Regardless of
                              the value of `preserve_strings`, String columns
                              with `null` values will be encoded as Enums when
                              saving to .h5ad, since the .h5ad format cannot
                              represent them otherwise.)
        """
        # Check inputs
        check_type(filename, 'filename', str,
                   "a string ending in '.h5ad', '.rds', or '.mtx.gz'")
        anndata = filename.endswith('.h5ad')
        seurat = filename.endswith('.rds')
        if not (anndata or seurat or filename.endswith('.mtx.gz')):
            error_message = (
                f"filename {filename!r} does not end with '.h5ad', '.rds', or "
                f"'.mtx.gz'")
            raise ValueError(error_message)
        if not overwrite and os.path.exists(filename):
            error_message = (
                f'filename {filename!r} already exists; set overwrite=True '
                f'to overwrite')
            raise FileExistsError(error_message)
        # Raise an error if obs or var contain columns with unsupported data
        # types (anything but float, int, String, Categorical, Enum, Boolean)
        valid_polars_dtypes = pl.FLOAT_DTYPES | pl.INTEGER_DTYPES | \
                              {pl.String, pl.Categorical, pl.Enum, pl.Boolean}
        for df, df_name in (self._obs, 'obs'), (self._var, 'var'):
            for column, dtype in df.schema.items():
                if dtype.base_type() not in valid_polars_dtypes:
                    error_message = (
                        f'{df_name}[{column!r}] has the data type '
                        f'{dtype.base_type()!r}, which is not supported when '
                        f'saving')
                    raise TypeError(error_message)
        # Raise an error if obsm, varm or uns contain keys with unsupported
        # data types (datetime64, timedelta64, unstructured void).
        # Do not specifically check for `dtype=object` to avoid extra overhead.
        for field, field_name in \
                (self._obsm, 'obsm'), (self._varm, 'varm'), (self._uns, 'uns'):
            for key, value in field.items():
                if field is self._uns and not isinstance(value, np.ndarray):
                    continue
                if value.dtype.type == np.void and value.dtype.names is None:
                    error_message = (
                        f'{field_name}[{key!r}] is an unstructured void '
                        f'array, which is not supported when saving')
                    raise TypeError(error_message)
                elif value.dtype == np.datetime64:
                    error_message = (
                        f'{field_name}[{key!r}] is a datetime64 array, which '
                        f'is not supported when saving')
                    raise TypeError(error_message)
                elif value.dtype == np.timedelta64:
                    error_message = (
                        f'{field_name}[{key!r}] is a timedelta64 array, which '
                        f'is not supported when saving')
                    raise TypeError(error_message)
        # Save, depending on the file extension
        if anndata:
            try:
                with h5py.File(filename, 'w') as h5ad_file:
                    # Add top-level metadata
                    h5ad_file.attrs['encoding-type'] = 'anndata'
                    h5ad_file.attrs['encoding-version'] = '0.1.0'
                    # Save obs and var
                    self._write_h5ad_dataframe(h5ad_file, self._obs, 'obs',
                                               preserve_strings)
                    self._write_h5ad_dataframe(h5ad_file, self._var, 'var',
                                               preserve_strings)
                    # Save obsm
                    if self._obsm:
                        obsm = h5ad_file.create_group('obsm')
                        obsm.attrs['encoding-type'] = 'dict'
                        obsm.attrs['encoding-version'] = '0.1.0'
                        for key, value in self._obsm.items():
                            obsm.create_dataset(key, data=value)
                    # Save varm
                    if self._varm:
                        varm = h5ad_file.create_group('varm')
                        varm.attrs['encoding-type'] = 'dict'
                        varm.attrs['encoding-version'] = '0.1.0'
                        for key, value in self._varm.items():
                            varm.create_dataset(key, data=value)
                    # Save uns
                    if self._uns:
                        SingleCell._save_uns(self._uns,
                                             h5ad_file.create_group('uns'),
                                             h5ad_file)
                    # Save X
                    X = h5ad_file.create_group('X')
                    X.attrs['encoding-type'] = 'csr_matrix' \
                        if isinstance(self._X, csr_array) else 'csc_matrix'
                    X.attrs['encoding-version'] = '0.1.0'
                    X.attrs['shape'] = self._X.shape
                    X.create_dataset('data', data=self._X.data)
                    X.create_dataset('indices', data=self._X.indices)
                    X.create_dataset('indptr', data=self._X.indptr)
            except:
                if os.path.exists(filename):
                    os.unlink(filename)
                raise
        elif seurat:
            from ryp import r
            if preserve_strings:
                sc = self
            else:
                enumify = lambda df: df.cast({
                    row[0]: pl.Enum(row[1]) for row in df
                    .select(pl.selectors.string()
                            .unique(maintain_order=True)
                            .implode()
                            .list.drop_nulls())
                    .unpivot()
                    .filter(pl.col.value.list.len() == len(df))
                    .rows()})
                sc = SingleCell(X=self._X, obs=enumify(self._obs),
                                var=enumify(self._var), obsm=self._obsm,
                                uns=self._uns)
            sc.to_seurat('.SingleCell.seurat_object')
            try:
                r(f'saveRDS(.SingleCell.seurat_object, {filename!r})')
            except:
                if os.path.exists(filename):
                    os.unlink(filename)
                raise
            finally:
                r('rm(.SingleCell.seurat_object)')
        else:
            from scipy.io import mmwrite
            barcode_filename = \
                os.path.join(os.path.dirname(filename), 'barcodes.tsv.gz')
            feature_filename = \
                os.path.join(os.path.dirname(filename), 'features.tsv.gz')
            if not overwrite:
                for ancillary_filename in barcode_filename, feature_filename:
                    if os.path.exists(ancillary_filename):
                        error_message = (
                            f'{ancillary_filename!r} already exists; set '
                            f'overwrite=True to overwrite')
                        raise FileExistsError(error_message)
            try:
                mmwrite(filename, self._X.T)
                self._obs.write_csv(barcode_filename, include_header=False)
                self._var.write_csv(feature_filename, include_header=False)
            except:
                if os.path.exists(filename):
                    os.unlink(filename)
                if os.path.exists(barcode_filename):
                    os.unlink(barcode_filename)
                if os.path.exists(feature_filename):
                    os.unlink(feature_filename)
                raise
    
    def _get_column(self,
                    obs_or_var_name: Literal['obs', 'var'],      
                    column: SingleCellColumn,
                    variable_name: str,
                    dtypes: pl.datatypes.classes.DataTypeClass | str |
                            tuple[pl.datatypes.classes.DataTypeClass | str,
                            ...],
                    *,
                    QC_column: pl.Series | None = None,
                    allow_missing: bool = False,
                    allow_null: bool = False,
                    custom_error: str | None = None) -> pl.Series | None:
        """
        Get a column of the same length as obs/var, or None if the column is
        missing from obs/var and `allow_missing=True`.
        
        Args:
            obs_or_var_name: the name of the DataFrame the column is with
                             respect to, i.e. `'obs'` or `'var'`
            column: a string naming a column of obs/var, a polars expression
                    that evaluates to a single column when applied to obs/var,
                    a polars Series or 1D NumPy array of the same length as
                    obs/var, or a function that takes in `self` and returns a
                    polars Series or 1D NumPy array of the same length as
                    obs/var
            variable_name: the name of the variable corresponding to `column`
            dtypes: the required dtype(s) of the column
            QC_column: an optional column of cells passing QC. If specified,
                       the presence of null values will only raise an error for
                       cells passing QC. Has no effect when `allow_null=True`.
            allow_missing: whether to allow `column` to be a string missing
                           from obs/var, returning None in this case
            allow_null: whether to allow `column` to contain null values
            custom_error: a custom error message for when `column` is a string
                          and is not found in obs/var, and
                          `allow_missing=False`; use `{}` as a placeholder for
                          the name of the column
        
        Returns:
            A polars Series of the same length as obs/var, or None if the
            column is missing from obs/var and `allow_missing=True`.
        """
        obs_or_var = self._obs if obs_or_var_name == 'obs' else self._var
        if isinstance(column, str):
            if column in obs_or_var:
                column = obs_or_var[column]
            elif allow_missing:
                return None
            else:
                error_message = \
                    f'{column!r} is not a column of {obs_or_var_name}' \
                    if custom_error is None else \
                    custom_error.format(f'{column!r}')
                raise ValueError(error_message)
        elif isinstance(column, pl.Expr):
            column = obs_or_var.select(column)
            if column.width > 1:
                error_message = (
                    f'{variable_name} is a polars expression that expands to '
                    f'{column.width:,} columns rather than 1')
                raise ValueError(error_message)
            column = column.to_series()
        elif isinstance(column, pl.Series):
            if len(column) != len(obs_or_var):
                error_message = (
                    f'{variable_name} is a polars Series of length '
                    f'{len(column):,}, which differs from the length of '
                    f'{obs_or_var_name} ({len(obs_or_var):,})')
                raise ValueError(error_message)
        elif isinstance(column, np.ndarray):
            if len(column) != len(obs_or_var):
                error_message = (
                    f'{variable_name} is a NumPy array of length '
                    f'{len(column):,}, which differs from the length of '
                    f'{obs_or_var_name} ({len(obs_or_var):,})')
                raise ValueError(error_message)
            column = pl.Series(variable_name, column)
        elif callable(column):
            column = column(self)
            if isinstance(column, np.ndarray):
                if column.ndim != 1:
                    error_message = (
                        f'{variable_name} is a function that returns a '
                        f'{column.ndim:,}D NumPy array, but must return a '
                        f'polars Series or 1D NumPy array')
                    raise ValueError(error_message)
                column = pl.Series(variable_name, column)
            elif not isinstance(column, pl.Series):
                error_message = (
                    f'{variable_name} is a function that returns a variable '
                    f'of type {type(column).__name__}, but must return a '
                    f'polars Series or 1D NumPy array')
                raise TypeError(error_message)
            if len(column) != len(obs_or_var):
                error_message = (
                    f'{variable_name} is a function that returns a column of '
                    f'length {len(column):,}, which differs from the length '
                    f'of {obs_or_var_name} ({len(obs_or_var):,})')
                raise ValueError(error_message)
        else:
            error_message = (
                f'{variable_name} must be a string column name, a polars '
                f'expression, a polars Series, a 1D NumPy array, or a '
                f'function that returns a polars Series or 1D NumPy array '
                f'when applied to this SingleCell dataset, but has type '
                f'{type(column).__name__!r}')
            raise TypeError(error_message)
        check_dtype(column, variable_name, dtypes)
        if not allow_null:
            if QC_column is None:
                null_count = column.null_count()
                if null_count > 0:
                    error_message = (
                        f'{variable_name} contains {null_count:,} '
                        f'{plural("null value", null_count)}, but must not '
                        f'contain any')
                    raise ValueError(error_message)
            else:
                null_count = (column.is_null() & QC_column).sum()
                if null_count > 0:
                    error_message = (
                        f'{variable_name} contains {null_count:,} '
                        f'{plural("null value", null_count)} for cells '
                        f'passing QC, but must not contain any')
                    raise ValueError(error_message)
        return column
    
    @staticmethod
    def _get_columns(obs_or_var_name: Literal['obs', 'var'],
                     datasets: Sequence[SingleCell],
                     columns: SingleCellColumn | None |
                              Sequence[SingleCellColumn | None],
                     variable_name: str,
                     dtypes: pl.datatypes.classes.DataTypeClass | str |
                             tuple[pl.datatypes.classes.DataTypeClass | str,
                                   ...],
                     *,
                     QC_column: pl.Series | None = None,
                     allow_None: bool = True,
                     allow_missing: bool = False,
                     allow_null: bool = False,
                     custom_error: str | None = None) -> \
            list[pl.Series | None]:
        """
        Get a column of the same length as obs/var from each dataset.
        
        Args:
            obs_or_var_name: the name of the DataFrame the column is with
                             respect to, i.e. `'obs'` or `'var'`
            datasets: a sequence of SingleCell datasets
            columns: a string naming a column of obs/var, a polars expression
                     that evaluates to a single column when applied to obs/var,
                     a polars Series or 1D NumPy array of the same length as 
                     obs/var, or a function that takes in `self` and returns a
                     polars Series or 1D NumPy array of the same length as 
                     obs/var. Or, a Sequence of these, one per dataset in
                     `datasets`. May also be None (or a Sequence containing
                     None) if `allow_None=True`.
            variable_name: the name of the variable corresponding to `columns`
            dtypes: the required dtype(s) of the columns
            QC_column: an optional column of cells passing QC. If specified,
                       the presence of null values will only raise an error for
                       cells passing QC. Has no effect when `allow_null=True`.
            allow_None: whether to allow `columns` or its elements to be None
            allow_missing: whether to allow `columns` to be a string (or
                           contain strings) missing from certain datasets' 
                           obs/var, returning None for these datasets
            allow_null: whether to allow `columns` to contain null values
            custom_error: a custom error message for when `column` is a string
                          and is not found in obs/var, and 
                          `allow_missing=False`; use `{}` as a placeholder for
                          the name of the column
        
        Returns:
            A list of polars Series of the same length as `datasets`, where
            each Series has the same length as the corresponding dataset's 
            obs/var. Or, if `columns` is None (or if some elements are None) or
            missing from obs/var (when `allow_missing=True`), a list of None
            (or where the corresponding elements are None).
        """
        if columns is None:
            if not allow_None:
                error_message = f'{variable_name} is None'
                raise TypeError(error_message)
            return [None for _ in datasets]
        elif isinstance(columns, Sequence) and not isinstance(columns, str):
            if len(columns) != len(datasets):
                error_message = (
                    f'{variable_name} has length {len(columns):,}, but you '
                    f'specified {len(datasets):,} datasets')
                raise ValueError(error_message)
            if not allow_None and any(column is None for column in columns):
                error_message = \
                    f'{variable_name} contains an element that is None'
                raise TypeError(error_message)
            return [dataset._get_column(
                obs_or_var_name=obs_or_var_name, column=column,
                variable_name=variable_name, dtypes=dtypes,
                QC_column=QC_column, allow_null=allow_null,
                allow_missing=allow_missing, custom_error=custom_error)
                if column is not None else None
                for dataset, column in zip(datasets, columns)]
        else:
            return [dataset._get_column(
                obs_or_var_name=obs_or_var_name, column=columns,
                variable_name=variable_name, dtypes=dtypes,
                QC_column=QC_column, allow_null=allow_null,
                allow_missing=allow_missing, custom_error=custom_error)
                for dataset in datasets]
    
    # noinspection PyUnresolvedReferences
    def to_anndata(self, *, QC_column: str | None = 'passed_QC') -> 'AnnData':
        """
        Converts this SingleCell dataset to an AnnData object.
        
        Make sure to remove cells failing QC with `filter_obs(QC_column)`
        first, or specify `subset=True` in `qc()`. Alternatively, to include
        cells failing QC in the AnnData object, set `QC_column` to None.
        
        Note that there is no `from_anndata()`; simply do
        `SingleCell(anndata_object)` to initialize a SingleCell dataset from an
        in-memory AnnData object.
        
        Args:
            QC_column: if not None, give an error if this column is present in
                       obs and not all cells pass QC
        
        Returns:
            An AnnData object. For AnnData versions older than 0.11.0, which
            do not support csr_array/csc_array, counts will be converted to
            csr_matrix/csc_matrix.
        """
        # Make anndata and pandas imports non-interruptible, to avoid bugs due
        # to partial imports; throw in pyarrow too, though not sure if it's
        # also vulnerable
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            import anndata
            import pandas as pd
            import pyarrow as pa
        finally:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        valid_dtypes = pl.FLOAT_DTYPES | pl.INTEGER_DTYPES | \
                       {pl.String, pl.Categorical, pl.Enum, pl.Boolean}
        for df, df_name in (self._obs, 'obs'), (self._var, 'var'):
            for column, dtype in df.schema.items():
                if dtype.base_type() not in valid_dtypes:
                    error_message = (
                        f'{df_name}[{column!r}] has the data type '
                        f'{dtype.base_type()!r}, which is not supported by '
                        f'AnnData')
                    raise TypeError(error_message)
        if QC_column is not None:
            check_type(QC_column, 'QC_column', str, 'a string')
            if QC_column in self._obs:
                QCed_cells = self._obs[QC_column]
                check_dtype(QCed_cells, f'obs[{QC_column!r}]',
                            pl.Boolean)
                if QCed_cells.null_count() or not QCed_cells.all():
                    error_message = (
                        f'not all cells pass QC; remove cells failing QC with '
                        f'filter_obs({QC_column!r}) or by specifying '
                        f'subset=True in qc(), or set QC_column=None to '
                        f'include them in the AnnData object')
                    raise ValueError(error_message)
        type_mapping = {
            pa.int8(): pd.Int8Dtype(), pa.int16(): pd.Int16Dtype(),
            pa.int32(): pd.Int32Dtype(), pa.int64(): pd.Int64Dtype(),
            pa.uint8(): pd.UInt8Dtype(), pa.uint16(): pd.UInt16Dtype(),
            pa.uint32(): pd.UInt32Dtype(), pa.uint64(): pd.UInt64Dtype(),
            pa.string(): pd.StringDtype(storage='pyarrow'),
            pa.bool_(): pd.BooleanDtype()}
        to_pandas = lambda df: df\
            .to_pandas(split_blocks=True, types_mapper=type_mapping.get)\
            .set_index(df.columns[0])
        return anndata.AnnData(
            X=self._X if version.parse(anndata.__version__) >=
                         version.parse('0.11.0') else
              csr_matrix(self._X) if isinstance(self._X, csr_array)
              else csc_matrix(self._X),
            obs=to_pandas(self._obs), var=to_pandas(self._var),
            obsm=self._obsm, varm=self._varm, uns=self._uns)
    
    @staticmethod
    def _from_seurat(seurat_object_name: str, *, assay: str | None,
                     slot: str) -> \
            tuple[csr_array | csc_array, pl.DataFrame, pl.DataFrame,
                  dict[str, np.ndarray[2, Any]], NestedScalarOrArrayDict]:
        """
        Create a SingleCell dataset from an in-memory Seurat object loaded with
        the ryp Python-R bridge. Used by __init__() and from_seurat().
        
        Args:
            seurat_object_name: the name of the Seurat object in the ryp R
                                workspace
            assay: the name of the assay within the Seurat object to load data
                   from; if None, defaults to `seurat_object@active.assay`
            slot: the slot within the active assay (or the assay specified by
                  the `assay` argument, if not None) to use as X. Set to
                  `'data'` to load the normalized counts, or `'scale.data'` to
                  load the normalized and scaled counts. If dense, will be
                  automatically converted to a sparse array.
        
        Returns:
            A length-5 tuple of (X, obs, var, obsm, uns).
        """
        from ryp import r, to_py
        if assay is None:
            assay = to_py(f'{seurat_object_name}@active.assay')
        elif to_py(f'{seurat_object_name}@{assay}') is None:
            error_message = (
                f'assay {assay!r} does not exist in the Seurat object '
                f'{seurat_object_name!r}')
            raise ValueError(error_message)
        # If Seurat v5, merge layers if necessary, and use $slot
        # instead of @slot for X and meta.data instead of
        # meta.features for var
        if to_py(f'inherits({seurat_object_name}@assays${assay}, "Assay5")'):
            if not to_py(f'"{slot}" %in% names({seurat_object_name}@assays$'
                         f'{assay}@layers)'):
                error_message = (
                    f'slot {slot!r} does not exist in '
                    f'{seurat_object_name}@assays${assay}@layers')
                raise ValueError(error_message)
            if to_py(f'length({seurat_object_name}@assays${assay}@'
                     f'layers)') > 1:
                r(f'{seurat_object_name}@assays${assay} = '
                  f'JoinLayers({seurat_object_name}@assays${assay}, "{slot}")')
            X_slot = f'{seurat_object_name}@assays${assay}${slot}'
            var = to_py(f'{seurat_object_name}@assays${assay}@meta.data')
        else:
            # unlike v5 objects, v3 objects indicate the absence of a slot with
            # a 0 x 0 matrix
            if not (to_py(f'"{slot}" %in% slotNames({seurat_object_name}@'
                         f'assays${assay})') and
                    to_py(f'prod(dim({seurat_object_name}@assays${assay}@'
                          f'{slot}))') > 0):
                error_message = (
                    f'slot {slot!r} does not exist in '
                    f'{seurat_object_name}@assays${assay}')
                raise ValueError(error_message)
            X_slot = f'{seurat_object_name}@assays${assay}@{slot}'
            var = to_py(f'{seurat_object_name}@assays${assay}@meta.features')
        X_classes = tuple(to_py(f'class({X_slot})', squeeze=False))
        if X_classes == ('dgCMatrix',):
            X = to_py(X_slot).T
        elif X_classes == ('matrix', 'array'):
            X = csr_array(to_py(X_slot, format='numpy').T)
        else:
            error_message = f'{X_slot} must be a dgCMatrix or matrix, but has '
            if len(X_classes) == 0:
                error_message += 'no classes'
            elif len(X_classes) == 1:
                error_message += f'the class {X_classes[0]!r}'
            else:
                error_message += (
                    f'the classes '
                    f'{", ".join(f"{c!r}" for c in X_classes[:-1])} and '
                    f'{X_classes[-1]}')
            error_message += \
                f'; consider setting slot to a different value than {slot!r}'
            raise TypeError(error_message)
        obs = to_py(f'{seurat_object_name}@meta.data', index='cell')
        if var is None:
            var = to_py(f'rownames({seurat_object_name}@assays${assay}@'
                        f'{slot}').to_frame('gene')
        reductions = to_py(f'names({seurat_object_name}@reductions)')
        obsm = {key: to_py(
            f'{seurat_object_name}@reductions${key}@cell.embeddings',
            format='numpy') for key in reductions
            if not to_py(f'is.null({seurat_object_name}@reductions${key})')} \
            if reductions is not None else {}
        obs = obs.cast({column.name: pl.Enum(column.cat.get_categories())
                        for column in obs.select(pl.col(pl.Categorical))})
        var = var.cast({column.name: pl.Enum(column.cat.get_categories())
                        for column in var.select(pl.col(pl.Categorical))})
        uns = to_py(f'{seurat_object_name}@misc')
        return X, obs, var, obsm, uns
    
    @staticmethod
    def from_seurat(seurat_object_name: str, *, assay: str | None = None,
                    slot: str = 'counts') -> SingleCell:
        """
        Create a SingleCell dataset from a Seurat object that was already
        loaded into memory via the ryp Python-R bridge. To load a Seurat object
        from disk, use e.g. `SingleCell('filename.rds')`.
        
        Args:
            seurat_object_name: the name of the Seurat object in the ryp R
                                workspace
            assay: the name of the assay within the Seurat object to load data
                   from; if None, defaults to seurat_object@active.assay
            slot: the slot within the active assay (or the assay specified by
                  the `assay` argument, if not None) to use as X; defaults to
                  `'counts'`. Set to `'data'` to load the normalized counts,
                  or `'scale.data'` to load the normalized and scaled counts.
                  If dense, will be automatically converted to a sparse array.
        
        Returns:
            The corresponding SingleCell dataset.
        """
        from ryp import to_py
        check_type(seurat_object_name, 'seurat_object_name', str, 'a string')
        if assay is not None:
            check_type(assay, 'assay', str, 'a string')
        check_type(slot, 'slot', str, 'a string')
        if not to_py(f'inherits({seurat_object_name}, "Seurat")'):
            classes = to_py(f'class({seurat_object_name})', squeeze=False)
            if len(classes) == 0:
                error_message = (
                    f'the R object named by seurat_object_name, '
                    f'{seurat_object_name}, must be a Seurat object, but has '
                    f'no classes')
            elif len(classes) == 1:
                error_message = (
                    f'the R object named by seurat_object_name, '
                    f'{seurat_object_name}, must be a Seurat object, but has '
                    f'the class {classes[0]!r}')
            else:
                error_message = (
                    f'the R object named by seurat_object_name, '
                    f'{seurat_object_name}, must be a Seurat object, but has '
                    f'the classes '
                    f'{", ".join(f"{c!r}" for c in classes[:-1])} and '
                    f'{classes[-1]!r}')
            raise TypeError(error_message)
        X, obs, var, obsm, uns = \
            SingleCell._from_seurat(seurat_object_name, assay=assay, slot=slot)
        return SingleCell(X=X, obs=obs, var=var, obsm=obsm, uns=uns)
    
    def to_seurat(self,
                  seurat_object_name: str,
                  *,
                  QC_column: str | None = 'passed_QC') -> None:
        """
        Convert this SingleCell dataset to a Seurat object (version 3, not
        version 5) in the ryp R workspace.
        
        Make sure to remove cells failing QC with `filter_obs(QC_column)`
        first, or specify `subset=True` in `qc()`. Alternatively, to include
        cells failing QC in the Seurat object, set `QC_column` to None.
        
        When converting to Seurat, to match the requirements of Seurat objects,
        the `'X_'` prefix (often used by Scanpy) will be removed from each key
        of obsm where it is present (e.g. `'X_umap'` will become `'umap'`).
        Seurat will also add `'orig.ident'`, `'nCount_RNA'` and
        `'nFeature_RNA'` as gene-level metadata. Only string keys of uns will
        be prserved.
        
        Args:
            seurat_object_name: the name of the R variable to assign the Seurat
                                object to
            QC_column: if not None, give an error if this column is present in
                       obs and not all cells pass QC
        """
        valid_dtypes = pl.FLOAT_DTYPES | pl.INTEGER_DTYPES | \
                       {pl.String, pl.Categorical, pl.Enum, pl.Boolean}
        for df, df_name in (self._obs, 'obs'), (self._var, 'var'):
            for column, dtype in df.schema.items():
                if dtype.base_type() not in valid_dtypes:
                    error_message = (
                        f'{df_name}[{column!r}] has the data type '
                        f'{dtype.base_type()!r}, which is not supported when '
                        f'converting to a Seurat object')
                    raise TypeError(error_message)
        if QC_column is not None:
            check_type(QC_column, 'QC_column', str, 'a string')
            if QC_column in self._obs:
                QCed_cells = self._obs[QC_column]
                check_dtype(QCed_cells, f'obs[{QC_column!r}]',
                            pl.Boolean)
                if QCed_cells.null_count() or not QCed_cells.all():
                    error_message = (
                        f'not all cells pass QC; remove cells failing QC with '
                        f'filter_obs({QC_column!r}) or by specifying '
                        f'subset=True in qc(), or set QC_column=None to '
                        f'include them in the Seurat object')
                    raise ValueError(error_message)
        from ryp import r, to_r
        r('suppressPackageStartupMessages(library(SeuratObject))')
        is_string = self.var_names.dtype == pl.String
        num_with_underscores = self.var_names.str.contains('_').sum() \
            if is_string else \
            self.var_names.cat.get_categories().str.contains('_').sum()
        if num_with_underscores:
            var_names_expression = f'pl.col.{self.var_names.name}' \
                if self.var_names.name.isidentifier() else \
                f'pl.col({self.var_names.name!r})'
            error_message = (
                f"var_names contains {num_with_underscores:,}"
                f"{'' if is_string else ' unique'} gene "
                f"{plural('name', num_with_underscores)} with "
                f"underscores, which are not supported by Seurat; Seurat "
                f"recommends changing the underscores to dashes, which you "
                f"can do with .with_columns_var({var_names_expression}"
                f"{'' if is_string else '.cast(pl.String)'}"
                f".str.replace_all('_', '-'))")
            raise ValueError(error_message)
        to_r(self._X.T, '.SingleCell.X.T', rownames=self.var_names,
             colnames=self.obs_names)
        try:
            to_r(self._obs.drop(self.obs_names.name), '.SingleCell.obs',
                 rownames=self.obs_names)
            try:
                r(f'{seurat_object_name} = CreateSeuratObject('
                  'CreateAssayObject(.SingleCell.X.T), '
                  'meta.data=.SingleCell.obs)')
            finally:
                r('rm(".SingleCell.obs")')
        finally:
            r('rm(".SingleCell.X.T")')
        to_r(self._var.drop(self.var_names.name), '.SingleCell.var',
             rownames=self.var_names)
        try:
            r(f'{seurat_object_name}@assays$RNA@meta.features = '
              f'.SingleCell.var')
        finally:
            r('rm(".SingleCell.var")')
        if self._obsm:
            for key, value in self._obsm.items():
                # Remove the initial X_ from the reduction name and suffix with
                # _ when creating the key, like
                # mojaveazure.github.io/seurat-disk/reference/Convert.html
                key = key.removeprefix('X_')
                to_r(value, '.SingleCell.value', rownames=self.obs_names,
                     colnames=[f'{key}_{i}'
                               for i in range(1, value.shape[1] + 1)])
                try:
                    r(f'{seurat_object_name}@reductions${key} = '
                      f'CreateDimReducObject(.SingleCell.value, '
                      f'key="{key}_", assay="RNA")')
                finally:
                    r('rm(".SingleCell.value")')
        if self._uns:
            to_r({key: value for key, value in self._uns.items()
                  if isinstance(value, str)}, '.SingleCell.uns')
            try:
                r(f'{seurat_object_name}@misc = .SingleCell.uns')
            finally:
                r('rm(".SingleCell.uns")')

    def copy(self, deep: bool = False) -> SingleCell:
        """
        Make a deep (if deep=True) or shallow copy of this SingleCell dataset.
        
        Returns:
            A copy of the SingleCell dataset. Since polars DataFrames are
            immutable, obs and var will always point to the same underlying
            data as the original. The only difference when deep=True is that X
            will point to a fresh copy of the data, rather than the same data.
            Watch out: when deep=False, any modifications to X will modify both
            copies!
        """
        check_type(deep, 'deep', bool, 'Boolean')
        # noinspection PyTypeChecker
        return SingleCell(X=self._X.copy() if deep else self._X, obs=self._obs,
                          var=self._var, obsm=self._obsm)
    
    def concat_obs(self,
                   datasets: SingleCell,
                   *more_datasets: SingleCell,
                   flexible: bool = False) -> SingleCell:
        """
        Concatenate the cells of multiple SingleCell datasets.
        
        By default, all datasets must have the same var, varm and uns. They
        must also have the same columns in obs and the same keys in obsm, with
        the same data types.
        
        Conversely, if `flexible=True`, subset to genes present in all datasets
        (according to the first column of var, i.e. `var_names`) before
        concatenating. Subset to columns of var and keys of varm and uns that
        are identical in all datasets after this subsetting. Also, subset to
        columns of obs and keys of obsm that are present in all datasets, and
        have the same data types. All datasets' `obs_names` must have the same
        name and dtype, and similarly for `var_names`.
        
        The one exception to the obs "same data type" rule: if a column is Enum
        in some datasets and Categorical in others, or Enum in all datasets but
        with different categories in each dataset, that column will be retained
        as an Enum column (with the union of the categories) in the
        concatenated obs.
        
        Args:
            datasets: one or more SingleCell datasets to concatenate with this
                      one
            *more_datasets: additional SingleCell datasets to concatenate with
                            this one, specified as positional arguments
            flexible: whether to subset to genes, columns of obs and var, and
                      keys of obsm, varm and uns common to all datasets before
                      concatenating, rather than raising an error on any
                      mismatches
        
        Returns:
            The concatenated SingleCell dataset.
        """
        # Check inputs
        if isinstance(datasets, Pseudobulk):
            datasets = datasets,
        datasets = (self,) + datasets + more_datasets
        if len(datasets) == 1:
            error_message = \
                'need at least one other SingleCell dataset to concatenate'
            raise ValueError(error_message)
        check_types(datasets[1:], 'datasets', SingleCell,
                    'SingleCell datasets')
        check_type(flexible, 'flexible', bool, 'Boolean')
        # Perform either flexible or non-flexible concatenation
        if flexible:
            # Check that `obs_names` and `var_names` have the same name and
            # data type across all datasets
            obs_names_name = self.obs_names.name
            if not all(dataset.obs_names.name == obs_names_name
                       for dataset in datasets[1:]):
                error_message = (
                    'not all SingleCell datasets have the same name for the '
                    'first column of obs (the obs_names column)')
                raise ValueError(error_message)
            var_names_name = self.var_names.name
            if not all(dataset.var_names.name == var_names_name
                       for dataset in datasets[1:]):
                error_message = (
                    'not all SingleCell datasets have the same name for the '
                    'first column of var (the var_names column)')
                raise ValueError(error_message)
            obs_names_dtype = self.obs_names.dtype
            if not all(dataset.obs_names.dtype == obs_names_dtype
                       for dataset in datasets[1:]):
                error_message = (
                    'not all SingleCell datasets have the same data type for '
                    'the first column of obs (the obs_names column)')
                raise TypeError(error_message)
            var_names_dtype = self.var_names.dtype
            if not all(dataset.var_names.dtype == var_names_dtype
                       for dataset in datasets[1:]):
                error_message = (
                    'not all SingleCell datasets have the same data type for '
                    'the first column of var (the var_names column)')
                raise TypeError(error_message)
            # Subset to genes in common across all datasets
            genes_in_common = self.var_names\
                .filter(self.var_names
                        .is_in(pl.concat([dataset.var_names
                                          for dataset in datasets[1:]])))
            if len(genes_in_common) == 0:
                error_message = \
                    'no genes are shared across all SingleCell datasets'
                raise ValueError(error_message)
            datasets = [dataset[:, genes_in_common] for dataset in datasets]
            # Subset to columns of var and keys of varm and uns that are
            # identical in all datasets after this subsetting
            var_columns_in_common = [
                column.name for column in datasets[0]._var[:, 1:]
                if all(column.name in dataset._var and
                       dataset._var[column.name].equals(column)
                       for dataset in datasets[1:])]
            varm = self._varm
            varm_keys_in_common = [
                key for key in varm
                if all(key in dataset._varm and
                       dataset._varm[key].dtype == varm[key].dtype and
                       (dataset._varm[key] == varm[key]).all()
                       for dataset in datasets[1:])]
            # noinspection PyTypeChecker,PyUnresolvedReferences
            uns_keys_in_common = [
                key for key, value in self._uns.items()
                if isinstance(value, dict) and
                   all(isinstance(dataset._uns[key], dict) and
                       SingleCell._eq_uns(value, dataset._uns[key])
                       for dataset in datasets[1:]) or
                   isinstance(value, np.ndarray) and
                   all(isinstance(dataset._uns[key], np.ndarray) and
                       (value == dataset._uns[key]).all()
                       for dataset in datasets[1:]) or
                   all(not isinstance(dataset._uns[key], (dict, np.ndarray))
                       and value == dataset._uns[key]
                       for dataset in datasets[1:])]
            for dataset in datasets:
                dataset._var = dataset._var.select(dataset.var_names,
                                                   *var_columns_in_common)
                dataset._varm = {key: dataset._varm[key]
                                  for key in varm_keys_in_common}
                dataset._uns = {key: dataset._uns[key]
                                for key in uns_keys_in_common}
            # Subset to columns of obs and keys of obsm that are present in all
            # datasets, and have the same data types. Also include columns of
            # obs that are Enum in some datasets and Categorical in others, or
            # Enum in all datasets but with different categories in each
            # dataset; cast these to Categorical.
            obs_mismatched_categoricals = {
                column for column, dtype in self._obs[:, 1:]
                .select(pl.col(pl.Categorical, pl.Enum)).schema.items()
                if all(column in dataset._obs and
                       dataset._obs[column].dtype in (pl.Categorical, pl.Enum)
                       for dataset in datasets[1:]) and
                   not all(dataset._obs[column].dtype == dtype
                           for dataset in datasets[1:])}
            obs_columns_in_common = [
                column for column, dtype in islice(self._obs.schema.items(), 1)
                if column in obs_mismatched_categoricals or
                   all(column in dataset._obs and
                       dataset._obs[column].dtype == dtype
                       for dataset in datasets[1:])]
            cast_dict = {column: pl.Enum(
                pl.concat([dataset._obs[column].cat.get_categories()
                           for dataset in datasets])
                .unique(maintain_order=True))
                for column in obs_mismatched_categoricals}
            for dataset in datasets:
                dataset._obs = dataset._obs\
                    .select(dataset.obs_names, *obs_columns_in_common)\
                    .cast(cast_dict)
            obsm_keys_in_common = [
                key for key in self._obsm
                if all(key in dataset._obsm and
                       dataset._obsm[key].dtype == self._obsm[key].dtype
                       for dataset in datasets[1:])]
            for dataset in datasets:
                dataset._obsm = {key: dataset._obsm[key]
                                 for key in obsm_keys_in_common}
        else:  # non-flexible
            # Check that all var, varm and uns are identical
            var = self._var
            for dataset in datasets[1:]:
                if not dataset._var.equals(var):
                    error_message = (
                        'all SingleCell datasets must have the same var, '
                        'unless flexible=True')
                    raise ValueError(error_message)
            varm = self._varm
            for dataset in datasets[1:]:
                if dataset._varm.keys() != varm.keys() or \
                        any(dataset._varm[key].dtype != varm[key].dtype
                            for key in varm) or \
                        any((dataset._varm[key] != varm[key]).any()
                            for key in varm):
                    error_message = (
                        'all SingleCell datasets must have the same varm, '
                        'unless flexible=True')
                    raise ValueError(error_message)
            for dataset in datasets[1:]:
                if not SingleCell._eq_uns(self._uns, dataset._uns):
                    error_message = (
                        'all SingleCell datasets must have the same uns, '
                        'unless flexible=True')
                    raise ValueError(error_message)
            # Check that all obs have the same columns and data types
            schema = self._obs.schema
            for dataset in datasets[1:]:
                if dataset._obs.schema != schema:
                    error_message = (
                        'all SingleCell datasets must have the same columns '
                        'in obs, with the same data types, unless '
                        'flexible=True')
                    raise ValueError(error_message)
            # Check that all obsm have the same keys and data types
            obsm = self._obsm
            for dataset in datasets[1:]:
                if dataset._obsm.keys() != obsm.keys() or any(
                        dataset._obsm[key].dtype != obsm[key].dtype
                        for key in obsm):
                    error_message = (
                        'all SingleCell datasets must have the same keys in '
                        'obsm, with the same data types, unless flexible=True')
                    raise ValueError(error_message)
        # Concatenate; output should be CSR when there's a mix of inputs
        X = vstack([dataset._X for dataset in datasets],
                   format='csr' if any(isinstance(dataset._X, csr_array)
                                       for dataset in datasets) else 'csc')
        obs = pl.concat([dataset._obs for dataset in datasets])
        obsm = {key: np.concatenate([dataset._obsm[key]
                                     for dataset in datasets])
                for key in self._obsm}
        return SingleCell(X=X, obs=obs, var=datasets[0]._var, obsm=obsm,
                          varm=datasets[0]._varm, uns=datasets[0]._uns)

    def concat_var(self,
                   datasets: SingleCell,
                   *more_datasets: SingleCell,
                   flexible: bool = False) -> SingleCell:
        """
        Concatenate the genes of multiple SingleCell datasets.
        
        By default, all datasets must have the same obs, obsm and uns. They
        must also have the same columns in var and the same keys in varm, with
        the same data types.
        
        Conversely, if `flexible=True`, subset to genes present in all datasets
        (according to the first column of obs, i.e. `obs_names`) before
        concatenating. Subset to columns of obs and keys of obsm and uns that
        are identical in all datasets after this subsetting. Also, subset to
        columns of var and keys of varm that are present in all datasets, and
        have the same data types. All datasets' `var_names` must have the same
        name and dtype, and similarly for `obs_names`.
        
        The one exception to the var "same data type" rule: if a column is Enum
        in some datasets and Categorical in others, or Enum in all datasets but
        with different categories in each dataset, that column will be retained
        as an Enum column (with the union of the categories) in the
        concatenated var.
        
        Args:
            datasets: one or more SingleCell datasets to concatenate with this
                      one
            *more_datasets: additional SingleCell datasets to concatenate with
                            this one, specified as positional arguments
            flexible: whether to subset to genes, columns of var and obs, and
                      keys of varm, obsm and uns common to all datasets before
                      concatenating, rather than raising an error on any
                      mismatches
        
        Returns:
            The concatenated SingleCell dataset.
        """
        # Check inputs
        if isinstance(datasets, Pseudobulk):
            datasets = datasets,
        datasets = (self,) + datasets + more_datasets
        if len(datasets) == 1:
            error_message = \
                'need at least one other SingleCell dataset to concatenate'
            raise ValueError(error_message)
        check_types(datasets[1:], 'datasets', SingleCell,
                    'SingleCell datasets')
        check_type(flexible, 'flexible', bool, 'Boolean')
        # Perform either flexible or non-flexible concatenation
        if flexible:
            # Check that `var_names` and `obs_names` have the same name and
            # data type across all datasets
            var_names_name = self.var_names.name
            if not all(dataset.var_names.name == var_names_name
                       for dataset in datasets[1:]):
                error_message = (
                    'not all SingleCell datasets have the same name for the '
                    'first column of var (the var_names column)')
                raise ValueError(error_message)
            obs_names_name = self.obs_names.name
            if not all(dataset.obs_names.name == obs_names_name
                       for dataset in datasets[1:]):
                error_message = (
                    'not all SingleCell datasets have the same name for the '
                    'first column of obs (the obs_names column)')
                raise ValueError(error_message)
            var_names_dtype = self.var_names.dtype
            if not all(dataset.var_names.dtype == var_names_dtype
                       for dataset in datasets[1:]):
                error_message = (
                    'not all SingleCell datasets have the same data type for '
                    'the first column of var (the var_names column)')
                raise TypeError(error_message)
            obs_names_dtype = self.obs_names.dtype
            if not all(dataset.obs_names.dtype == obs_names_dtype
                       for dataset in datasets[1:]):
                error_message = (
                    'not all SingleCell datasets have the same data type for '
                    'the first column of obs (the obs_names column)')
                raise TypeError(error_message)
            # Subset to genes in common across all datasets
            genes_in_common = self.obs_names\
                .filter(self.obs_names
                        .is_in(pl.concat([dataset.obs_names
                                          for dataset in datasets[1:]])))
            if len(genes_in_common) == 0:
                error_message = \
                    'no genes are shared across all SingleCell datasets'
                raise ValueError(error_message)
            datasets = [dataset[:, genes_in_common] for dataset in datasets]
            # Subset to columns of obs and keys of obsm and uns that are
            # identical in all datasets after this subsetting
            obs_columns_in_common = [
                column.name for column in datasets[0]._obs[:, 1:]
                if all(column.name in dataset._obs and
                       dataset._obs[column.name].equals(column)
                       for dataset in datasets[1:])]
            obsm = self._obsm
            obsm_keys_in_common = [
                key for key in obsm
                if all(key in dataset._obsm and
                       dataset._obsm[key].dtype == obsm[key].dtype and
                       (dataset._obsm[key] == obsm[key]).all()
                       for dataset in datasets[1:])]
            # noinspection PyTypeChecker,PyUnresolvedReferences
            uns_keys_in_common = [
                key for key, value in self._uns.items()
                if isinstance(value, dict) and
                   all(isinstance(dataset._uns[key], dict) and
                       SingleCell._eq_uns(value, dataset._uns[key])
                       for dataset in datasets[1:]) or
                   isinstance(value, np.ndarray) and
                   all(isinstance(dataset._uns[key], np.ndarray) and
                       (value == dataset._uns[key]).all()
                       for dataset in datasets[1:]) or
                   all(not isinstance(dataset._uns[key], (dict, np.ndarray))
                       and value == dataset._uns[key]
                       for dataset in datasets[1:])]
            for dataset in datasets:
                dataset._obs = dataset._obs.select(dataset.obs_names,
                                                   *obs_columns_in_common)
                dataset._obsm = {key: dataset._obsm[key]
                                  for key in obsm_keys_in_common}
                dataset._uns = {key: dataset._uns[key]
                                for key in uns_keys_in_common}
            # Subset to columns of var and keys of varm that are present in all
            # datasets, and have the same data types. Also include columns of
            # var that are Enum in some datasets and Categorical in others, or
            # Enum in all datasets but with different categories in each
            # dataset; cast these to Categorical.
            var_mismatched_categoricals = {
                column for column, dtype in self._var[:, 1:]
                .select(pl.col(pl.Categorical, pl.Enum)).schema.items()
                if all(column in dataset._var and
                       dataset._var[column].dtype in (pl.Categorical, pl.Enum)
                       for dataset in datasets[1:]) and
                   not all(dataset._var[column].dtype == dtype
                           for dataset in datasets[1:])}
            var_columns_in_common = [
                column for column, dtype in islice(self._var.schema.items(), 1)
                if column in var_mismatched_categoricals or
                   all(column in dataset._var and
                       dataset._var[column].dtype == dtype
                       for dataset in datasets[1:])]
            cast_dict = {column: pl.Enum(
                pl.concat([dataset._var[column].cat.get_categories()
                           for dataset in datasets])
                .unique(maintain_order=True))
                for column in var_mismatched_categoricals}
            for dataset in datasets:
                dataset._var = dataset._var\
                    .select(dataset.var_names, *var_columns_in_common)\
                    .cast(cast_dict)
            varm_keys_in_common = [
                key for key in self._varm
                if all(key in dataset._varm and
                       dataset._varm[key].dtype == self._varm[key].dtype
                       for dataset in datasets[1:])]
            for dataset in datasets:
                dataset._varm = {key: dataset._varm[key]
                                  for key in varm_keys_in_common}
        else:  # non-flexible
            # Check that all obs, obsm and uns are identical
            obs = self._obs
            for dataset in datasets[1:]:
                if not dataset._obs.equals(obs):
                    error_message = (
                        'all SingleCell datasets must have the same obs, '
                        'unless flexible=True')
                    raise ValueError(error_message)
            obsm = self._obsm
            for dataset in datasets[1:]:
                if dataset._obsm.keys() != obsm.keys() or \
                        any(dataset._obsm[key].dtype != obsm[key].dtype
                            for key in obsm) or \
                        any((dataset._obsm[key] != obsm[key]).any()
                            for key in obsm):
                    error_message = (
                        'all SingleCell datasets must have the same obsm, '
                        'unless flexible=True')
                    raise ValueError(error_message)
            for dataset in datasets[1:]:
                if not SingleCell._eq_uns(self._uns, dataset._uns):
                    error_message = (
                        'all SingleCell datasets must have the same uns, '
                        'unless flexible=True')
                    raise ValueError(error_message)
            # Check that all var have the same columns and data types
            schema = self._var.schema
            for dataset in datasets[1:]:
                if dataset._var.schema != schema:
                    error_message = (
                        'all SingleCell datasets must have the same columns '
                        'in var, with the same data types, unless '
                        'flexible=True')
                    raise ValueError(error_message)
            # Check that all varm have the same keys and data types
            varm = self._varm
            for dataset in datasets[1:]:
                if dataset._varm.keys() != varm.keys() or any(
                        dataset._varm[key].dtype != varm[key].dtype
                        for key in varm):
                    error_message = (
                        'all SingleCell datasets must have the same keys in '
                        'varm, with the same data types, unless flexible=True')
                    raise ValueError(error_message)
        # Concatenate; output should be CSR when there's a mix of inputs
        X = hstack([dataset._X for dataset in datasets],
                   format='csr' if any(isinstance(dataset._X, csr_array)
                                       for dataset in datasets) else 'csc')
        var = pl.concat([dataset._var for dataset in datasets])
        varm = {key: np.concatenate([dataset._varm[key]
                                     for dataset in datasets])
                for key in self._varm}
        return SingleCell(X=X, obs=datasets[0]._obs, var=var,
                          obsm=datasets[0]._obsm, varm=varm,
                          uns=datasets[0]._uns)

    def split_by_obs(self,
                     column: SingleCellColumn,
                     *,
                     QC_column: SingleCellColumn | None = 'passed_QC',
                     sort: bool = False) -> tuple[SingleCell, ...]:
        """
        The opposite of concat_obs(): splits a SingleCell dataset into a tuple
        of SingleCell datasets, one per unique value of a column of obs.

        Args:
            column: a String, Categorical or Enum column of obs to split by.
                    Can be a column name, a polars expression, a polars Series,
                    a 1D NumPy array, or a function that takes in this 
                    SingleCell dataset and returns a polars Series or 1D NumPy
                    array. Can contain null entries: the corresponding cells
                    will not be included in the result.
            QC_column: an optional Boolean column of obs indicating which cells
                       passed QC. Can be a column name, a polars expression, a
                       polars Series, a 1D NumPy array, or a function that
                       takes in this SingleCell dataset and returns a polars
                       Series or 1D NumPy array. Set to None to include all
                       cells. Cells failing QC will not be selected when
                       subsampling.
            sort: if True, sort the SingleCell datasets in the returned tuple
                  in decreasing size. If False, sort in order of each value's
                  first appearance in `column`.
        
        Returns:
            A tuple of SingleCell datasets, one per unique value of `column`.
        """
        if QC_column is not None:
            QC_column = self._get_column(
                'obs', QC_column, 'QC_column', pl.Boolean,
                allow_missing=QC_column == 'passed_QC')
        column = self._get_column('obs', column, 'column',
                                  (pl.String, pl.Categorical, pl.Enum),
                                  QC_column=QC_column, allow_null=True)
        check_type(sort, 'sort', pl.Boolean, 'Boolean')
        values = (column.value_counts(sort=True).to_series().drop_nulls()
                  if sort else column.unique(maintain_order=True))
        if QC_column is None:
            return tuple(self.filter_obs(column == value) for value in values)
        else:
            return tuple(self.filter_obs(column.eq(value) & QC_column)
                         for value in values)
    
    def split_by_var(self,
                     column: SingleCellColumn,
                     *,
                     sort: bool = False) -> tuple[SingleCell, ...]:
        """
        The opposite of concat_var(): splits a SingleCell dataset into a tuple
        of SingleCell datasets, one per unique value of a column of var.

        Args:
            column: a String, Categorical or Enum column of var to split by.
                    Can be a column name, a polars expression, a polars Series,
                    a 1D NumPy array, or a function that takes in this 
                    SingleCell dataset and returns a polars Series or 1D NumPy
                    array. Can contain null entries: the corresponding genes
                    will not be included in the result.
            sort: if True, sort the SingleCell datasets in the returned tuple
                  in decreasing size. If False, sort in order of each value's
                  first appearance in `column`.
        
        Returns:
            A tuple of SingleCell datasets, one per unique value of `column`.
        """
        column = self._get_column('var', column, 'column',
                                  (pl.String, pl.Categorical, pl.Enum),
                                  allow_null=True)
        check_type(sort, 'sort', pl.Boolean, 'Boolean')
        return tuple(self.filter_var(column == value) for value in
                     (column.value_counts(sort=True).to_series().drop_nulls()
                      if sort else column.unique(maintain_order=True)))
    
    def tocsr(self) -> SingleCell:
        """
        Make a copy of this SingleCell dataset, converting X to a csr_array.
        Raise an error if X is already a csr_array.
        
        Returns:
            A copy of this SingleCell dataset, with X as a csr_array.
        """
        if isinstance(self._X, csr_array):
            error_message = 'X is already a csr_array'
            raise TypeError(error_message)
        return SingleCell(X=self._X.tocsr(), obs=self._obs, var=self._var,
                          obsm=self._obsm, varm=self._varm, uns=self._uns)
    
    def tocsc(self) -> SingleCell:
        """
        Make a copy of this SingleCell dataset, converting X to a csc_array.
        Raise an error if X is already a csc_array.
        
        This function is provided for completeness, but csc_array is a far
        better format for cell-wise operations like pseudobulking.
        
        Returns:
            A copy of this SingleCell dataset, with X as a csc_array.
        """
        if isinstance(self._X, csc_array):
            error_message = 'X is already a csc_array'
            raise TypeError(error_message)
        return SingleCell(X=self._X.tocsc(), obs=self._obs, var=self._var,
                          obsm=self._obsm, varm=self._varm, uns=self._uns)
    
    def filter_obs(self,
                   *predicates: pl.Expr | pl.Series | str |
                                Iterable[pl.Expr | pl.Series | str] | bool |
                                list[bool] | np.ndarray[1, np.bool_],
                   **constraints: Any) -> SingleCell:
        """
        Equivalent to `df.filter()` from polars, but applied to both obs/obsm
        and X.
        
        Args:
            *predicates: one or more column names, expressions that evaluate to
                         Boolean Series, Boolean Series, lists of Booleans,
                         and/or 1D Boolean NumPy arrays
            **constraints: column filters: `name=value` filters to cells
                           where the column named `name` has the value `value`
        
        Returns:
            A new SingleCell dataset filtered to cells passing all the
            Boolean filters in `predicates` and `constraints`.
        """
        obs = self._obs\
            .with_row_index('_SingleCell_index')\
            .filter(*predicates, **constraints)
        mask = obs['_SingleCell_index'].to_numpy()
        return SingleCell(X=self._X[mask],
                          obs=obs.drop('_SingleCell_index'), var=self._var,
                          obsm={key: value[mask]
                                for key, value in self._obsm.items()},
                          varm=self._varm, uns=self._uns)
    
    def filter_var(self,
                   *predicates: pl.Expr | pl.Series | str |
                                Iterable[pl.Expr | pl.Series | str] | bool |
                                list[bool] | np.ndarray[1, np.bool_],
                   **constraints: Any) -> SingleCell:
        """
        Equivalent to `df.filter()` from polars, but applied to both var/varm
        and X.
        
        Args:
            *predicates: one or more column names, expressions that evaluate to
                         Boolean Series, Boolean Series, lists of Booleans,
                         and/or 1D Boolean NumPy arrays
            **constraints: column filters: `name=value` filters to genes
                           where the column named `name` has the value `value`
        
        Returns:
            A new SingleCell dataset filtered to genes passing all the
            Boolean filters in `predicates` and `constraints`.
        """
        var = self._var\
            .with_row_index('_SingleCell_index')\
            .filter(*predicates, **constraints)
        return SingleCell(X=self._X[:, var['_SingleCell_index'].to_numpy()],
                          obs=self._obs, var=var.drop('_SingleCell_index'),
                          obsm=self._obsm, varm=self._varm, uns=self._uns)
    
    def select_obs(self,
                   *exprs: Scalar | pl.Expr | pl.Series |
                           Iterable[Scalar | pl.Expr | pl.Series],
                   **named_exprs: Scalar | pl.Expr | pl.Series) -> SingleCell:
        """
        Equivalent to `df.select()` from polars, but applied to obs. obs_names
        will be automatically included as the first column, if not included
        explicitly.
        
        Args:
            *exprs: column(s) to select, specified as positional arguments.
                    Accepts expression input. Strings are parsed as column
                    names, other non-expression inputs are parsed as literals.
            **named_exprs: additional columns to select, specified as keyword
                           arguments. The columns will be renamed to the
                           keyword used.
        
        Returns:
            A new SingleCell dataset with
            `obs=obs.select(*exprs, **named_exprs)`, and obs_names as the first
            column unless already included explicitly.
        """
        obs = self._obs.select(*exprs, **named_exprs)
        if self.obs_names.name not in obs:
            obs = obs.select(self.obs_names, pl.all())
        return SingleCell(X=self._X, obs=obs, var=self._var, obsm=self._obsm,
                          varm=self._varm, uns=self._uns)
    
    def select_var(self,
                   *exprs: Scalar | pl.Expr | pl.Series |
                           Iterable[Scalar | pl.Expr | pl.Series],
                   **named_exprs: Scalar | pl.Expr | pl.Series) -> SingleCell:
        """
        Equivalent to `df.select()` from polars, but applied to var. var_names
        will be automatically included as the first column, if not included
        explicitly.
        
        Args:
            *exprs: column(s) to select, specified as positional arguments.
                    Accepts expression input. Strings are parsed as column
                    names, other non-expression inputs are parsed as literals.
            **named_exprs: additional columns to select, specified as keyword
                           arguments. The columns will be renamed to the
                           keyword used.
        
        Returns:
            A new SingleCell dataset with
            `var=var.select(*exprs, **named_exprs)`, and var_names as the first
            column unless already included explicitly.
        """
        var = self._var.select(*exprs, **named_exprs)
        if self.var_names.name not in var:
            var = var.select(self.var_names, pl.all())
        return SingleCell(X=self._X, obs=self._obs, var=var, obsm=self._obsm,
                          varm=self._varm, uns=self._uns)

    def select_obsm(self, keys: str | Iterable[str], *more_keys: str) -> \
            SingleCell:
        """
        Subsets obsm to the specified key(s).
        
        Args:
            keys: key(s) to select
            *more_keys: additional keys to select, specified as positional
                        arguments
        
        Returns:
            A new SingleCell dataset with obsm subset to the specified key(s).
        """
        keys = to_tuple(keys) + more_keys
        check_types(keys, 'keys', str, 'strings')
        for key in keys:
            if key not in self._obsm:
                error_message = \
                    f'tried to select {key!r}, which is not a key of obsm'
                raise ValueError(error_message)
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm={key: value for key, value in self._obsm.items()
                                if key in keys},
                          varm=self._varm, uns=self._uns)
    
    def select_varm(self, keys: str | Iterable[str], *more_keys: str) -> \
            SingleCell:
        """
        Subsets varm to the specified key(s).
        
        Args:
            keys: key(s) to select
            *more_keys: additional keys to select, specified as positional
                        arguments
        
        Returns:
            A new SingleCell dataset with varm subset to the specified key(s).
        """
        keys = to_tuple(keys) + more_keys
        check_types(keys, 'keys', str, 'strings')
        for key in keys:
            if key not in self._varm:
                error_message = \
                    f'tried to select {key!r}, which is not a key of varm'
                raise ValueError(error_message)
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm=self._obsm,
                          varm={key: value for key, value in self._varm.items()
                                if key in keys},
                          uns=self._uns)
    
    def select_uns(self, keys: str | Iterable[str], *more_keys: str) -> \
            SingleCell:
        """
        Subsets uns to the specified key(s).
        
        Args:
            keys: key(s) to select
            *more_keys: additional keys to select, specified as positional
                        arguments
        
        Returns:
            A new SingleCell dataset with uns subset to the specified key(s).
        """
        keys = to_tuple(keys) + more_keys
        check_types(keys, 'keys', str, 'strings')
        for key in keys:
            if key not in self._uns:
                error_message = \
                    f'tried to select {key!r}, which is not a key of uns'
                raise ValueError(error_message)
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm=self._obsm, varm=self._varm,
                          uns={key: value for key, value in self._uns.items()
                                if key in keys})
    
    def with_columns_obs(self,
                         *exprs: Scalar | pl.Expr | pl.Series |
                                 Iterable[Scalar | pl.Expr | pl.Series],
                         **named_exprs: Scalar | pl.Expr | pl.Series) -> \
            SingleCell:
        """
        Equivalent to `df.with_columns()` from polars, but applied to obs.
        
        Args:
            *exprs: column(s) to add, specified as positional arguments.
                    Accepts expression input. Strings are parsed as column
                    names, other non-expression inputs are parsed as literals.
            **named_exprs: additional columns to add, specified as keyword
                           arguments. The columns will be renamed to the
                           keyword used.
        
        Returns:
            A new SingleCell dataset with
            `obs=obs.with_columns(*exprs, **named_exprs)`.
        """
        # noinspection PyTypeChecker
        return SingleCell(X=self._X,
                          obs=self._obs.with_columns(*exprs, **named_exprs),
                          var=self._var, obsm=self._obsm, varm=self._varm,
                          uns=self._uns)
    
    def with_columns_var(self,
                         *exprs: Scalar | pl.Expr | pl.Series |
                                 Iterable[Scalar | pl.Expr | pl.Series],
                         **named_exprs: Scalar | pl.Expr | pl.Series) -> \
            SingleCell:
        """
        Equivalent to `df.with_columns()` from polars, but applied to var.
        
        Args:
            *exprs: column(s) to add, specified as positional arguments.
                    Accepts expression input. Strings are parsed as column
                    names, other non-expression inputs are parsed as literals.
            **named_exprs: additional columns to add, specified as keyword
                           arguments. The columns will be renamed to the
                           keyword used.
        
        Returns:
            A new SingleCell dataset with
            `var=var.with_columns(*exprs, **named_exprs)`.
        """
        # noinspection PyTypeChecker
        return SingleCell(X=self._X, obs=self._obs,
                          var=self._var.with_columns(*exprs, **named_exprs),
                          obsm=self._obsm, varm=self._varm, uns=self._uns)
    
    def with_obsm(self,
                  obsm: dict[str, np.ndarray[2, Any]] = {},
                  **more_obsm: np.ndarray[2, Any]) -> SingleCell:
        """
        Adds one or more keys to obsm, overwriting existing keys with the same
        names if present.
        
        Args:
            obsm: a dictionary of keys to add to (or overwrite in) obsm
            **more_obsm: additional keys to add to (or overwrite in) obsm,
                         specified as keyword arguments

        Returns:
            A new SingleCell dataset with the new key(s) added to or
            overwritten in obsm.
        """
        check_type(obsm, 'obsm', dict, 'a dictionary')
        for key, value in obsm.items():
            if not isinstance(key, str):
                error_message = (
                    f'all keys of obsm must be strings, but it contains a key '
                    f'of type {type(key).__name__!r}')
                raise TypeError(error_message)
        obsm |= more_obsm
        if len(obsm) == 0:
            error_message = \
                'obsm is empty and no keyword arguments were specified'
            raise ValueError(error_message)
        for key, value in obsm.items():
            if not isinstance(value, np.ndarray):
                error_message = (
                    f'all values of obsm must be NumPy arrays, but '
                    f'obsm[{key!r}] has type {type(value).__name__!r}')
                raise TypeError(error_message)
            if value.ndim != 2:
                error_message = (
                    f'all values of obsm must be 2D NumPy arrays, but '
                    f'obsm[{key!r}] is {value.ndim:,}D')
                raise ValueError(error_message)
            if len(value) != self._X.shape[0]:
                error_message = (
                    f'len(obsm[{key!r}]) is {len(value):,}, but X.shape[0] is '
                    f'{self._X.shape[0]:,}')
                raise ValueError(error_message)
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm=self._obsm | obsm, varm=self._varm,
                          uns=self._uns)
    
    def with_varm(self,
                  varm: dict[str, np.ndarray[2, Any]] = {},
                  **more_varm: np.ndarray[2, Any]) -> SingleCell:
        """
        Adds one or more keys to varm, overwriting existing keys with the same
        names if present.
        
        Args:
            varm: a dictionary of keys to add to (or overwrite in) varm
            **more_varm: additional keys to add to (or overwrite in) varm,
                         specified as keyword arguments

        Returns:
            A new SingleCell dataset with the new key(s) added to or
            overwritten in varm.
        """
        check_type(varm, 'varm', dict, 'a dictionary')
        for key, value in varm.items():
            if not isinstance(key, str):
                error_message = (
                    f'all keys of varm must be strings, but it contains a key '
                    f'of type {type(key).__name__!r}')
                raise TypeError(error_message)
        varm |= more_varm
        if len(varm) == 0:
            error_message = \
                'varm is empty and no keyword arguments were specified'
            raise ValueError(error_message)
        for key, value in varm.items():
            if not isinstance(value, np.ndarray):
                error_message = (
                    f'all values of varm must be NumPy arrays, but '
                    f'varm[{key!r}] has type {type(value).__name__!r}')
                raise TypeError(error_message)
            if value.ndim != 2:
                error_message = (
                    f'all values of varm must be 2D NumPy arrays, but '
                    f'varm[{key!r}] is {value.ndim:,}D')
                raise ValueError(error_message)
            if len(value) != self._X.shape[1]:
                error_message = (
                    f'len(varm[{key!r}]) is {len(value):,}, but X.shape[1] is '
                    f'{self._X.shape[1]:,}')
                raise ValueError(error_message)
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm=self._obsm, varm=self._varm | varm,
                          uns=self._uns)
    
    def with_uns(self,
                 uns: dict[str, NestedScalarOrArrayDict] = {},
                  **more_uns: NestedScalarOrArrayDict) -> SingleCell:
        """
        Adds one or more keys to uns, overwriting existing keys with the same
        names if present.
        
        Args:
            uns: a dictionary of keys to add to (or overwrite in) uns
            **more_uns: additional keys to add to (or overwrite in) uns,
                        specified as keyword arguments

        Returns:
            A new SingleCell dataset with the new key(s) added to or
            overwritten in uns.
        """
        check_type(uns, 'uns', dict, 'a dictionary')
        for key, value in uns.items():
            if not isinstance(key, str):
                error_message = (
                    f'all keys of uns must be strings, but it contains a key '
                    f'of type {type(key).__name__!r}')
                raise TypeError(error_message)
        uns |= more_uns
        if len(uns) == 0:
            error_message = \
                'uns is empty and no keyword arguments were specified'
            raise ValueError(error_message)
        valid_uns_types = str, int, np.integer, float, np.floating, \
            bool, np.bool_, np.ndarray
        for description, value in SingleCell._iter_uns(uns):
            if not isinstance(value, valid_uns_types):
                error_message = (
                    f'all values of uns must be scalars (strings, numbers or '
                    f'Booleans) or NumPy arrays, or nested dictionaries '
                    f'thereof, but {description} has type '
                    f'{type(value).__name__!r}')
                raise TypeError(error_message)
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm=self._obsm, varm=self._varm,
                          uns=self._uns | uns)
    
    def drop_obs(self,
                 columns: pl.type_aliases.ColumnNameOrSelector |
                          Iterable[pl.type_aliases.ColumnNameOrSelector],
                 *more_columns: pl.type_aliases.ColumnNameOrSelector) -> \
            SingleCell:
        """
        Create a new SingleCell dataset with `columns` and `more_columns`
        removed from obs.
        
        Args:
            columns: columns(s) to drop
            *more_columns: additional columns to drop, specified as
                              positional arguments
        
        Returns:
            A new SingleCell dataset with the column(s) removed.
        """
        columns = to_tuple(columns) + more_columns
        return SingleCell(X=self._X, obs=self._obs.drop(columns),
                          var=self._var, obsm=self._obsm, varm=self._varm,
                          uns=self._uns)

    def drop_var(self,
                 columns: pl.type_aliases.ColumnNameOrSelector |
                          Iterable[pl.type_aliases.ColumnNameOrSelector],
                 *more_columns: pl.type_aliases.ColumnNameOrSelector) -> \
            SingleCell:
        """
        Create a new SingleCell dataset with `columns` and `more_columns`
        removed from var.
        
        Args:
            columns: columns(s) to drop
            *more_columns: additional columns to drop, specified as
                           positional arguments
        
        Returns:
            A new SingleCell dataset with the column(s) removed.
        """
        columns = to_tuple(columns) + more_columns
        return SingleCell(X=self._X, obs=self._obs,
                          var=self._var.drop(columns), obsm=self._obsm,
                          varm=self._varm, uns=self._uns)
    
    def drop_obsm(self, keys: str | Iterable[str], *more_keys: str) -> \
            SingleCell:
        """
        Create a new SingleCell dataset with `keys` and `more_keys` removed
        from obsm.
        
        Args:
            keys: key(s) to drop
            *more_keys: additional keys to drop, specified as positional
                        arguments
        
        Returns:
            A new SingleCell dataset with the specified key(s) removed from
            obsm.
        """
        keys = to_tuple(keys) + more_keys
        check_types(keys, 'keys', str, 'strings')
        for key in keys:
            if key not in self._obsm:
                error_message = \
                    f'tried to drop {key!r}, which is not a key of obsm'
                raise ValueError(error_message)
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm={key: value for key, value in self._obsm.items()
                                if key not in keys},
                          varm=self._varm, uns=self._uns)
    
    def drop_varm(self, keys: str | Iterable[str], *more_keys: str) -> \
            SingleCell:
        """
        Create a new SingleCell dataset with `keys` and `more_keys` removed
        from varm.
        
        Args:
            keys: key(s) to drop
            *more_keys: additional keys to drop, specified as positional
                        arguments
        
        Returns:
            A new SingleCell dataset with the specified key(s) removed from
            varm.
        """
        keys = to_tuple(keys) + more_keys
        check_types(keys, 'keys', str, 'strings')
        for key in keys:
            if key not in self._varm:
                error_message = \
                    f'tried to drop {key!r}, which is not a key of varm'
                raise ValueError(error_message)
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm=self._obsm,
                          varm={key: value for key, value in self._varm.items()
                                if key not in keys},
                          uns=self._uns)
    
    def drop_uns(self, keys: str | Iterable[str], *more_keys: str) -> \
            SingleCell:
        """
        Create a new SingleCell dataset with `keys` and `more_keys` removed
        from uns.
        
        Args:
            keys: key(s) to drop
            *more_keys: additional keys to drop, specified as positional
                        arguments
        
        Returns:
            A new SingleCell dataset with the specified key(s) removed from
            uns.
        """
        keys = to_tuple(keys) + more_keys
        check_types(keys, 'keys', str, 'strings')
        for key in keys:
            if key not in self._uns:
                error_message = \
                    f'tried to drop {key!r}, which is not a key of uns'
                raise ValueError(error_message)
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm=self._obsm, varm=self._varm,
                          uns={key: value for key, value in self._uns.items()
                                if key not in keys})
    
    def rename_obs(self, mapping: dict[str, str] | Callable[[str], str]) -> \
            SingleCell:
        """
        Create a new SingleCell dataset with column(s) of obs renamed.
        
        Rename column(s) of obs.
        
        Args:
            mapping: the renaming to apply, either as a dictionary with the old
                     names as keys and the new names as values, or a function
                     that takes an old name and returns a new name

        Returns:
            A new SingleCell dataset with the column(s) of obs renamed.
        """
        return SingleCell(X=self._X, obs=self._obs.rename(mapping),
                          var=self._var, obsm=self._obsm, varm=self._varm,
                          uns=self._uns)
    
    def rename_var(self, mapping: dict[str, str] | Callable[[str], str]) -> \
            SingleCell:
        """
        Create a new SingleCell dataset with column(s) of var renamed.
        
        Args:
            mapping: the renaming to apply, either as a dictionary with the old
                     names as keys and the new names as values, or a function
                     that takes an old name and returns a new name

        Returns:
            A new SingleCell dataset with the column(s) of var renamed.
        """
        return SingleCell(X=self._X, obs=self._obs,
                          var=self._var.rename(mapping), obsm=self._obsm,
                          varm=self._varm, uns=self._uns)
    
    def rename_obsm(self, mapping: dict[str, str] | Callable[[str], str]) -> \
            SingleCell:
        """
        Create a new SingleCell dataset with key(s) of obsm renamed.
        
        Args:
            mapping: the renaming to apply, either as a dictionary with the old
                     names as keys and the new names as values, or a function
                     that takes an old name and returns a new name

        Returns:
            A new SingleCell dataset with the key(s) of obsm renamed.
        """
        check_types(mapping.keys(), 'mapping.keys()', str, 'strings')
        check_types(mapping.values(), 'mapping.values()', str, 'strings')
        if isinstance(mapping, dict):
            for key, new_key in mapping.items():
                if key not in self._obsm:
                    error_message = \
                        f'tried to rename {key!r}, which is not a key of obsm'
                    raise ValueError(error_message)
                if new_key in self._obsm:
                    error_message = (
                        f'tried to rename obsm[{key!r}] to obsm[{new_key!r}], '
                        f'but obsm[{new_key!r}] already exists')
                    raise ValueError(error_message)
            obsm = {mapping.get(key, key): value
                    for key, value in self._obsm.items()}
        elif isinstance(mapping, Callable):
            obsm = {}
            for key, value in self._obsm.items():
                new_key = mapping(key)
                if not isinstance(new_key, str):
                    error_message = (
                        f'tried to rename obsm[{key!r}] to a non-string value '
                        f'of type {type(new_key).__name__!r}')
                    raise TypeError(error_message)
                if new_key in self._obsm:
                    error_message = (
                        f'tried to rename obsm[{key!r}] to obsm[{new_key!r}], '
                        f'but obsm[{new_key!r}] already exists')
                    raise ValueError(error_message)
                obsm[new_key] = value
        else:
            error_message = (
                f'mapping must be a dictionary or function, but has type '
                f'{type(mapping).__name__!r}')
            raise TypeError(error_message)
        return SingleCell(X=self._X, obs=self._obs, var=self._var, obsm=obsm,
                          varm=self._varm, uns=self._uns)
    
    def rename_varm(self, mapping: dict[str, str] | Callable[[str], str]) -> \
            SingleCell:
        """
        Create a new SingleCell dataset with key(s) of varm renamed.
        
        Args:
            mapping: the renaming to apply, either as a dictionary with the old
                     names as keys and the new names as values, or a function
                     that takes an old name and returns a new name

        Returns:
            A new SingleCell dataset with the key(s) of varm renamed.
        """
        check_types(mapping.keys(), 'mapping.keys()', str, 'strings')
        check_types(mapping.values(), 'mapping.values()', str, 'strings')
        if isinstance(mapping, dict):
            for key, new_key in mapping.items():
                if key not in self._varm:
                    error_message = \
                        f'tried to rename {key!r}, which is not a key of varm'
                    raise ValueError(error_message)
                if new_key in self._varm:
                    error_message = (
                        f'tried to rename varm[{key!r}] to varm[{new_key!r}], '
                        f'but varm[{new_key!r}] already exists')
                    raise ValueError(error_message)
            varm = {mapping.get(key, key): value
                    for key, value in self._varm.items()}
        elif isinstance(mapping, Callable):
            varm = {}
            for key, value in self._varm.items():
                new_key = mapping(key)
                if not isinstance(new_key, str):
                    error_message = (
                        f'tried to rename varm[{key!r}] to a non-string value '
                        f'of type {type(new_key).__name__!r}')
                    raise TypeError(error_message)
                if new_key in self._varm:
                    error_message = (
                        f'tried to rename varm[{key!r}] to varm[{new_key!r}], '
                        f'but varm[{new_key!r}] already exists')
                    raise ValueError(error_message)
                varm[new_key] = value
        else:
            error_message = (
                f'mapping must be a dictionary or function, but has type '
                f'{type(mapping).__name__!r}')
            raise TypeError(error_message)
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm=self._obsm, varm=varm, uns=self._uns)
    
    def rename_uns(self, mapping: dict[str, str] | Callable[[str], str]) -> \
            SingleCell:
        """
        Create a new SingleCell dataset with key(s) of uns renamed.
        
        Args:
            mapping: the renaming to apply, either as a dictionary with the old
                     names as keys and the new names as values, or a function
                     that takes an old name and returns a new name

        Returns:
            A new SingleCell dataset with the key(s) of uns renamed.
        """
        check_types(mapping.keys(), 'mapping.keys()', str, 'strings')
        check_types(mapping.values(), 'mapping.values()', str, 'strings')
        if isinstance(mapping, dict):
            for key, new_key in mapping.items():
                if key not in self._uns:
                    error_message = \
                        f'tried to rename {key!r}, which is not a key of uns'
                    raise ValueError(error_message)
                if new_key in self._uns:
                    error_message = (
                        f'tried to rename uns[{key!r}] to uns[{new_key!r}], '
                        f'but uns[{new_key!r}] already exists')
                    raise ValueError(error_message)
            uns = {mapping.get(key, key): value
                   for key, value in self._uns.items()}
        elif isinstance(mapping, Callable):
            uns = {}
            for key, value in self._uns.items():
                new_key = mapping(key)
                if not isinstance(new_key, str):
                    error_message = (
                        f'tried to rename uns[{key!r}] to a non-string value '
                        f'of type {type(new_key).__name__!r}')
                    raise TypeError(error_message)
                if new_key in self._uns:
                    error_message = (
                        f'tried to rename uns[{key!r}] to uns[{new_key!r}], '
                        f'but uns[{new_key!r}] already exists')
                    raise ValueError(error_message)
                uns[new_key] = value
        else:
            error_message = (
                f'mapping must be a dictionary or function, but has type '
                f'{type(mapping).__name__!r}')
            raise TypeError(error_message)
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm=self._obsm, varm=self._varm, uns=uns)
    
    def cast_X(self, dtype: np._typing.DTypeLike) -> SingleCell:
        """
        Cast X to the specified data type.
        
        Args:
            dtype: a NumPy data type

        Returns:
            A new SingleCell dataset with X cast to the specified data type.
        """
        return SingleCell(X=self._X.astype(dtype),
                          obs=self._obs, var=self._var, obsm=self._obsm,
                          varm=self._varm, uns=self._uns)
    
    def cast_obs(self,
                 dtypes: Mapping[pl.type_aliases.ColumnNameOrSelector |
                                 pl.type_aliases.PolarsDataType,
                                 pl.type_aliases.PolarsDataType] |
                         pl.type_aliases.PolarsDataType,
                 *,
                 strict: bool = True) -> SingleCell:
        """
        Cast column(s) of obs to the specified data type(s).
        
        Args:
            dtypes: a mapping of column names (or selectors) to data types, or
                    a single data type to which all columns will be cast
            strict: whether to raise an error if a cast could not be done (for
                    instance, due to numerical overflow)

        Returns:
            A new SingleCell dataset with column(s) of obs cast to the
            specified data type(s).
        """
        return SingleCell(X=self._X, obs=self._obs.cast(dtypes, strict=strict),
                          var=self._var, obsm=self._obsm, varm=self._varm,
                          uns=self._uns)
    
    def cast_var(self,
                 dtypes: Mapping[pl.type_aliases.ColumnNameOrSelector |
                                 pl.type_aliases.PolarsDataType,
                                 pl.type_aliases.PolarsDataType] |
                         pl.type_aliases.PolarsDataType,
                 *,
                 strict: bool = True) -> SingleCell:
        """
        Cast column(s) of var to the specified data type(s).
        
        Args:
            dtypes: a mapping of column names (or selectors) to data types, or
                    a single data type to which all columns will be cast
            strict: whether to raise an error if a cast could not be done (for
                    instance, due to numerical overflow)

        Returns:
            A new SingleCell dataset with column(s) of var cast to the
            specified data type(s).
        """
        return SingleCell(X=self._X, obs=self._obs,
                          var=self._var.cast(dtypes, strict=strict),
                          obsm=self._obsm, varm=self._varm, uns=self._uns)
    
    def join_obs(self,
                 other: pl.DataFrame,
                 on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
                 *,
                 left_on: str | pl.Expr | Sequence[str | pl.Expr] |
                          None = None,
                 right_on: str | pl.Expr | Sequence[str | pl.Expr] |
                           None = None,
                 suffix: str = '_right',
                 validate: Literal['m:m', 'm:1', '1:m', '1:1'] = 'm:m',
                 join_nulls: bool = False,
                 coalesce: bool = True) -> SingleCell:
        """
        Left join obs with another DataFrame.
        
        Args:
            other: a polars DataFrame to join obs with
            on: the name(s) of the join column(s) in both DataFrames
            left_on: the name(s) of the join column(s) in obs
            right_on: the name(s) of the join column(s) in `other`
            suffix: a suffix to append to columns with a duplicate name
            validate: checks whether the join is of the specified type. Can be:
                      - 'm:m' (many-to-many): the default, no checks performed.
                      - '1:1' (one-to-one): check that none of the values in
                        the join column(s) appear more than once in obs or more
                        than once in `other`.
                      - '1:m' (one-to-many): check that none of the values in
                        the join column(s) appear more than once in obs.
                      - 'm:1' (many-to-one): check that none of the values in
                        the join column(s) appear more than once in `other`.
            join_nulls: whether to include null as a valid value to join on.
                        By default, null values will never produce matches.
            coalesce: if True, coalesce each of the pairs of join columns
                      (the columns in `on` or `left_on`/`right_on`) from obs
                      and `other` into a single column, filling missing values
                      from one with the corresponding values from the other.
                      If False, include both as separate columns, adding
                      `suffix` to the join columns from `other`.
        
        Returns:
            A new SingleCell dataset with the columns from `other` joined to
            obs.
        
        Note:
            If a column of `on`, `left_on` or `right_on` is Enum in obs and
            Categorical in `other` (or vice versa), or Enum in both but with
            different categories in each, that pair of columns will be
            automatically cast to a common Enum data type (with the union of
            the categories) before joining.
        """
        # noinspection PyTypeChecker
        check_type(other, 'other', pl.DataFrame, 'a polars DataFrame')
        left = self._obs
        right = other
        if on is None:
            if left_on is None and right_on is None:
                error_message = (
                    "either 'on' or both of 'left_on' and 'right_on' must be "
                    "specified")
                raise ValueError(error_message)
            elif left_on is None:
                error_message = \
                    'right_on is specified, so left_on must be specified'
                raise ValueError(error_message)
            elif right_on is None:
                error_message = \
                    'left_on is specified, so right_on must be specified'
                raise ValueError(error_message)
            left_columns = left.select(left_on)
            right_columns = right.select(right_on)
        else:
            if left_on is not None:
                error_message = "'on' is specified, so 'left_on' must be None"
                raise ValueError(error_message)
            if right_on is not None:
                error_message = "'on' is specified, so 'right_on' must be None"
                raise ValueError(error_message)
            left_columns = left.select(on)
            right_columns = right.select(on)
        left_cast_dict = {}
        right_cast_dict = {}
        for left_column, right_column in zip(left_columns, right_columns):
            left_dtype = left_column.dtype
            right_dtype = right_column.dtype
            if left_dtype == right_dtype:
                continue
            if (left_dtype == pl.Enum or left_dtype == pl.Categorical) and (
                    right_dtype == pl.Enum or right_dtype == pl.Categorical):
                common_dtype = \
                    pl.Enum(pl.concat([left_column.cat.get_categories(),
                                       right_column.cat.get_categories()])
                            .unique(maintain_order=True))
                left_cast_dict[left_column.name] = common_dtype
                right_cast_dict[right_column.name] = common_dtype
            else:
                error_message = (
                    f'obs[{left_column.name!r}] has data type '
                    f'{left_dtype.base_type()!r}, but '
                    f'other[{right_column.name!r}] has data type '
                    f'{right_dtype.base_type()!r}')
                raise TypeError(error_message)
        if left_cast_dict is not None:
            left = left.cast(left_cast_dict)
            right = right.cast(right_cast_dict)
        obs = left.join(right, on=on, how='left', left_on=left_on,
                        right_on=right_on, suffix=suffix, validate=validate,
                        join_nulls=join_nulls, coalesce=coalesce)
        if len(obs) > len(self):
            other_on = to_tuple(right_on if right_on is not None else on)
            assert other.select(other_on).is_duplicated().any()
            duplicate_column = other_on[0] if len(other_on) == 1 else \
                next(column for column in other_on
                     if other[column].is_duplicated().any())
            error_message = (
                f'other[{duplicate_column!r}] contains duplicate values, so '
                f'it must be deduplicated before being joined on')
            raise ValueError(error_message)
        return SingleCell(X=self._X, obs=obs, var=self._var, obsm=self._obsm,
                          varm=self._varm, uns=self._uns)
    
    def join_var(self,
                 other: pl.DataFrame,
                 on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
                 *,
                 left_on: str | pl.Expr | Sequence[str | pl.Expr] |
                          None = None,
                 right_on: str | pl.Expr | Sequence[str | pl.Expr] |
                           None = None,
                 suffix: str = '_right',
                 validate: Literal['m:m', 'm:1', '1:m', '1:1'] = 'm:m',
                 join_nulls: bool = False,
                 coalesce: bool = True) -> SingleCell:
        """
        Join var with another DataFrame.
        
        Args:
            other: a polars DataFrame to join var with
            on: the name(s) of the join column(s) in both DataFrames
            left_on: the name(s) of the join column(s) in var
            right_on: the name(s) of the join column(s) in `other`
            suffix: a suffix to append to columns with a duplicate name
            validate: checks whether the join is of the specified type. Can be:
                      - 'm:m' (many-to-many): the default, no checks performed.
                      - '1:1' (one-to-one): check that none of the values in
                        the join column(s) appear more than once in var or more
                        than once in `other`.
                      - '1:m' (one-to-many): check that none of the values in
                        the join column(s) appear more than once in var.
                      - 'm:1' (many-to-one): check that none of the values in
                        the join column(s) appear more than once in `other`.
            join_nulls: whether to include null as a valid value to join on.
                        By default, null values will never produce matches.
            coalesce: if True, coalesce each of the pairs of join columns
                      (the columns in `on` or `left_on`/`right_on`) from obs
                      and `other` into a single column, filling missing values
                      from one with the corresponding values from the other.
                      If False, include both as separate columns, adding
                      `suffix` to the join columns from `other`.
        
        Returns:
            A new SingleCell dataset with the columns from `other` joined to
            var.
        
        Note:
            If a column of `on`, `left_on` or `right_on` is Enum in obs and
            Categorical in `other` (or vice versa), or Enum in both but with
            different categories in each, that pair of columns will be
            automatically cast to a common Enum data type (with the union of
            the categories) before joining.
        """
        check_type(other, 'other', pl.DataFrame, 'a polars DataFrame')
        left = self._var
        right = other
        if on is None:
            if left_on is None and right_on is None:
                error_message = (
                    "either 'on' or both of 'left_on' and 'right_on' must be "
                    "specified")
                raise ValueError(error_message)
            elif left_on is None:
                error_message = \
                    'right_on is specified, so left_on must be specified'
                raise ValueError(error_message)
            elif right_on is None:
                error_message = \
                    'left_on is specified, so right_on must be specified'
                raise ValueError(error_message)
            left_columns = left.select(left_on)
            right_columns = right.select(right_on)
        else:
            if left_on is not None:
                error_message = "'on' is specified, so 'left_on' must be None"
                raise ValueError(error_message)
            if right_on is not None:
                error_message = "'on' is specified, so 'right_on' must be None"
                raise ValueError(error_message)
            left_columns = left.select(on)
            right_columns = right.select(on)
        left_cast_dict = {}
        right_cast_dict = {}
        for left_column, right_column in zip(left_columns, right_columns):
            left_dtype = left_column.dtype
            right_dtype = right_column.dtype
            if left_dtype == right_dtype:
                continue
            if (left_dtype == pl.Enum or left_dtype == pl.Categorical) and (
                    right_dtype == pl.Enum or right_dtype == pl.Categorical):
                common_dtype = \
                    pl.Enum(pl.concat([left_column.cat.get_categories(),
                                       right_column.cat.get_categories()])
                            .unique(maintain_order=True))
                left_cast_dict[left_column.name] = common_dtype
                right_cast_dict[right_column.name] = common_dtype
            else:
                error_message = (
                    f'var[{left_column.name!r}] has data type '
                    f'{left_dtype.base_type()!r}, but '
                    f'other[{right_column.name!r}] has data type '
                    f'{right_dtype.base_type()!r}')
                raise TypeError(error_message)
        if left_cast_dict is not None:
            left = left.cast(left_cast_dict)
            right = right.cast(right_cast_dict)
        # noinspection PyTypeChecker
        var = left.join(right, on=on, how='left', left_on=left_on,
                        right_on=right_on, suffix=suffix, validate=validate,
                        join_nulls=join_nulls, coalesce=coalesce)
        if len(var) > len(self):
            other_on = to_tuple(right_on if right_on is not None else on)
            assert other.select(other_on).is_duplicated().any()
            duplicate_column = other_on[0] if len(other_on) == 1 else \
                next(column for column in other_on
                     if other[column].is_duplicated().any())
            error_message = (
                f'other[{duplicate_column!r}] contains duplicate values, so '
                f'it must be deduplicated before being joined on')
            raise ValueError(error_message)
        return SingleCell(X=self._X, obs=self._obs, var=var, obsm=self._obsm,
                          varm=self._varm, uns=self._uns)
    
    def peek_obs(self, row: int = 0) -> None:
        """
        Print a row of obs (the first row, by default) with each column on its
        own line.
        
        Args:
            row: the index of the row to print
        """
        with pl.Config(tbl_rows=-1):
            print(self._obs[row].unpivot(variable_name='column'))
    
    def peek_var(self, row: int = 0) -> None:
        """
        Print a row of var (the first row, by default) with each column on its
        own line.
        
        Args:
            row: the index of the row to print
        """
        with pl.Config(tbl_rows=-1):
            print(self._var[row].unpivot(variable_name='column'))
    
    def subsample_obs(self,
                      n: int | np.integer | None = None,
                      *,
                      fraction: int | float | np.integer | np.floating |
                                None = None,
                      QC_column: SingleCellColumn | None = 'passed_QC',
                      by_column: SingleCellColumn | None = None,
                      subsample_column: str | None = None,
                      seed: int | np.integer | None = 0) -> SingleCell:
        """
        Subsample a specific number or fraction of cells.
        
        Args:
            n: the number of cells to return; mutually exclusive with
               `fraction`
            fraction: the fraction of cells to return; mutually exclusive with
                      `n`
            QC_column: an optional Boolean column of obs indicating which cells
                       passed QC. Can be a column name, a polars expression, a 
                       polars Series, a 1D NumPy array, or a function that 
                       takes in this SingleCell dataset and returns a polars 
                       Series or 1D NumPy array. Set to None to include all
                       cells. Cells failing QC will not be selected when
                       subsampling, and will not count towards the denominator
                       of `fraction`; QC_column will not appear in the returned
                       SingleCell object, since it would be redundant.
            by_column: an optional String, Categorical, Enum, or integer column
                       of obs to subsample by. Can be a column name, a polars 
                       expression, a polars Series, a 1D NumPy array, or a 
                       function that takes in this SingleCell dataset and 
                       returns a polars Series or 1D NumPy array. Specifying
                       `by_column` ensures that the same fraction of cells with
                       each value of `by_column` are subsampled. When combined
                       with `n`, to make sure the total number of samples is
                       exactly `n`, some of the smallest groups may be
                       oversampled by one element, or some of the largest
                       groups may be undersampled by one element. Can contain
                       null entries: the corresponding cells will not be
                       included in the result.
            subsample_column: an optional name of a Boolean column to add to
                              obs indicating the subsampled genes; if None,
                              subset to these genes instead
            seed: the random seed to use when subsampling; if None, do not set
                  a seed
        
        Returns:
            A new SingleCell dataset subset to the subsampled cells, or if
            `subsample_column` is not None, the full dataset with
            `subsample_column` added to obs.
        """
        if n is not None:
            check_type(n, 'n', int, 'a positive integer')
            check_bounds(n, 'n', 1)
        elif fraction is not None:
            check_type(fraction, 'fraction', float,
                       'a floating-point number between 0 and 1')
            check_bounds(fraction, 'fraction', 0, 1, left_open=True,
                         right_open=True)
        else:
            error_message = 'one of n and fraction must be specified'
            raise ValueError(error_message)
        if n is not None and fraction is not None:
            error_message = 'only one of n and fraction must be specified'
            raise ValueError(error_message)
        if QC_column is not None:
            QC_column = self._get_column(
                'obs', QC_column, 'QC_column', pl.Boolean,
                allow_missing=QC_column == 'passed_QC')
        if subsample_column is not None:
            check_type(subsample_column, 'subsample_column', str, 'a string')
            if subsample_column in self._obs:
                error_message = (
                    f'subsample_column {subsample_column!r} is already a '
                    f'column of obs')
                raise ValueError(error_message)
        if seed is not None:
            check_type(seed, 'seed', int, 'an integer')
        if by_column is not None:
            by_column = self._get_column(
                'obs', by_column, 'by_column',
                (pl.String, pl.Categorical, pl.Enum, 'integer'),
                QC_column=QC_column, allow_null=True)
            if QC_column is not None:
                by_column = by_column.filter(QC_column)
            by_frame = by_column.to_frame()
            by_name = by_column.name
            if n is not None:
                # Get a vector of the number of elements to sample per group.
                # The total sample size should exactly match the original n; if
                # necessary, oversample the smallest groups or undersample the
                # largest groups to make this happen.
                group_counts = by_frame\
                    .group_by(by_name)\
                    .agg(pl.len(), n=(n * pl.len() / len(by_column))
                                     .round().cast(pl.Int32))\
                    .drop_nulls(by_name)
                diff = n - group_counts['n'].sum()
                if diff != 0:
                    group_counts = group_counts\
                        .sort('len', descending=diff < 0)\
                        .with_columns(n=pl.col.n +
                                      pl.int_range(pl.len()).lt(abs(diff))
                                      .cast(pl.Int32) * pl.lit(diff).sign())
                selected = by_frame\
                    .join(group_counts, on=by_name)\
                    .select(pl.int_range(pl.len())
                            .shuffle(seed=seed)
                            .over(by_name)
                            .lt(n))
            else:
                # noinspection PyUnresolvedReferences
                selected = by_frame\
                    .shuffle(seed=seed)\
                    .over(by_name)\
                    .lt((fraction * pl.len().over(by_name)).round())
        elif QC_column is not None:
            selected = pl.int_range(QC_column.sum(), eager=True)\
                .shuffle(seed=seed)\
                .lt(n if fraction is None else (fraction * pl.len()).round())
        else:
            selected = self._obs\
                .select(pl.int_range(pl.len())
                        .shuffle(seed=seed)
                        .lt(n if fraction is None else
                            (fraction * pl.len()).round()))
        selected = selected.to_series()
        if QC_column is not None:
            # Back-project from QCed cells to all cells, filling with nulls
            selected = pl.when(QC_column)\
                .then(selected.gather(QC_column.cum_sum() - 1))
        sc = self.filter_obs(selected) if subsample_column is None else \
            self.with_columns_obs(selected.alias(subsample_column))
        if QC_column is not None:
            # noinspection PyTypeChecker
            sc._obs = sc._obs.drop(QC_column.name)
        return sc
    
    def subsample_var(self,
                      n: int | np.integer | None = None,
                      *,
                      fraction: int | float | np.integer | np.floating |
                                None = None,
                      by_column: SingleCellColumn | None = None,
                      subsample_column: str | None = None,
                      seed: int | np.integer | None = 0) -> SingleCell:
        """
        Subsample a specific number or fraction of genes.
        
        Args:
            n: the number of genes to return; mutually exclusive with
               `fraction`
            fraction: the fraction of genes to return; mutually exclusive with
                      `n`
            by_column: an optional String, Categorical, Enum, or integer column
                       of var to subsample by. Can be a column name, a polars 
                       expression, a polars Series, a 1D NumPy array, or a 
                       function that takes in this SingleCell dataset and 
                       returns a polars Series or 1D NumPy array. Specifying
                       `by_column` ensures that the same fraction of genes with
                       each value of `by_column` are subsampled. When combined
                       with `n`, to make sure the total number of samples is
                       exactly `n`, some of the smallest groups may be
                       oversampled by one element, or some of the largest
                       groups may be undersampled by one element. Can contain
                       null entries: the corresponding genes will not be
                       included in the result.
            subsample_column: an optional name of a Boolean column to add to
                              var indicating the subsampled genes; if None,
                              subset to these genes instead
            seed: the random seed to use when subsampling; if None, do not set
                  a seed
        
        Returns:
            A new SingleCell dataset subset to the subsampled genes, or if
            `subsample_column` is not None, the full dataset with
            `subsample_column` added to var.
        """
        if n is not None:
            check_type(n, 'n', int, 'a positive integer')
            check_bounds(n, 'n', 1)
        elif fraction is not None:
            check_type(fraction, 'fraction', float,
                       'a floating-point number between 0 and 1')
            check_bounds(fraction, 'fraction', 0, 1, left_open=True,
                         right_open=True)
        else:
            error_message = 'one of n and fraction must be specified'
            raise ValueError(error_message)
        if n is not None and fraction is not None:
            error_message = 'only one of n and fraction must be specified'
            raise ValueError(error_message)
        if subsample_column is not None:
            check_type(subsample_column, 'subsample_column', str, 'a string')
            if subsample_column in self._var:
                error_message = (
                    f'subsample_column {subsample_column!r} is already a '
                    f'column of var')
                raise ValueError(error_message)
        if seed is not None:
            check_type(seed, 'seed', int, 'an integer')
        if by_column is not None:
            by_column = self._get_column(
                'var', by_column, 'by_column',
                (pl.String, pl.Categorical, pl.Enum, 'integer'),
                allow_null=True)
            by_frame = by_column.to_frame()
            by_name = by_column.name
            if n is not None:
                # Get a vector of the number of elements to sample per group.
                # The total sample size should exactly match the original n; if
                # necessary, oversample the smallest groups or undersample the
                # largest groups to make this happen.
                group_counts = by_frame\
                    .group_by(by_name)\
                    .agg(pl.len(), n=(n * pl.len() / len(by_column))
                                     .round().cast(pl.Int32))\
                    .drop_nulls(by_name)
                diff = n - group_counts['n'].sum()
                if diff != 0:
                    group_counts = group_counts\
                        .sort('len', descending=diff < 0)\
                        .with_columns(n=pl.col.n +
                                      pl.int_range(pl.len()).lt(abs(diff))
                                      .cast(pl.Int32) * pl.lit(diff).sign())
                selected = by_frame\
                    .join(group_counts, on=by_name)\
                    .select(pl.int_range(pl.len())
                            .shuffle(seed=seed)
                            .over(by_name)
                            .lt(n))
            else:
                # noinspection PyUnresolvedReferences
                selected = by_frame\
                    .shuffle(seed=seed)\
                    .over(by_name)\
                    .lt((fraction * pl.len().over(by_name)).round())
        else:
            selected = self._var\
                .select(pl.int_range(pl.len())
                        .shuffle(seed=seed)
                        .lt(n if fraction is None else
                            (fraction * pl.len()).round()))
        selected = selected.to_series()
        return self.filter_var(selected) if subsample_column is None else \
            self.with_columns_var(selected.alias(subsample_column))
    
    def pipe(self,
             function: Callable[[SingleCell, ...], Any],
             *args: Any,
             **kwargs: Any) -> Any:
        """
        Apply a function to a SingleCell dataset.
        
        Args:
            function: the function to apply
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            function(self, *args, **kwargs)
        """
        return function(self, *args, **kwargs)
    
    def pipe_X(self,
               function: Callable[[csr_array | csc_array, ...],
                                  csr_array | csc_array],
               *args: Any,
               **kwargs: Any) -> SingleCell:
        """
        Apply a function to a SingleCell dataset's X.
        
        Args:
            function: the function to apply to X
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new SingleCell dataset where the function has been applied to X.
        """
        return SingleCell(X=function(self._X, *args, **kwargs), obs=self._obs,
                          var=self._var, obsm=self._obsm, varm=self._varm,
                          uns=self._uns)
    
    def pipe_obs(self,
                 function: Callable[[pl.DataFrame, ...], pl.DataFrame],
                 *args: Any,
                 **kwargs: Any) -> SingleCell:
        """
        Apply a function to a SingleCell dataset's obs.
        
        Args:
            function: the function to apply to obs
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new SingleCell dataset where the function has been applied to
            obs.
        """
        return SingleCell(X=self._X, obs=function(self._obs, *args, **kwargs),
                          var=self._var, obsm=self._obsm, varm=self._varm,
                          uns=self._uns)
    
    def pipe_var(self,
                 function: Callable[[pl.DataFrame, ...], pl.DataFrame],
                 *args: Any,
                 **kwargs: Any) -> SingleCell:
        """
        Apply a function to a SingleCell dataset's var.
        
        Args:
            function: the function to apply to var
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new SingleCell dataset where the function has been applied to
            var.
        """
        return SingleCell(X=self._X, obs=self._obs,
                          var=function(self._var, *args, **kwargs),
                          obsm=self._obsm, varm=self._varm, uns=self._uns)
    
    def pipe_obsm(self,
                  function: Callable[[dict[str, np.ndarray[2, Any]], ...],
                                     dict[str, np.ndarray[2, Any]]],
                  *args: Any,
                  **kwargs: Any) -> SingleCell:
        """
        Apply a function to a SingleCell dataset's obsm.
        
        Args:
            function: the function to apply to obsm
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new SingleCell dataset where the function has been applied to
            obsm.
        """
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm=function(self._obsm, *args, **kwargs),
                          varm=self._varm, uns=self._uns)
    
    def pipe_varm(self,
                  function: Callable[[dict[str, np.ndarray[2, Any]], ...],
                                     dict[str, np.ndarray[2, Any]]],
                  *args: Any,
                  **kwargs: Any) -> SingleCell:
        """
        Apply a function to a SingleCell dataset's varm.
        
        Args:
            function: the function to apply to varm
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new SingleCell dataset where the function has been applied to
            varm.
        """
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm=self._obsm,
                          varm=function(self._varm, *args, **kwargs),
                          uns=self._uns)
    
    def pipe_uns(self,
                 function: Callable[[dict[str, np.ndarray[2, Any]], ...],
                                    dict[str, np.ndarray[2, Any]]],
                 *args: Any,
                 **kwargs: Any) -> SingleCell:
        """
        Apply a function to a SingleCell dataset's uns.
        
        Args:
            function: the function to apply to uns
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new SingleCell dataset where the function has been applied to
            uns.
        """
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm=self._obsm, varm=self._varm,
                          uns=function(self._uns, *args, **kwargs))
    
    def qc(self,
           custom_filter: SingleCellColumn | None = None,
           *,
           subset: bool = False,
           QC_column: str = 'passed_QC',
           max_mito_fraction: int | float | np.integer | np.floating |
                              None = 0.1,
           min_genes: int | np.integer | None = 100,
           MALAT1_filter: bool = True,
           allow_float: bool = False,
           num_threads: int | np.integer | None = 1,
           verbose: bool = True) -> SingleCell:
        """
        Adds a Boolean column to obs indicating which cells passed quality
        control (QC), or subsets to these cells if `subset=True`. By default,
        filters to non-doublet cells with a cell-type confidence of ≥90%, ≤10%
        mitochondrial reads, and ≥100 genes detected. Raises an error if any
        gene names appear more than once in var_names (they can be deduplicated
        with `deduplicate_var_names()`).
        
        Args:
            custom_filter: an optional Boolean column of obs containing a
                           filter to apply on top of the other QC filters; True
                           elements will be kept. Can be a column name, a
                           polars expression, a polars Series, a 1D NumPy
                           array, or a function that takes in this SingleCell
                           dataset and returns a polars Series or 1D NumPy
                           array.
            subset: whether to subset to cells passing QC, instead of merely
                    adding a `QC_column` to obs. This will roughly double
                    memory usage, but speed up subsequent operations.
            QC_column: the name of a Boolean column to add to obs indicating
                       which cells passed QC, if `subset=False`; gives an error
                       if obs already has a column with this name
            max_mito_fraction: if not None, filter to cells with <= this
                               fraction of mitochondrial counts. The default
                               value of 10% is in between Seurat's recommended
                               value of 5%, and Scanpy's recommendation to not
                               filter on mitochondrial counts at all, at least
                               initially.
            min_genes: if not None, filter to cells with >= this many genes
                       detected (with nonzero count). The default of 100
                       matches Scanpy's recommended value, while Seurat
                       recommends a minimum of 200.
            MALAT1_filter: if True, filter out cells with 0 expression of the
                           nuclear-expressed lncRNA MALAT1, which likely
                           represent empty droplets or poor-quality cells
                           (biorxiv.org/content/10.1101/2024.07.14.603469v1)
            allow_float: if False, raise an error if `X.dtype` is
                         floating-point (suggesting the user may not be using
                         the raw counts); if True, disable this sanity check
            num_threads: the number of threads to use when filtering based on
                         mitochondrial counts (the bottleneck of this
                         function). Set `num_threads=None` to use all available
                         cores (as determined by `os.cpu_count()`).
                         `num_threads` must be 1 (the default) when
                         `max_mito_fraction` is None and `MALAT1_filter` is
                         False, since `num_threads` is only used for the
                         mitochondrial count and MALAT1 filters.
            verbose: whether to print how many cells were filtered out at each
                     step of the QC process
        
        Returns:
            A new SingleCell dataset with `QC_column` added to obs, or subset
            to QCed cells if `subset=True`, and `uns['QCed']` set to True.
        """
        X = self._X
        # Check that `self` is not already QCed
        if self._uns['QCed']:
            error_message = (
                "uns['QCed'] is True; did you already run qc()? Set "
                "uns['QCed'] = False or run with_uns(QCed=False) to bypass "
                "this check.")
            raise ValueError(error_message)
        # Check inputs
        if self.var_names.n_unique() < len(self.var):
            error_message = (
                'var_names contains duplicates; deduplicate with '
                'deduplicate_var_names()')
            raise ValueError(error_message)
        if custom_filter is not None:
            custom_filter = self._get_column(
                'obs', custom_filter, 'custom_filter', pl.Boolean)
        check_type(subset, 'subset', bool, 'Boolean')
        if not subset:
            check_type(QC_column, 'QC_column', str, 'a string')
            if QC_column in self._obs:
                error_message = (
                    f'QC_column {QC_column!r} is already a column of obs; did '
                    f'you already run qc()?')
                raise ValueError(error_message)
        if max_mito_fraction is not None:
            check_type(max_mito_fraction, 'max_mito_fraction', (int, float),
                       'a number between 0 and 1, inclusive')
            check_bounds(max_mito_fraction, 'max_mito_fraction', 0, 1)
        if min_genes is not None:
            check_type(min_genes, 'min_genes', int, 'a non-negative integer')
            check_bounds(min_genes, 'min_genes', 0)
        check_type(MALAT1_filter, 'MALAT1_filter', bool, 'Boolean')
        check_type(allow_float, 'allow_float', bool, 'Boolean')
        if num_threads is None:
            num_threads = os.cpu_count()
        else:
            check_type(num_threads, 'num_threads', int, 'a positive integer')
            check_bounds(num_threads, 'num_threads', 1)
        if num_threads != 1 and max_mito_fraction is None:
            error_message = (
                'num_threads must be 1 (the default) when max_mito_fraction '
                'is None, since num_threads is only used for the '
                'mitochondrial count filter')
            raise ValueError(error_message)
        check_type(verbose, 'verbose', bool, 'Boolean')
        # If `allow_float` is False, raise an error if `X` is floating-point
        if not allow_float and np.issubdtype(X.dtype, np.floating):
            error_message = (
                f'qc() requires raw counts but X.dtype is {X.dtype!r}, '
                f'a floating-point data type. If you are sure that all values '
                f'are raw integer counts, i.e. that (X.data == '
                f'X.data.astype(int)).all(), then set allow_float=True.')
            raise TypeError(error_message)
        # Apply the custom filter, if specified
        if verbose:
            print(f'Starting with {len(self):,} cells.')
        mask = None
        if custom_filter is not None:
            if verbose:
                print('Applying the custom filter...')
            mask = custom_filter
            if verbose:
                print(f'{mask.sum():,} cells remain after applying the custom '
                      f'filter.')
        # Filter to cells with ≤ `100 * max_mito_fraction`% mitochondrial
        # counts, if max_mito_fraction was specified
        if max_mito_fraction is not None:
            if verbose:
                print(f'Filtering to cells with ≤{100 * max_mito_fraction}% '
                      f'mitochondrial counts...')
            var_names = self.var_names
            if var_names.dtype != pl.String:
                var_names = var_names.cast(pl.String)
            mt_genes = var_names.str.to_uppercase().str.starts_with('MT-')
            if not mt_genes.any():
                error_message = (
                    'no genes are mitochondrial (start with "MT-" or "mt-"); '
                    'this may happen if your var_names are Ensembl IDs (ENSG) '
                    'rather than gene symbols (in which case you should set '
                    'the gene symbols as the var_names with set_var_names()), '
                    'or if mitochondrial genes have already been filtered out '
                    '(in which case you can set max_mito_fraction to None)')
                raise ValueError(error_message)
            mito_mask = np.empty(X.shape[0], dtype=bool)
            prange_import = 'from cython.parallel cimport prange' \
                if num_threads > 1 else ''
            sum_variable_type = X.dtype
            if sum_variable_type == np.float32:
                sum_variable_type = float
            if isinstance(X, csr_array):
                cython_inline(rf'''
                    {prange_import}
                    def mito_mask(
                            const {cython_type(X.dtype)}[::1] data,
                            const {cython_type(X.indices.dtype)}[::1] indices,
                            const {cython_type(X.indptr.dtype)}[::1] indptr,
                            char[::1] mt_genes,
                            const double max_mito_fraction,
                            char[::1] mito_mask,
                            const unsigned num_threads):
                        cdef long row, col
                        cdef {cython_type(sum_variable_type)} row_sum, mt_sum
                        for row in \
                                {prange('indptr.shape[0] - 1', num_threads)}:
                            row_sum = mt_sum = 0
                            for col in range(indptr[row], indptr[row + 1]):
                                row_sum = row_sum + data[col]
                                if mt_genes[indices[col]]:
                                    mt_sum = mt_sum + data[col]
                            mito_mask[row] = (<double> mt_sum / row_sum) <= \
                                             max_mito_fraction
                        ''')['mito_mask'](
                            data=X.data, indices=X.indices, indptr=X.indptr,
                            mt_genes=mt_genes.to_numpy(),
                            max_mito_fraction=max_mito_fraction,
                            mito_mask=mito_mask, num_threads=num_threads)
            else:
                row_sums = np.zeros(X.shape[0], dtype=sum_variable_type)
                mt_sums = np.zeros(X.shape[0], dtype=sum_variable_type)
                cython_inline(rf'''
                    {prange_import}
                    def mito_mask(
                            const {cython_type(X.dtype)}[::1] data,
                            const {cython_type(X.indices.dtype)}[::1] indices,
                            const {cython_type(X.indptr.dtype)}[::1] indptr,
                            char[::1] mt_genes,
                            const double max_mito_fraction,
                            {cython_type(sum_variable_type)}[::1] row_sums,
                            {cython_type(sum_variable_type)}[::1] mt_sums,
                            char[::1] mito_mask,
                            const unsigned num_threads):
                        cdef long row, col, i
                        for col in \
                                {prange('indptr.shape[0] - 1', num_threads)}:
                            if mt_genes[col]:
                                for i in range(indptr[col], indptr[col + 1]):
                                    row_sums[indices[i]] += data[i]
                                    mt_sums[indices[i]] += data[i]
                            else:
                                for i in range(indptr[col], indptr[col + 1]):
                                    row_sums[indices[i]] += data[i]
                        for row in {prange('mito_mask.shape[0]', num_threads)}:
                            mito_mask[row] = \
                                (<double> mt_sums[row] / row_sums[row]) <= \
                                max_mito_fraction
                        ''')['mito_mask'](
                            data=X.data, indices=X.indices, indptr=X.indptr,
                            mt_genes=mt_genes.to_numpy(),
                            max_mito_fraction=max_mito_fraction,
                            row_sums=row_sums, mt_sums=mt_sums,
                            mito_mask=mito_mask, num_threads=num_threads)
            mito_mask = pl.Series(mito_mask)
            if not mito_mask.any():
                error_message = (
                    f'no cells remain after filtering to cells with '
                    f'≤{100 * max_mito_fraction}% mitochondrial counts')
                raise ValueError(error_message)
            if mask is None:
                mask = mito_mask
            else:
                mask &= mito_mask
            if verbose:
                print(f'{mask.sum():,} cells remain after filtering to cells '
                      f'with ≤{100 * max_mito_fraction}% mitochondrial '
                      f'counts.')
        # Filter to cells with ≥ `min_genes` genes detected, if specified
        if min_genes is not None:
            if verbose:
                print(f'Filtering to cells with ≥{min_genes:,} genes '
                      f'detected (with nonzero count)...')
            gene_mask = pl.Series(getnnz(X, axis=1) >= min_genes)
            if not gene_mask.any():
                error_message = (
                    f'no cells remain after filtering to cells with '
                    f'≥{min_genes:,} genes detected')
                raise ValueError(error_message)
            if mask is None:
                mask = gene_mask
            else:
                mask &= gene_mask
            if verbose:
                print(f'{mask.sum():,} cells remain after filtering to cells '
                      f'with ≥{min_genes:,} genes detected.')
        # Filter to cells with non-zero MALAT1 expression, if `MALAT1_filter`
        # is True
        if MALAT1_filter:
            if verbose:
                print(f'Filtering to cells with non-zero MALAT1 expression...')
            MALAT1_index = self._var\
                .select(pl.arg_where(pl.col(self.var_names.name) == 'MALAT1'))
            if len(MALAT1_index) == 0:
                error_message = (
                    f"'MALAT1' was not found in var_names; this may happen if "
                    f"your var_names are Ensembl IDs (ENSG) rather than gene "
                    f"symbols (in which case you should set the gene symbols "
                    f"as the var_names with set_var_names()). Alternatively, "
                    f"set MALAT1_filter=False to disable filtering on MALAT1 "
                    f"expression.")
                raise ValueError(error_message)
            MALAT1_index = MALAT1_index.item()
            # the code below is a faster version of:
            # MALAT1_mask = (X[:, [MALAT1_index]] != 0).toarray().squeeze()
            MALAT1_mask = np.zeros(X.shape[0], dtype=bool)
            if isinstance(X, csr_array):
                prange_import = 'from cython.parallel cimport prange' \
                    if num_threads > 1 else ''
                # don't use binary search because CSR indices may not be sorted
                cython_inline(rf'''
                    {prange_import}
                    def get_MALAT1_mask_CSR(
                            const {cython_type(X.dtype)}[::1] data,
                            const {cython_type(X.indices.dtype)}[::1] indices,
                            const {cython_type(X.indptr.dtype)}[::1] indptr,
                            const int MALAT1_index,
                            char[::1] MALAT1_mask,
                            const unsigned num_threads):
                        cdef long row, col
                        for row in \
                                {prange('indptr.shape[0] - 1', num_threads)}:
                            for col in range(indptr[row], indptr[row + 1]):
                                if indices[col] == MALAT1_index:
                                    MALAT1_mask[row] = True
                                    break
                ''')['get_MALAT1_mask_CSR'](
                    data=X.data, indices=X.indices, indptr=X.indptr,
                    MALAT1_index=MALAT1_index, MALAT1_mask=MALAT1_mask,
                    num_threads=num_threads)
            else:
                start = X.indptr[MALAT1_index]
                end = X.indptr[MALAT1_index + 1]
                MALAT1_mask[X.indices[start:end]] = True
            MALAT1_mask = pl.Series(MALAT1_mask)
            if mask is None:
                mask = MALAT1_mask
            else:
                mask &= MALAT1_mask
            if verbose:
                print(f'{mask.sum():,} cells remain after filtering to cells '
                      f'with non-zero MALAT1 expression.')
        # Add the mask of QCed cells as a column, or subset if `subset=True`
        if mask is None:
            error_message = 'no QC filters were specified'
            raise ValueError(error_message)
        if subset:
            if verbose:
                print(f'Subsetting to cells passing QC...')
            sc = self.filter_obs(mask)
        else:
            if verbose:
                print(f'Adding a Boolean column, obs[{QC_column!r}], '
                      f'indicating which cells passed QC...')
            sc = SingleCell(X=X, obs=self._obs.with_columns(
                pl.lit(mask).alias(QC_column)), var=self._var, obsm=self._obsm,
                              varm=self._varm, uns=self._uns)
        sc._uns['QCed'] = True
        return sc
    
    def deduplicate_obs_names(self, join: str = '-') -> SingleCell:
        """
        Make obs names unique by appending `'-1'` to the second occurence of
        a given obs name, `'-2'` to the third occurrence, and so on, where
        `'-'` can be switched to a different string via the `join` argument.
        Raises an error if any obs_names already contain the `join` string.
        
        Args:
            join: the string connecting the original obs name and the integer
                  suffix

        Returns:
            A new SingleCell dataset with the obs names made unique.
        """
        check_type(join, 'join', str, 'a string')
        if self.obs_names.str.contains(join).any():
            error_message = (
                f'some obs_names already contain the join string {join!r}; '
                f'did you already run deduplicate_obs_names()? If not, set '
                f'the join argument to a different string.')
            raise ValueError(error_message)
        obs_names = pl.col(self.obs_names.name)
        num_times_seen = pl.int_range(pl.len()).over(obs_names)
        return SingleCell(X=self._X,
                          obs=self._obs.with_columns(
                              pl.when(num_times_seen > 0)
                              .then(obs_names + join +
                                    num_times_seen.cast(str))
                              .otherwise(obs_names)),
                          var=self._var, obsm=self._obsm, varm=self._varm,
                          uns=self._uns)
    
    def deduplicate_var_names(self, join: str = '-') -> SingleCell:
        """
        Make var names unique by appending `'-1'` to the second occurence of
        a given var name, `'-2'` to the third occurrence, and so on, where
        `'-'` can be switched to a different string via the `join` argument.
        Raises an error if any var_names already contain the `join` string.
        
        Args:
            join: the string connecting the original var name and the integer
                  suffix

        Returns:
            A new SingleCell dataset with the var names made unique.
        """
        var_names = pl.col(self.var_names.name)
        num_times_seen = pl.int_range(pl.len()).over(var_names)
        return SingleCell(X=self._X,
                          obs=self._obs,
                          var=self._var.with_columns(
                              pl.when(num_times_seen > 0)
                              .then(var_names + join +
                                    num_times_seen.cast(str))
                              .otherwise(var_names)),
                          obsm=self._obsm, varm=self._varm, uns=self._uns)
    
    def get_sample_covariates(self,
                              ID_column: SingleCellColumn,
                              *,
                              QC_column: SingleCellColumn |
                                         None = 'passed_QC') -> pl.DataFrame:
        """
        Get a DataFrame of sample-level covariates, i.e. the columns of obs
        that are the same for all cells within each sample.
        
        Args:
            ID_column: a column of obs containing sample IDs. Can be a column
                       name, a polars expression, a polars Series, a 1D NumPy
                       array, or a function that takes in this SingleCell
                       dataset and returns a polars Series or 1D NumPy array.
            QC_column: an optional Boolean column of obs indicating which cells
                       passed QC. Can be a column name, a polars expression, a
                       polars Series, a 1D NumPy array, or a function that
                       takes in this SingleCell dataset and returns a polars
                       Series or 1D NumPy array. Set to None to include all
                       cells. Cells failing QC will be ignored.
        
        Returns:
            A DataFrame of the sample-level covariates, with ID_column (sorted)
            as the first column.
        """
        ID_column = self._get_column('obs', ID_column, 'ID_column',
                                     (pl.String, pl.Categorical, pl.Enum,
                                      'integer'))
        if QC_column is not None:
            QC_column = self._get_column(
                'obs', QC_column, 'QC_column', pl.Boolean,
                allow_missing=QC_column == 'passed_QC')
        obs = self._obs
        if QC_column is not None:
            obs = obs.filter(QC_column)
            ID_column = ID_column.filter(QC_column)
        return obs\
            .select(ID_column,
                    *obs
                    .group_by(ID_column)
                    .n_unique()
                    .pipe(filter_columns,
                          (pl.exclude(ID_column.name)
                           if ID_column.name in obs else pl.all()).max().eq(1))
                    .columns)\
            .unique(ID_column.name)\
            .sort(ID_column.name)
    
    def pseudobulk(self,
                   ID_column: SingleCellColumn,
                   cell_type_column: SingleCellColumn,
                   *,
                   QC_column: SingleCellColumn | None = 'passed_QC',
                   num_threads: int | np.integer | None = 1,
                   additional_obs: pl.DataFrame | None = None,
                   sort_genes: bool = True,
                   verbose: bool = True) -> Pseudobulk:
        """
        Pseudobulks a single-cell dataset with sample ID and cell type columns,
        after filtering to cells passing QC according to `QC_column`. Returns a
        Pseudobulk dataset.
        
        You can run this function multiple times at different cell type
        resolutions by setting a different cell_type_column each time.
        
        Args:
            ID_column: a column of obs containing sample IDs. Can be a column
                       name, a polars expression, a polars Series, a 1D NumPy
                       array, or a function that takes in this SingleCell
                       dataset and returns a polars Series or 1D NumPy array.
            cell_type_column: a column of obs containing cell-type labels. Can
                              be a column name, a polars expression, a polars
                              Series, a 1D NumPy array, or a function that
                              takes in this SingleCell dataset and returns a
                              polars Series or 1D NumPy array.
            QC_column: an optional Boolean column of obs indicating which cells
                       passed QC. Can be a column name, a polars expression, a
                       polars Series, a 1D NumPy array, or a function that
                       takes in this SingleCell dataset and returns a polars
                       Series or 1D NumPy array. Set to None to include all
                       cells. Cells failing QC will be excluded from the
                       pseudobulk. 
            num_threads: the number of threads to use when pseudobulking;
                         parallelism happens across {sample, cell type} pairs
                         (or just samples, if `cell_type_column` is None).
                         Set `num_threads=None` to use all available cores
                         (as determined by `os.cpu_count()`).
            additional_obs: an optional DataFrame of additional sample-level
                            covariates, which will be joined to the
                            pseudobulk's obs for each cell type
            sort_genes: whether to sort genes in alphabetical order in the
                        pseudobulk
            verbose: whether to print additional detail about the pseudobulking
                     process
    
        Returns:
            A Pseudobulk dataset with X (counts), obs (metadata per sample),
            and var (metadata per gene) fields, each dicts across cell types.
            The columns of each cell type's obs will be:
            - 'ID' (a renamed version of `ID_column`)
            - 'num_cells' (the number of cells for that sample and cell type)
            followed by whichever columns of the SingleCell dataset's obs are
            constant across samples.
        """
        X = self._X
        # Check that `self` is QCed and not normalized
        if not self._uns['QCed']:
            error_message = (
                "uns['QCed'] is False; did you forget to run qc()? Set "
                "uns['QCed'] = True or run .with_uns(QCed=True) to bypass "
                "this check.")
            raise ValueError(error_message)
        if self._uns['normalized']:
            error_message = (
                "uns['normalized'] is True; did you already run normalize()?")
            raise ValueError(error_message)
        # Check inputs
        ID_column = self._get_column('obs', ID_column, 'ID_column',
                                     (pl.String, pl.Categorical, pl.Enum,
                                      'integer'))
        cell_type_column = \
            self._get_column('obs', cell_type_column, 'cell_type_column',
                             (pl.String, pl.Categorical, pl.Enum))
        if QC_column is not None:
            QC_column = self._get_column(
                'obs', QC_column, 'QC_column', pl.Boolean,
                allow_missing=QC_column == 'passed_QC')
        if num_threads is None:
            num_threads = os.cpu_count()
        else:
            check_type(num_threads, 'num_threads', int, 'a positive integer')
            check_bounds(num_threads, 'num_threads', 1)
        if additional_obs is not None:
            check_type(additional_obs, 'additional_obs', pl.DataFrame,
                       'a polars DataFrame')
            if ID_column.name not in additional_obs:
                error_message = (
                    f'ID_column {ID_column.name!r} is not a column of '
                    f'additional_obs')
                raise ValueError(error_message)
            if ID_column.dtype != additional_obs[ID_column.name].dtype:
                error_message = (
                    f'ID_column {ID_column.name!r} has a different data type '
                    f'in additional_obs than in this SingleCell dataset')
                raise TypeError(error_message)
        check_type(sort_genes, 'sort_genes', bool, 'Boolean')
        check_type(verbose, 'verbose', bool, 'Boolean')
        # Ensure the first column of var is Categorical, Enum, or String: this
        # is a requirement of the Pseudobulk class. (The first column of obs
        # must be as well, but this will always be true by construction, since
        # it will always be the sample ID.)
        if self.var_names.dtype not in (pl.Categorical, pl.Enum, pl.String):
            error_message = (
                f'the first column of var (var_names) has data type '
                f'{self.obs_names.dtype!r}, but must be Categorical, Enum, '
                f'or String')
            raise ValueError(error_message)
        # Get the row indices in sc.X that will be pseudobulked across for each
        # group (cell type-sample pair)
        groups = (pl.DataFrame([cell_type_column, ID_column])
                  .with_row_index('_SingleCell_index')
                  if QC_column is None else
                  pl.DataFrame([cell_type_column, ID_column, QC_column])
                  .with_row_index('_SingleCell_index')
                  .filter(QC_column.name))\
            .group_by(cell_type_column.name, ID_column.name,
                      maintain_order=True)\
            .agg(num_cells=pl.len(), group_indices='_SingleCell_index')\
            .sort(cell_type_column.name, ID_column.name)
        # Pseudobulk, storing the result in a preallocated NumPy array
        result = np.zeros((len(groups), X.shape[1]), dtype='int32', order='F')
        prange_import = \
            'from cython.parallel cimport prange' if num_threads > 1 else ''
        if isinstance(X, csr_array):
            group_indices = groups['group_indices'].explode().to_numpy()
            group_ends = groups['num_cells'].cum_sum().to_numpy()
            cython_inline(rf'''
                {prange_import}
                def csr_groupby_sum(
                        const {cython_type(X.dtype)}[::1] data,
                        const {cython_type(X.indices.dtype)}[::1] indices,
                        const {cython_type(X.indptr.dtype)}[::1] indptr,
                        const unsigned[::1] group_indices,
                        const unsigned[::1] group_ends,
                        int[::1, :] result,
                        const unsigned num_threads):
                    cdef unsigned num_groups = group_ends.shape[0]
                    cdef unsigned group, group_element, row
                    cdef {cython_type(X.indptr.dtype)} column
                    # For each group (samples, or cell type-sample pairs)...
                    for group in {prange('num_groups', num_threads)}:
                        # For each element (cell) within this group...
                        for group_element in range(
                                0 if group == 0 else group_ends[group - 1],
                                group_ends[group]):
                            # Get this element's row index in the sparse matrix
                            row = group_indices[group_element]
                            # For each column (gene) that's nonzero for this
                            # row...
                            for column in range(indptr[row], indptr[row + 1]):
                                # Add the value at this row and column to the
                                # total for this group and column
                                result[group, indices[column]] += \
                                    int(data[column])
                ''')['csr_groupby_sum'](
                    data=X.data, indices=X.indices, indptr=X.indptr,
                    group_indices=group_indices, group_ends=group_ends,
                    result=result, num_threads=num_threads)
        else:
            if verbose:
                print('Warning: X is a csc_array rather than a csr_array, '
                      'so pseudobulking may be thousands of times slower. '
                      'If you have enough memory, call .tocsr() on your '
                      'SingleCell dataset before pseudobulking.')
            group_indices = pl.int_range(X.shape[0], eager=True,
                                         dtype=pl.UInt32)\
                .to_frame('group_indices')\
                .join(groups
                      .with_row_index()
                      .select('index', 'group_indices')
                      .explode('group_indices'),
                      on='group_indices', how='left')\
                ['index']\
                .fill_null(-1)\
                .to_numpy()
            cython_inline(f'''
                {prange_import}
                def csc_groupby_sum(
                        const {cython_type(X.dtype)}[::1] data,
                        const {cython_type(X.indices.dtype)}[::1] indices,
                        const {cython_type(X.indptr.dtype)}[::1] indptr,
                        const long[::1] group_indices,
                        const unsigned num_columns,
                        int[::1, :] result,
                        const unsigned num_threads):
                    cdef unsigned row
                    cdef {cython_type(X.indptr.dtype)} column
                    cdef long group
                    # For each column of the sparse matrix (gene)...
                    for column in {prange('num_columns', num_threads)}:
                        # For each row (cell) that's non-zero for this
                        # column...
                        for row in range(indptr[column], indptr[column + 1]):
                            # Get the group index for this row (-1 if it failed
                            # QC)
                            group = group_indices[indices[row]]
                            if group == -1:
                                continue
                            # Add the value at this row and column to the total
                            # for this group and column
                            result[group, column] += <int> data[row]
                    ''')['csc_groupby_sum'](
                        data=X.data, indices=X.indices, indptr=X.indptr,
                        group_indices=group_indices, num_columns=X.shape[1],
                        result=result, num_threads=num_threads)
        # Sort genes
        if sort_genes:
            result = result[:, self.var_names.arg_sort()]
        # Break up the results by cell type (this could be done faster if X
        # were split by cell type in Cython, but it's already quite fast, e.g.
        # ~1s for ~100 samples)
        sample_covariates = self.get_sample_covariates(ID_column,
                                                       QC_column=QC_column)
        X, obs, var = {}, {}, {}
        for cell_type in groups[cell_type_column.name].unique():
            cell_type_mask = groups[cell_type_column.name] == cell_type
            # noinspection PyUnresolvedReferences
            X[cell_type] = result[cell_type_mask.to_numpy()]
            obs[cell_type] = groups.lazy()\
                .filter(pl.col(cell_type_column.name) == cell_type)\
                .select(ID_column.name, 'num_cells')\
                .join(sample_covariates.lazy(), on=ID_column.name, how='left')\
                .pipe(lambda df: df.join(additional_obs.lazy(),
                                         on=ID_column.name, how='left')
                      if additional_obs is not None else df)\
                .rename({ID_column.name: 'ID'})\
                .pipe(lambda df: df if QC_column is None else
                                 df.drop(QC_column.name))\
                .collect()
            var[cell_type] = self._var
        return Pseudobulk(X=X, obs=obs, var=var)
    
    def hvg(self,
            *others: SingleCell,
            QC_column: SingleCellColumn | None |
                       Sequence[SingleCellColumn | None] = 'passed_QC',
            batch_column: SingleCellColumn | None |
                          Sequence[SingleCellColumn | None] = None,
            num_genes: int | np.integer = 2000,
            min_cells: int | np.integer | None = 3,
            flavor: Literal['seurat_v3', 'seurat_v3_paper'] = 'seurat_v3',
            span: int | float | np.integer | np.floating = 0.3,
            hvg_column: str = 'highly_variable',
            rank_column: str = 'highly_variable_rank',
            allow_float: bool = False,
            num_threads: int | np.integer | None = 1) -> \
            SingleCell | tuple[SingleCell, ...]:
        """
        Select highly variable genes using Seurat's algorithm. Operates on
        raw counts.
        
        By default, exactly matches Scanpy's `scanpy.pp.highly_variable_genes`
        function with the `flavor` argument set to the non-default value
        `'seurat_v3'`. It also exactly matches Seurat's `FindVariableFeatures`
        function with the `selection.method` argument set to the default value
        `'vst'`.

        Requires the scikit-misc package; install with:
        pip install --no-deps --no-build-isolation scikit-misc
        
        The general idea is that since genes with higher mean expression tend
        to have higher variance in expression (because they have more non-zero
        values), we want to select genes that have a high variance *relative to
        their mean expression*. Otherwise, we'd only be picking highly
        expressed genes! To correct for the mean-variance relationship, fit a
        LOESS curve fit to the mean-variance trend.
        
        Args:
            others: optional SingleCell datasets to jointly compute highly
                    variable genes across, alongside this one. Each dataset
                    will be treated as a separate batch. If `batch_column` is
                    not None, each dataset AND each distinct value of
                    `batch_column` within each dataset will be treated as a
                    separate batch. Variances will be computed per batch and
                    then aggregated (see `flavor`) across batches.
            QC_column: an optional Boolean column of obs indicating which cells
                       passed QC. Can be a column name, a polars expression, a 
                       polars Series, a 1D NumPy array, or a function that 
                       takes in this SingleCell dataset and returns a polars 
                       Series or 1D NumPy array. Set to None to include all
                       cells. Cells failing QC will be ignored. When `others`
                       is specified, `QC_column` can be a
                       length-`1 + len(others)` sequence of columns,
                       expressions, Series, functions, or None for each
                       dataset (for `self`, followed by each dataset in
                       `others`).
            batch_column: an optional String, Categorical, Enum, or integer
                          column of obs indicating which batch each cell is
                          from. Can be a column name, a polars expression, a 
                          polars Series, a 1D NumPy array, or a function that 
                          takes in this SingleCell dataset and returns a polars
                          Series or 1D NumPy array. Each batch will be treated
                          as if it were a distinct dataset; this is exactly
                          equivalent to splitting the dataset with
                          `split_by(batch_column)` and then passing each of the
                          resulting datasets to `hvg()`, except that the
                          `min_cells` filter will always be calculated
                          per-dataset rather than per-batch. Variances will be
                          computed per batch and then aggregated (see `flavor`)
                          across batches. Set to None to treat each dataset as
                          having a single batch. When `others` is specified,
                          `batch_column` can be a length-`1 + len(others)`
                          sequence of columns, expressions, Series, functions,
                          or None for each dataset (for `self`, followed by
                          each dataset in `others`).
            num_genes: the number of highly variable genes to select. The
                       default of 2000 matches Seurat and Scanpy's recommended
                       value. Fewer than `num_genes` genes will be selected if
                       not enough genes have nonzero count in >= `min_cells`
                       cells (or when `min_cells` is None, if not enough genes
                       are present).
            min_cells: if not None, filter to genes detected (with nonzero
                       count) in >= this many cells in every dataset, before
                       calculating highly variable genes. The default value of
                       3 matches Seurat and Scanpy's recommended value.
            flavor: the highly variable gene algorithm to use. Must be one of
                    `seurat_v3` and `seurat_v3_paper`, both of which match the
                    algorithms with the same name in scanpy. Both algorithms
                    select genes based on two criteria: 1) which genes are
                    ranked as most variable (taking the median of the ranks
                    across batches) and 2) the number of batches in which a
                    gene is ranked in among the top `num_genes` in variability.
                    `seurat_v3` ranks genes by 1) and uses 2) to tiebreak,
                    whereas `seurat_v3_paper` ranks genes by 2) and uses 1) to
                    tiebreak. When there is only one batch, both algorithms are
                    the same and only rank based on 1).
            span: the span of the LOESS fit; higher values will lead to more
                  smoothing
            hvg_column: the name of a Boolean column to be added to (each
                        dataset's) var indicating the highly variable genes
            rank_column: the name of an integer column to be added to (each
                         dataset's) var with the rank of each highly variable
                         gene's variance (1 = highest variance, 2 =
                         next-highest, etc.); will be null for non-highly
                         variable genes. In the very unlikely event of ties,
                         the gene that appears first in var will get the lowest
                         rank.
            allow_float: if False, raise an error if `X.dtype` is
                         floating-point (suggesting the user may not be using
                         the raw counts); if True, disable this sanity check
            num_threads: the number of threads to use when finding highly
                         variable genes. Set `num_threads=None` to use all
                         available cores (as determined by `os.cpu_count()`).
        
        Returns:
            A new SingleCell dataset where var contains an additional Boolean
            column, `hvg_column` (default: 'highly_variable'), indicating the
            `num_genes` most highly variable genes, and `rank_column` (default:
            'highly_variable_rank') indicating the rank of each highly variable
            gene's variance. Or, if additional SingleCell dataset(s) are
            specified via the `others` argument, a length-`1 + len(others)`
            tuple of SingleCell datasets with these two columns added: `self`,
            followed by each dataset in `others`.
        
        Note:
            This function may give an incorrect output if the count matrix
            contains explicit zeros (i.e. if `(X.data == 0).any()`): this is
            not checked for, due to speed considerations. In the unlikely event
            that your dataset contains explicit zeros, remove them by running
            `X.eliminate_zeros()` (an in-place operation).
        """
        # noinspection PyUnresolvedReferences
        from skmisc.loess import loess
        X = self._X
        # Check that all elements of `others` are SingleCell datasets
        if others:
            check_types(others, 'others', SingleCell, 'SingleCell datasets')
        datasets = [self] + list(others)
        # Check that all datasets are QCed and not normalized
        if not all(dataset._uns['QCed'] for dataset in datasets):
            suffix = ' for at least one dataset' if others else ''
            error_message = (
                f"uns['QCed'] is False{suffix}; did you forget to run qc()? "
                f"Set uns['QCed'] = True or run with_uns(QCed=True) to bypass "
                f"this check.")
            raise ValueError(error_message)
        if any(dataset._uns['normalized'] for dataset in datasets):
            suffix = ' for at least one dataset' if others else ''
            error_message = (
                f"hvg() requires raw counts but uns['normalized'] is "
                f"True{suffix}; did you already run normalize()?")
            raise ValueError(error_message)
        # Get `QC_column` and `batch_column` from every dataset, if not None
        QC_columns = SingleCell._get_columns(
            'obs', datasets, QC_column, 'QC_column', pl.Boolean,
            allow_missing=True)
        batch_columns = SingleCell._get_columns(
            'obs', datasets, batch_column, 'batch_column',
            (pl.String, pl.Categorical, pl.Enum, 'integer'))
        # Check that `num_genes` and (if not None) `min_cells` are positive
        # integers
        check_type(num_genes, 'num_genes', int, 'a positive integer')
        check_bounds(num_genes, 'num_genes', 1)
        if min_cells is not None:
            check_type(min_cells, 'min_cells', int, 'a positive integer')
            check_bounds(min_cells, 'min_cells', 1)
        # Check that `hvg_column` and `rank_column` are strings and not already
        # in var
        for column, column_name in (hvg_column, 'hvg_column'), \
                (rank_column, 'rank_column'):
            check_type(column, column_name, str, 'a string')
            if hvg_column in self._var:
                error_message = (
                    f'{column_name} {column!r} is already a column of var; '
                    f'did you already run hvg()?')
                raise ValueError(error_message)
        # If `allow_float` is False, raise an error if `X` is floating-point
        check_type(allow_float, 'allow_float', bool, 'Boolean')
        if not allow_float and np.issubdtype(X.dtype, np.floating):
            error_message = (
                f'hvg() requires raw counts but X.dtype is {X.dtype!r}, a '
                f'floating-point data type. If you are sure that all values '
                f'are raw integer counts, i.e. that (X.data == '
                f'X.data.astype(int)).all(), then set allow_float=True.')
            raise TypeError(error_message)
        # Check that `num_threads` is a positive integer or None; if None, set
        # to `os.cpu_count()`
        if num_threads is None:
            num_threads = os.cpu_count()
        else:
            check_type(num_threads, 'num_threads', int, 'a positive integer')
            check_bounds(num_threads, 'num_threads', 1)
        # Get the batches to calculate variance across (datasets + batches
        # within each dataset). Subset to genes with non-zero counts in at
        # least `min_cells` in every dataset. Use X[np.ix_(rows, columns)] as
        # a faster, more memory-efficient alternative to X[rows][:, columns].
        if others:
            if min_cells is None:
                get_valid_genes = lambda dataset: dataset.var_names
            else:
                get_valid_genes = lambda dataset: dataset.var_names\
                    .filter(pl.Series(getnnz(dataset._X, axis=0) >= min_cells))
            self_valid_genes = get_valid_genes(self)
            genes_in_all_datasets = self_valid_genes\
                .filter(self_valid_genes.is_in(pl.concat([
                    get_valid_genes(dataset) for dataset in others])))
            gene_masks = (dataset.var_names.is_in(genes_in_all_datasets)
                          .to_numpy() for dataset in datasets)
            if batch_column is None:
                batches = ((dataset._X, QC_column.to_numpy()
                            if QC_column is not None else None, gene_mask)
                           for dataset, QC_column, gene_mask in
                           zip(datasets, QC_columns, gene_masks))
            else:
                batches = ((dataset._X,
                            (batch_column.eq(batch) if QC_column is None else
                             batch_column.eq(batch) & QC_column).to_numpy()
                            if batch is not None else
                            (QC_column.to_numpy() if QC_column is not None else
                             None), gene_mask)
                           for dataset, QC_column, batch_column, gene_mask in
                           zip(datasets, QC_columns, batch_columns, gene_masks)
                           for batch in ((None,) if batch_column is None else
                                         batch_column.unique()))
        else:
            if min_cells is None:
                gene_mask = None
                genes_in_all_datasets = self.var_names
            else:
                gene_mask = getnnz(X, axis=0) >= min_cells
                genes_in_all_datasets = self.var_names\
                    .filter(pl.Series(gene_mask))
            if batch_column is None:
                batches = (X, self._obs[QC_column].to_numpy()
                              if QC_column is not None else None, gene_mask),
            else:
                batches = ((X, (self._obs[batch_column].eq(batch)
                                if QC_column is None else
                                self._obs[batch_column].eq(batch) &
                                self._obs[QC_column]).to_numpy(), gene_mask)
                           for batch in self._obs[batch_column].unique())
        # Get the variance of each gene, across only cells passing QC;
        # get a mask of the `num_genes` genes with the highest variance
        prange_import = \
            'from cython.parallel cimport prange' if num_threads > 1 else ''
        norm_gene_vars = []
        for X, cell_mask, gene_mask in batches:
            is_CSR = isinstance(X, csr_array)
            if is_CSR:
                # To avoid having an additional level of indirection and
                # conditionals for each non-zero element, calculate mean and
                # variance across all genes, then subset after
                num_total_genes = X.shape[1]
                mean = np.zeros(num_total_genes)
                var = np.zeros(num_total_genes)
                counts = np.zeros(num_total_genes, dtype=int)
                if cell_mask is None:
                    cell_indices = np.array([], dtype=int)
                    num_cells = X.shape[0]
                else:
                    cell_indices = np.flatnonzero(cell_mask)
                    num_cells = len(cell_indices)
                pranges = {key: prange(key, num_threads, nogil=False)
                           for key in ('num_cells', 'num_total_genes',
                                       'indices.shape[0]')}
                cython_inline(rf'''
                    {prange_import}
                    def sparse_mean_var_minor_axis(
                            const {cython_type(X.dtype)}[::1] data,
                            const {cython_type(X.indices.dtype)}[::1] indices,
                            const {cython_type(X.indptr.dtype)}[::1] indptr,
                            const long[::1] cell_indices,
                            const long num_cells,
                            const long num_total_genes,
                            double[::1] mean,
                            double[::1] var,
                            long[::1] counts,
                            const unsigned num_threads):
                        
                        cdef long cell, gene, i, j
                        
                        with nogil:
                            if cell_indices.shape[0] == 0:
                                for i in {pranges['indices.shape[0]']}:
                                    gene = indices[i]
                                    mean[gene] += data[i]
                            else:
                                for i in {pranges['num_cells']}:
                                    cell = cell_indices[i]
                                    for j in range(indptr[cell],
                                                   indptr[cell + 1]):
                                        gene = indices[j]
                                        mean[gene] += data[j]
                            
                            for gene in {pranges['num_total_genes']}:
                                mean[gene] /= num_cells
                            
                            if cell_indices.shape[0] == 0:
                                for i in {pranges['indices.shape[0]']}:
                                    gene = indices[i]
                                    var[gene] += (data[i] - mean[gene]) ** 2
                                    counts[gene] += 1
                            else:
                                for i in {pranges['num_cells']}:
                                    cell = cell_indices[i]
                                    for j in range(indptr[cell],
                                                   indptr[cell + 1]):
                                        gene = indices[j]
                                        var[gene] += \
                                            (data[j] - mean[gene]) ** 2
                                        counts[gene] += 1
                            
                            for gene in {pranges['num_total_genes']}:
                                var[gene] += (num_cells - counts[gene]) * \
                                             mean[gene] ** 2
                                var[gene] /= num_cells - 1
                    ''')['sparse_mean_var_minor_axis'](
                        data=X.data, indices=X.indices, indptr=X.indptr,
                        cell_indices=cell_indices, num_cells=num_cells,
                        num_total_genes=num_total_genes, mean=mean, var=var,
                        counts=counts, num_threads=num_threads)
                if gene_mask is not None:
                    mean = mean[gene_mask]
                    var = var[gene_mask]
                    num_total_genes = len(mean)
            else:
                if cell_mask is None:
                    cell_mask = np.array([], dtype=bool)
                    num_cells = X.shape[0]
                else:
                    num_cells = cell_mask.sum()
                if gene_mask is None:
                    gene_indices = np.array([], dtype=int)
                    num_total_genes = X.shape[1]
                else:
                    gene_indices = np.flatnonzero(gene_mask)
                    num_total_genes = len(gene_indices)
                mean = np.zeros(num_total_genes)
                var = np.zeros(num_total_genes)
                cython_inline(rf'''
                    {prange_import}
                    def sparse_mean_var_major_axis(
                            const {cython_type(X.dtype)}[::1] data,
                            const {cython_type(X.indices.dtype)}[::1] indices,
                            const {cython_type(X.indptr.dtype)}[::1] indptr,
                            char[::1] cell_mask,
                            const long[::1] gene_indices,
                            const long num_cells,
                            const long num_total_genes,
                            double[::1] mean,
                            double[::1] var,
                            const unsigned num_threads):
                    
                        cdef long cell, gene, i
                        cdef {cython_type(X.indptr.dtype)} startptr, endptr, \
                            count
                        
                        if gene_indices.shape[0] == 0:
                            for gene in \
                                    {prange('num_total_genes', num_threads)}:
                                startptr = indptr[gene]
                                endptr = indptr[gene + 1]
                                
                                if cell_mask.shape[0] == 0:
                                    count = endptr - startptr
                                    for i in range(startptr, endptr):
                                        mean[gene] += data[i]
                                else:
                                    count = 0
                                    for i in range(startptr, endptr):
                                        cell = indices[i]
                                        if cell_mask[cell]:
                                            mean[gene] += data[i]
                                            count = count + 1
                                mean[gene] /= num_cells
                                
                                if cell_mask.shape[0] == 0:
                                    for i in range(startptr, endptr):
                                        var[gene] += \
                                            (data[i] - mean[gene]) ** 2
                                else:
                                    for i in range(startptr, endptr):
                                        cell = indices[i]
                                        if cell_mask[cell]:
                                            var[gene] += \
                                                (data[i] - mean[gene]) ** 2
                                var[gene] += \
                                    (num_cells - count) * mean[gene] ** 2
                                var[gene] /= num_cells - 1
                        else:
                            for gene in \
                                    {prange('num_total_genes', num_threads)}:
                                startptr = indptr[gene_indices[gene]]
                                endptr = indptr[gene_indices[gene] + 1]
                                
                                if cell_mask.shape[0] == 0:
                                    count = endptr - startptr
                                    for i in range(startptr, endptr):
                                        mean[gene] += data[i]
                                else:
                                    count = 0
                                    for i in range(startptr, endptr):
                                        cell = indices[i]
                                        if cell_mask[cell]:
                                            mean[gene] += data[i]
                                            count = count + 1
                                mean[gene] /= num_cells
                                
                                if cell_mask.shape[0] == 0:
                                    for i in range(startptr, endptr):
                                        var[gene] += \
                                            (data[i] - mean[gene]) ** 2
                                else:
                                    for i in range(startptr, endptr):
                                        cell = indices[i]
                                        if cell_mask[cell]:
                                            var[gene] += \
                                                (data[i] - mean[gene]) ** 2
                                var[gene] += \
                                    (num_cells - count) * mean[gene] ** 2
                                var[gene] /= num_cells - 1
                    ''')['sparse_mean_var_major_axis'](
                        data=X.data, indices=X.indices, indptr=X.indptr,
                        cell_mask=cell_mask, gene_indices=gene_indices,
                        num_cells=num_cells, num_total_genes=num_total_genes,
                        mean=mean, var=var, num_threads=num_threads)
                    
            not_constant = var > 0
            y = np.log10(var[not_constant])
            x = np.log10(mean[not_constant])
            model = loess(x, y, span=span)
            model.fit()
            
            estimated_variance = np.empty(num_total_genes)
            estimated_variance[not_constant] = model.outputs.fitted_values
            estimated_variance[~not_constant] = 0
            estimated_stddev = np.sqrt(10 ** estimated_variance)
            clip_val = mean + estimated_stddev * np.sqrt(num_cells)
            
            batch_counts_sum = np.zeros(num_total_genes)
            squared_batch_counts_sum = np.zeros(num_total_genes)
            if is_CSR:
                # Map the index of each gene in `X` to its index in `clip_val`
                # (-1 if not found)
                if gene_mask is None:
                    gene_map = np.array([], dtype=np.int32)
                else:
                    gene_map = \
                        np.where(gene_mask,
                                 np.cumsum(gene_mask, dtype=np.int32) - 1, -1)
                pranges = {key: prange(key, num_threads) for key in
                           ('indptr.shape[0] - 1', 'cell_indices.shape[0]')}
                # noinspection PyUnboundLocalVariable
                cython_inline(rf'''
                    {prange_import}
                    def clipped_sum(const {cython_type(X.dtype)}[::1] data,
                                    const {cython_type(X.indices.dtype)}[::1]
                                        indices,
                                    const {cython_type(X.indptr.dtype)}[::1]
                                        indptr,
                                    const long[::1] cell_indices,
                                    const int[::1] gene_map,
                                    const double[::1] clip_val,
                                    double[::1] batch_counts_sum,
                                    double[::1] squared_batch_counts_sum,
                                    const unsigned num_threads):
                        cdef long cell, gene, i, j
                        cdef double value
                        
                        if gene_map.shape[0] == 0:
                            if cell_indices.shape[0] == 0:
                                for cell in {pranges['indptr.shape[0] - 1']}:
                                    for i in range(indptr[cell],
                                                   indptr[cell + 1]):
                                        gene = indices[i]
                                        value = data[i]
                                        if value > clip_val[gene]:
                                            value = clip_val[gene]
                                        batch_counts_sum[gene] += value
                                        squared_batch_counts_sum[gene] += \
                                            value ** 2
                            else:
                                for j in {pranges['cell_indices.shape[0]']}:
                                    cell = cell_indices[j]
                                    for i in range(indptr[cell],
                                                   indptr[cell + 1]):
                                        gene = indices[i]
                                        value = data[i]
                                        if value > clip_val[gene]:
                                            value = clip_val[gene]
                                        batch_counts_sum[gene] += value
                                        squared_batch_counts_sum[gene] += \
                                            value ** 2
                        else:
                            if cell_indices.shape[0] == 0:
                                for cell in {pranges['indptr.shape[0] - 1']}:
                                    for i in range(indptr[cell],
                                                   indptr[cell + 1]):
                                        gene = gene_map[indices[i]]
                                        if gene == -1:
                                            continue
                                        value = data[i]
                                        if value > clip_val[gene]:
                                            value = clip_val[gene]
                                        batch_counts_sum[gene] += value
                                        squared_batch_counts_sum[gene] += \
                                            value ** 2
                            else:
                                for j in {pranges['cell_indices.shape[0]']}:
                                    cell = cell_indices[j]
                                    for i in range(indptr[cell],
                                                   indptr[cell + 1]):
                                        gene = gene_map[indices[i]]
                                        if gene == -1:
                                            continue
                                        value = data[i]
                                        if value > clip_val[gene]:
                                            value = clip_val[gene]
                                        batch_counts_sum[gene] += value
                                        squared_batch_counts_sum[gene] += \
                                            value ** 2
                    ''')['clipped_sum'](
                        data=X.data, indices=X.indices, indptr=X.indptr,
                        cell_indices=cell_indices, gene_map=gene_map,
                        clip_val=clip_val, batch_counts_sum=batch_counts_sum,
                        squared_batch_counts_sum=squared_batch_counts_sum,
                        num_threads=num_threads)
            else:
                pranges = {key: prange(key, num_threads) for key in
                           ('indptr.shape[0] - 1', 'gene_indices.shape[0]')}
                # noinspection PyUnboundLocalVariable
                cython_inline(rf'''
                    {prange_import}
                    def clipped_sum(const {cython_type(X.dtype)}[::1] data,
                                    const {cython_type(X.indices.dtype)}[::1]
                                        indices,
                                    const {cython_type(X.indptr.dtype)}[::1]
                                        indptr,
                                    char[::1] cell_mask,
                                    const long[::1] gene_indices,
                                    const double[::1] clip_val,
                                    double[::1] batch_counts_sum,
                                    double[::1] squared_batch_counts_sum,
                                    const unsigned num_threads):
                        cdef long cell, gene, i
                        cdef double value, clip_val_gene
                        
                        if cell_mask.shape[0] == 0:
                            if gene_indices.shape[0] == 0:
                                for gene in {pranges['indptr.shape[0] - 1']}:
                                    clip_val_gene = clip_val[gene]
                                    for i in range(indptr[gene],
                                                   indptr[gene + 1]):
                                        value = data[i]
                                        if value > clip_val_gene:
                                            value = clip_val_gene
                                        batch_counts_sum[gene] += value
                                        squared_batch_counts_sum[gene] += \
                                            value ** 2
                            else:
                                for gene in {pranges['gene_indices.shape[0]']}:
                                    clip_val_gene = clip_val[gene]
                                    for i in range(
                                            indptr[gene_indices[gene]],
                                            indptr[gene_indices[gene] + 1]):
                                        value = data[i]
                                        if value > clip_val_gene:
                                            value = clip_val_gene
                                        batch_counts_sum[gene] += value
                                        squared_batch_counts_sum[gene] += \
                                            value ** 2
                        else:
                            if gene_indices.shape[0] == 0:
                                for gene in {pranges['indptr.shape[0] - 1']}:
                                    clip_val_gene = clip_val[gene]
                                    for i in range(indptr[gene],
                                                   indptr[gene + 1]):
                                        cell = indices[i]
                                        if cell_mask[cell]:
                                            value = data[i]
                                            if value > clip_val_gene:
                                                value = clip_val_gene
                                            batch_counts_sum[gene] += value
                                            squared_batch_counts_sum[gene] += \
                                                value ** 2
                            else:
                                for gene in {pranges['gene_indices.shape[0]']}:
                                    clip_val_gene = clip_val[gene]
                                    for i in range(
                                            indptr[gene_indices[gene]],
                                            indptr[gene_indices[gene] + 1]):
                                        cell = indices[i]
                                        if cell_mask[cell]:
                                            value = data[i]
                                            if value > clip_val_gene:
                                                value = clip_val_gene
                                            batch_counts_sum[gene] += value
                                            squared_batch_counts_sum[gene] += \
                                                value ** 2
                    ''')['clipped_sum'](
                        data=X.data, indices=X.indices, indptr=X.indptr,
                        cell_mask=cell_mask, gene_indices=gene_indices,
                        clip_val=clip_val, batch_counts_sum=batch_counts_sum,
                        squared_batch_counts_sum=squared_batch_counts_sum,
                        num_threads=num_threads)
            norm_gene_var = \
                (1 / ((num_cells - 1) * np.square(estimated_stddev))) * \
                ((num_cells * np.square(mean)) + squared_batch_counts_sum -
                 2 * batch_counts_sum * mean)
            norm_gene_vars.append(norm_gene_var)
        
        norm_gene_vars = np.vstack(norm_gene_vars)
        # argsort twice gives ranks, small rank means most variable
        ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1),
                                           axis=1).astype(float)
        num_batches_high_var = (ranked_norm_gene_vars < num_genes).sum(axis=0)
        ranked_norm_gene_vars[ranked_norm_gene_vars >= num_genes] = np.nan
        median_ranked = np.ma.median(
            np.ma.masked_invalid(ranked_norm_gene_vars), axis=0).filled(np.nan)
        sort_cols = \
            'highly_variable_median_rank', -pl.col.highly_variable_nbatches
        if flavor == 'seurat_v3_paper':
            sort_cols = sort_cols[::-1]
        rank = pl.struct(sort_cols).rank('ordinal')
        var = pl.DataFrame({
            'gene': genes_in_all_datasets,
            'highly_variable_nbatches': num_batches_high_var,
            'highly_variable_median_rank': pl.Series(median_ranked,
                                                     nan_to_null=True)})\
            .select('gene', rank.le(num_genes).alias(hvg_column),
                    pl.when(rank <= num_genes).then(rank).alias(rank_column))
        # Return a new SingleCell dataset (or a tuple of datasets, if others
        # is non-empty) containing the highly variable genes
        for dataset_index, dataset in enumerate(datasets):
            new_var = dataset._var\
                .join(var.rename({'gene': dataset.var_names.name}),
                      on=dataset.var_names.name, how='left')\
                .with_columns(pl.col(hvg_column).fill_null(False))
            datasets[dataset_index] = \
                SingleCell(X=dataset._X, obs=dataset._obs, var=new_var)
        return tuple(datasets) if others else datasets[0]
    
    def normalize(self,
                  QC_column: SingleCellColumn | None = 'passed_QC',
                  method: Literal['PFlog1pPF', 'log1pPF',
                                  'logCP10k'] = 'PFlog1pPF',
                  allow_float: bool = False,
                  num_threads: int | np.integer | None = 1) -> SingleCell:
        """
        Normalize this SingleCell dataset's counts.
        
        By default, uses the PFlog1pPF method introduced in Booeshaghi et al.
        2022 (biorxiv.org/content/10.1101/2022.05.06.490859v1.full). With
        `method='logCP10k'`, it exactly matches the default settings of
        Seurat's `NormalizeData` function.
        
        PFlog1pPF is a three-step process:
        1. Divide each cell's counts by a "size factor", namely the total
        number of counts for that cell, divided by the mean number of counts
        across all cells. Booeshaghi et al. call this process, which performs
        rowwise division of a matrix `X` by the vector
        `X.sum(axis=1) / X.sum(axis=1).mean()`, "proportional fitting" (PF).
        2. Take the logarithm of each entry plus 1, i.e. `log1p()`.
        3. Run an additional round of proportional fitting.
        
        If method='log1pPF', only performs steps 1 and 2 and leaves out step 3.
        Booeshaghi et al. call this method "log1pPF". Ahlmann-Eltze and Huber
        2023 (nature.com/articles/s41592-023-01814-1) recommend this method and
        argue that it outperforms log(CPM) normalization. However, Booeshaghi
        et al. note that log1pPF does not fully normalize for read depth,
        because the log transform of step 2 partially undoes the normalization
        introduced by step 1. This is the reasoning behind their use of step 3:
        to restore full depth normalization. By default, scanpy's
        normalize_total() uses a variation of proportional fitting that divides
        by the median instead of the mean, so it's closest to method='log1pPF'.
        
        If method='logCP10k', uses 10,000 for the denominator of the size
        factors instead of `X.sum(axis=1).mean()`, and leaves out step 3. This
        method is not recommended because it implicitly assumes an
        unrealistically large amount of overdispersion, and performs worse than
        log1pPF and PFlog1pPF in Ahlmann-Eltze and Huber and Booeshaghi et
        al.'s benchmarks. Seurat's NormalizeData() uses logCP10k normalization.
        
        Args:
            QC_column: an optional Boolean column of obs indicating which cells
                       passed QC. Can be a column name, a polars expression, a
                       polars Series, a 1D NumPy array, or a function that 
                       takes in this SingleCell dataset and returns a polars 
                       Series or 1D NumPy array. Set to None to include all
                       cells. Cells failing QC will still be normalized, but
                       will not count towards the calculation of the mean total
                       count across cells when `method` is `'PFlog1pPF'` or
                       `'log1pPF'`. Has no effect when `method` is
                       `'logCP10k'`.
            method: the normalization method to use (see above)
            allow_float: if False, raise an error if `X.dtype` is
                         floating-point (suggesting the user may not be using
                         the raw counts); if True, disable this sanity check
            num_threads: the number of threads to use when normalizing. Set
                         `num_threads=None` to use all available cores (as
                         determined by `os.cpu_count()`).
        
        Returns:
            A new SingleCell dataset with the normalized counts, and
            `uns['normalized']` set to True.
        """
        # Check that `self` is QCed and not already normalized
        if not self._uns['QCed']:
            error_message = (
                "uns['QCed'] is False; did you forget to run qc()? Set "
                "uns['QCed'] = True or run with_uns(QCed=True) to bypass this "
                "check.")
            raise ValueError(error_message)
        if self._uns['normalized']:
            error_message = \
                "uns['normalized'] is True; did you already run normalize()?"
            raise ValueError(error_message)
        # Get `QC_column`, if not None
        if QC_column is not None:
            QC_column = self._get_column(
                'obs', QC_column, 'QC_column', pl.Boolean,
                allow_missing=QC_column == 'passed_QC')
        # Check that `method` is one of the three valid methods
        if method not in ('PFlog1pPF', 'log1pPF', 'logCP10k'):
            error_message = \
                "method must be one of 'PFlog1pPF', 'log1pPF', or 'logCP10k'"
            raise ValueError(error_message)
        # If `allow_float` is False, raise an error if `X` is floating-point
        X = self._X
        check_type(allow_float, 'allow_float', bool, 'Boolean')
        if not allow_float and np.issubdtype(X.dtype, np.floating):
            error_message = (
                f'normalize() requires raw counts but X.dtype is {X.dtype!r}, '
                f'a floating-point data type. If you are sure that all values '
                f'are raw integer counts, i.e. that (X.data == '
                f'X.data.astype(int)).all(), then set allow_float=True.')
            raise TypeError(error_message)
        # Check that `num_threads` is a positive integer or None; if None, set
        # to `os.cpu_count()`
        if num_threads is None:
            num_threads = os.cpu_count()
        else:
            check_type(num_threads, 'num_threads', int, 'a positive integer')
            check_bounds(num_threads, 'num_threads', 1)
        # Step 1
        rowsums = X.sum(axis=1)
        if method == 'logCP10k':
            # Purposely use an order of operations with more floating-point
            # error, to exactly match Seurat
            # (github.com/satijalab/seurat/blob/master/R/preprocessing5.R#L870)
            X = sparse_matrix_vector_op(X, '/', rowsums, axis=0,
                                        return_dtype=float,
                                        num_threads=num_threads)
            X *= 10_000
        else:
            inverse_size_factors = np.empty_like(rowsums, dtype=float) \
                if np.issubdtype(rowsums.dtype, np.integer) else rowsums
            # Note: QCed cells will have null as the batch, and over() treats
            # null as its own category, so effectively all cells failing QC
            # will be treated as their own batch. This doesn't matter since we
            # never use the counts for these cells anyway.
            np.divide(rowsums.mean() if QC_column is None else
                      rowsums[QC_column].mean(), rowsums, inverse_size_factors)
            X = sparse_matrix_vector_op(X, '*', inverse_size_factors, axis=0,
                                        return_dtype=float,
                                        num_threads=num_threads)
        # Step 2
        if num_threads == 1:
            np.log1p(X.data, X.data)
        else:
            cython_inline(f'''
                from cython.parallel cimport prange
                from libc.math cimport log1p
                
                def log1p_parallel(double[::1] array,
                                   const unsigned num_threads):
                    cdef long i
                    for i in prange(array.shape[0], nogil=True,
                                    num_threads=num_threads):
                        array[i] = log1p(array[i])
                ''')['log1p_parallel'](X.data, num_threads)
        # Step 3
        if method == 'PFlog1pPF':
            rowsums = X.sum(axis=1)
            inverse_size_factors = rowsums
            np.divide(rowsums.mean() if QC_column is None else
                      rowsums[QC_column].mean(), rowsums, inverse_size_factors)
            sparse_matrix_vector_op(X, '*', inverse_size_factors, axis=0,
                                    inplace=True, return_dtype=float,
                                    num_threads=num_threads)
        sc = SingleCell(X=X, obs=self._obs, var=self._var, obsm=self._obsm,
                        varm=self._varm, uns=self._uns)
        sc._uns['normalized'] = True
        return sc
    
    def PCA(self,
            *others: SingleCell,
            QC_column: SingleCellColumn | None |
                       Sequence[SingleCellColumn | None] = 'passed_QC',
            hvg_column: SingleCellColumn |
                        Sequence[SingleCellColumn] = 'highly_variable',
            PC_key: str = 'PCs',
            num_PCs: int | np.integer = 50,
            seed: int | np.integer | None = 0,
            num_threads: int | np.integer | None = 1,
            verbose: bool = False) -> SingleCell | tuple[SingleCell, ...]:
        """
        Compute principal components using irlba, the package used by Seurat.
        Operates on normalized counts (see `normalize()`).
        
        Install irlba with:
        
        from ryp import r
        r('install.packages("irlba", type="source")')
        
        IMPORTANT: if you already have a copy of irlba from CRAN (e.g.
        installed with Seurat), you will get the error:
        
        RuntimeError: in irlba(X, 50, verbose = FALSE) :
          function 'as_cholmod_sparse' not provided by package 'Matrix'
        
        This error will go away if you install irlba from source as described
        above.
        
        Args:
            others: optional SingleCell datasets to jointly compute principal
                    components across, alongside this one.
            QC_column: an optional Boolean column of obs indicating which cells
                       passed QC. Can be a column name, a polars expression, a 
                       polars Series, a 1D NumPy array, or a function that 
                       takes in this SingleCell dataset and returns a polars 
                       Series or 1D NumPy array. Set to None to include all
                       cells. Cells failing QC will be ignored and have their
                       PCs set to NaN. When `others` is specified, `QC_column`
                       can be a length-`1 + len(others)` sequence of columns,
                       expressions, Series, functions, or None for each dataset
                       (for `self`, followed by each dataset in `others`).
            hvg_column: a Boolean column of var indicating the highly variable
                        genes. Can be a column name, a polars expression, a 
                        polars Series, a 1D NumPy array, or a function that 
                        takes in this SingleCell dataset and returns a polars 
                        Series or 1D NumPy array. Set to None to use all genes.
                        When `others` is specified, `hvg_column`
                        can be a length-`1 + len(others)` sequence of columns,
                        expressions, Series, functions, or None for each
                        dataset (for `self`, followed by each dataset in
                        `others`).
            PC_key: the key of obsm where the principal components will be
                    stored
            num_PCs: the number of top principal components to calculate
            seed: the random seed to use for irlba when computing PCs, via R's
                  set.seed() function; if None, do not set a seed
            num_threads: the number of threads to use when running PCA. Set
                         `num_threads=None` to use all available cores (as
                         determined by `os.cpu_count()`).
            verbose: whether to set the verbose flag in irlba
        
        Returns:
            A new SingleCell dataset where obsm contains an additional key,
            `PC_key` (default: 'PCs'), containing a NumPy array of the top
            `num_PCs` principal components. Or, if additional SingleCell
            dataset(s) are specified via the `others` argument, a
            length-`1 + len(others)` tuple of SingleCell datasets with the PCs
            added: `self`, followed by each dataset in `others`.
        
        Note:
            Unlike Seurat's `RunPCA` function, which requires `ScaleData` to be
            run first, this function does not require the data to be scaled
            beforehand. Instead, it scales the data implicitly. It does this by
            providing the standard deviation and mean of the data to `irlba()`
            via its `scale` and `center` arguments, respectively. This approach
            is much more computationally efficient than explicit scaling, and
            is also taken by Seurat's internal (and currently unused)
            `RunPCA_Sparse` function, which this function is based on.
        """
        from ryp import r, to_py, to_r
        from sklearn.utils.sparsefuncs import mean_variance_axis
        from threadpoolctl import threadpool_limits
        r('suppressPackageStartupMessages(library(irlba))')
        # Check that all elements of `others` are SingleCell datasets
        if others:
            check_types(others, 'others', SingleCell, 'SingleCell datasets')
        datasets = [self] + list(others)
        # Check that all datasets are normalized
        suffix = ' for at least one dataset' if others else ''
        if not all(dataset._uns['normalized'] for dataset in datasets):
            suffix = ' for at least one dataset' if others else ''
            error_message = (
                f"PCA() requires normalized counts but uns['normalized'] is "
                f"False{suffix}; did you forget to run normalize()?")
            raise ValueError(error_message)
        # Raise an error if `X` has an integer data type for any dataset
        for dataset in datasets:
            if np.issubdtype(dataset._X.dtype, np.integer):
                error_message = (
                    f'PCA() requires raw counts, but X.dtype is '
                    f'{dataset._X.dtype!r}, an integer data type{suffix}; did '
                    f'you forget to run normalize() before PCA()?')
                raise TypeError(error_message)
        # Get `QC_column` (if not None) and `hvg_column` from every dataset
        QC_columns = SingleCell._get_columns(
            'obs', datasets, QC_column, 'QC_column', pl.Boolean,
            allow_missing=True)
        hvg_columns = SingleCell._get_columns(
            'var', datasets, hvg_column, 'hvg_column', pl.Boolean,
            allow_None=False,
            custom_error=f'hvg_column {{}} is not a column of var{suffix}; '
                         f'did you forget to run hvg() before PCA()?')
        # Check that `PC_key` is not already in obsm
        check_type(PC_key, 'PC_key', str, 'a string')
        for dataset in datasets:
            if PC_key in dataset._obsm:
                error_message = (
                    f'PC_key {PC_key!r} is already a key of obsm{suffix}; did '
                    f'you already run PCA()?')
                raise ValueError(error_message)
        # Check other inputs
        check_type(num_PCs, 'num_PCs', int, 'a positive integer')
        check_bounds(num_PCs, 'num_PCs', 1)
        if seed is not None:
            check_type(seed, 'seed', int, 'an integer')
        if num_threads is None:
            num_threads = os.cpu_count()
        else:
            check_type(num_threads, 'num_threads', int, 'a positive integer')
            check_bounds(num_threads, 'num_threads', 1)
        check_type(verbose, 'verbose', bool, 'Boolean')
        # Get the matrix to compute PCA across: a CSC array of counts for
        # highly variable genes (or all genes, if `hvg_column` is None) across
        # cells passing QC. Use X[np.ix_(rows, columns)] as a faster, more
        # memory-efficient alternative to X[rows][:, columns]. Use CSC rather
        # than CSR because irlba has a fast C-based implementation for CSC.
        if others:
            if hvg_column is None:
                genes_in_all_datasets = self.var_names\
                    .filter(self.var_names
                            .is_in(pl.concat([dataset.var_names
                                              for dataset in others])))
            else:
                hvg_in_self = self._var.filter(hvg_columns[0]).to_series()
                genes_in_all_datasets = hvg_in_self\
                    .filter(hvg_in_self
                            .is_in(pl.concat([dataset.var.filter(hvg_col)
                                              .to_series()
                                              for dataset, hvg_col in
                                              zip(others, hvg_columns[1:])])))
            gene_indices = (
                genes_in_all_datasets
                .to_frame()
                .join(dataset._var.with_row_index('_SingleCell_index'),
                      left_on=genes_in_all_datasets.name,
                      right_on=dataset.var_names.name, how='left')
                ['_SingleCell_index']
                .to_numpy()
                for dataset in datasets)
            if QC_column is None:
                Xs = [dataset._X[:, genes]
                      for dataset, genes in zip(datasets, gene_indices)]
            else:
                Xs = [dataset._X[np.ix_(dataset._obs[QC_col].to_numpy(),
                                        genes)]
                      for dataset, genes, QC_col in
                      zip(datasets, gene_indices, QC_columns)]
        else:
            if QC_column is None:
                if hvg_column is None:
                    Xs = [dataset.X for dataset in datasets]
                else:
                    Xs = [dataset.X[:, dataset._var[hvg_col].to_numpy()]
                          for dataset, hvg_col in zip(datasets, hvg_columns)]
            else:
                if hvg_column is None:
                    Xs = [dataset.X[dataset._obs[QC_col].to_numpy()]
                          for dataset, QC_col in zip(datasets, QC_columns)]
                else:
                    Xs = [dataset.X[np.ix_(
                              dataset._obs[QC_col].to_numpy(),
                              dataset._var[hvg_col].to_numpy())]
                          for dataset, QC_col, hvg_col in
                          zip(datasets, QC_columns, hvg_columns)]
        X = vstack(Xs, format='csc')
        num_cells_per_dataset = np.array([X.shape[0] for X in Xs])
        del Xs
        # Check that `num_PCs` is at most the width of this matrix
        check_bounds(num_PCs, 'num_PCs', upper_bound=X.shape[1])
        # Run PCA with irlba (github.com/bwlewis/irlba/blob/master/R/irlba.R)
        # This section is adapted from
        # github.com/satijalab/seurat/blob/master/R/integration.R#L7276-L7317
        # Note: totalvar doesn't seem to be used by irlba, maybe a Seurat bug?
        center, feature_var = mean_variance_axis(X, axis=0)
        scale = np.sqrt(feature_var)
        scale.clip(min=1e-8, out=scale)
        to_r(X, '.SingleCell.X')
        try:
            to_r(center, '.SingleCell.center')
            try:
                to_r(scale, '.SingleCell.scale')
                try:
                    if seed is not None:
                        r(f'set.seed({seed})')
                    with threadpool_limits(limits=num_threads):
                        r(f'.SingleCell.PCs = irlba(.SingleCell.X, {num_PCs}, '
                          f'verbose={str(verbose).upper()}, '
                          f'scale=.SingleCell.scale, '
                          f'center=.SingleCell.center)')
                    try:
                        PCs = to_py('.SingleCell.PCs$u', format='numpy') * \
                              to_py('.SingleCell.PCs$d', format='numpy')
                    finally:
                        r('rm(".SingleCell.PCs")')
                finally:
                    r('rm(".SingleCell.scale")')
            finally:
                r('rm(".SingleCell.center")')
        finally:
            r('rm(".SingleCell.X")')
        # Store each dataset's PCs in its obsm
        for dataset_index, (dataset, QC_col, num_cells, end_index) in \
                enumerate(zip(datasets, QC_columns, num_cells_per_dataset,
                              num_cells_per_dataset.cumsum())):
            start_index = end_index - num_cells
            dataset_PCs = PCs[start_index:end_index]
            # If `QC_col` is not None for this dataset, back-project from QCed
            # cells to all cells, filling with NaN
            if QC_col is not None:
                dataset_PCs_QCed = dataset_PCs
                dataset_PCs = np.full((len(dataset),
                                       dataset_PCs_QCed.shape[1]), np.nan)
                dataset_PCs[dataset._obs[QC_col].to_numpy()] = dataset_PCs_QCed
            datasets[dataset_index] = SingleCell(
                X=dataset._X, obs=dataset._obs, var=dataset._var,
                obsm=dataset._obsm | {PC_key: dataset_PCs}, varm=self._varm,
                uns=self._uns)
        return tuple(datasets) if others else datasets[0]
    
    def harmonize(self,
                  *others: SingleCell,
                  QC_column: SingleCellColumn | None |
                             Sequence[SingleCellColumn | None] = 'passed_QC',
                  batch_column: SingleCellColumn | None |
                                Sequence[SingleCellColumn | None] = None,
                  PC_key: str = 'PCs',
                  Harmony_key: str = 'Harmony_PCs',
                  max_iter_harmony: int | np.integer = sys.maxsize,
                  max_iter_kmeans: int | np.integer = sys.maxsize,
                  pytorch: bool = True,
                  seed: int | np.integer = 0,
                  num_threads: int | None = 1,
                  verbose: bool = True,
                  **kwargs: Any) -> SingleCell | tuple[SingleCell, ...]:
        """
        Harmonize this SingleCell dataset with other datasets, using Harmony
        (nature.com/articles/s41592-019-0619-0).
        
        The original Harmony R package (github.com/immunogenomics/harmony) has
        two Python ports:
        - harmonypy (github.com/slowkow/harmonypy):
          pip install --no-deps --no-build-isolation harmonypy
        - harmony-pytorch (github.com/lilab-bcb/harmony-pytorch):
          mamba install -y pytorch && \
            pip install --no-deps --no-build-isolation harmony-pytorch
        
        We use harmonypy when `pytorch=False` (the default) and harmony-pytorch
        when `pytorch=True`.
        
        Args:
            others: the other SingleCell datasets to harmonize this one with
            QC_column: an optional Boolean column of obs indicating which cells
                       passed QC. Can be a column name, a polars expression, a 
                       polars Series, a 1D NumPy array, or a function that 
                       takes in this SingleCell dataset and returns a polars 
                       Series or 1D NumPy array. Set to None to include all
                       cells. Cells failing QC will be ignored and have their
                       Harmony embeddings set to NaN. When `others` is
                       specified, `QC_column` can be a length-`1 + len(others)`
                       sequence of columns, expressions, Series, functions, or
                       None for each dataset (for `self`, followed by each
                       dataset in `others`).
            batch_column: an optional String, Categorical, Enum, or integer
                          column of obs indicating which batch each cell is
                          from. Can be a column name, a polars expression, a 
                          polars Series, a 1D NumPy array, or a function that 
                          takes in this SingleCell dataset and returns a polars 
                          Series or 1D NumPy array. Each batch will be treated
                          as if it were a distinct dataset; this is exactly
                          equivalent to splitting the dataset with
                          `split_by(batch_column)` and then passing each of the
                          resulting datasets to `harmonize()`. Set to None to
                          treat each dataset as having a single batch. When
                          `others` is specified, `batch_column` may be a
                          length-`1 + len(others)` sequence of columns,
                          expressions, Series, functions, or None for each
                          dataset (for `self`, followed by each dataset in
                          `others`).
            PC_key: the key of obsm containing the principal components
                    calculated with PCA(), to use as the input to Harmony
            Harmony_key: the key of obsm where the Harmony embeddings will be
                         stored; will be added in-place to both `self` and each
                         of the datasets in `others`!
            max_iter_harmony: the maximum number of iterations to run Harmony
                              for, if convergence is not achieved; overrides
                              the default of 10 in `harmony()` and
                              `run_harmony()`
            max_iter_kmeans: the maximum number of iterations to run the
                             clustering step within each Harmony iteration for,
                             if convergence is not achieved; overrides the
                             default of 200 in `harmony()` and 20 in
                             `run_harmony()`
            pytorch: if True, use harmony-pytorch instead of harmony
            seed: the random seed for Harmony
            num_threads: the number of threads to use when running Harmony; set
                         `num_threads=None` to use all available cores (as
                         determined by `os.cpu_count()`). Only used when
                         `pytorch` is True.
            verbose: whether to print details from `run_harmony()` or
                     `harmony()`
            **kwargs: other keyword arguments passed to `run_harmony()` from
                      harmonypy (if `pytorch=False`) or `harmony()` from
                      harmony-pytorch (if `pytorch=True`); see
                      github.com/slowkow/harmonypy/blob/master/harmonypy/
                      harmony.py#L35-L55 and
                      github.com/lilab-bcb/harmony-pytorch/blob/main/harmony/
                      harmony.py#L13-L31 for possible arguments. When
                      `pytorch=True`, specify `use_gpu=True` to run on the GPU,
                      if available.
        
        Returns:
            A length-`1 + len(others)` tuple of SingleCell datasets with the
            Harmony embeddings added as obsm[Harmony_key]: `self`, followed by
            each dataset in `others`.
        """
        # Make pandas import non-interruptible, to avoid bugs due to partial
        # imports
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            import pandas as pd
        finally:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        # Check `others`
        if not others:
            error_message = 'others cannot be empty'
            raise ValueError(error_message)
        check_types(others, 'others', SingleCell, 'SingleCell datasets')
        datasets = [self] + list(others)
        # Get `QC_column` and `batch_column` from every dataset, if not None
        QC_columns = SingleCell._get_columns(
            'obs', datasets, QC_column, 'QC_column', pl.Boolean,
            allow_missing=True)
        # noinspection PyUnresolvedReferences
        QC_columns_NumPy = [QC_col.to_numpy() if QC_col is not None else None
                            for QC_col in QC_columns]
        batch_columns = SingleCell._get_columns(
            'obs', datasets, batch_column, 'batch_column',
            (pl.String, pl.Categorical, pl.Enum, 'integer'))
        # Check other inputs
        check_type(PC_key, 'PC_key', str, 'a string')
        if not all(PC_key in dataset._obsm for dataset in datasets):
            error_message = (
                f'PC_key {PC_key!r} is not a column of obs for at least one '
                f'dataset; did you forget to run PCA() before harmonize()?')
            raise ValueError(error_message)
        check_type(Harmony_key, 'Harmony_key', str, 'a string')
        if any(Harmony_key in dataset._obsm for dataset in datasets):
            error_message = (
                f'Harmony_key {Harmony_key!r} is already a key of obsm for at '
                f'least one dataset; did you already run harmonize()?')
            raise ValueError(error_message)
        check_type(max_iter_harmony, 'max_iter_harmony', int,
                   'a positive integer')
        check_bounds(max_iter_harmony, 'max_iter_harmony', 1)
        check_type(max_iter_kmeans, 'max_iter_kmeans', int,
                   'a positive integer')
        check_bounds(max_iter_kmeans, 'max_iter_kmeans', 1)
        check_type(pytorch, 'pytorch', bool, 'Boolean')
        check_type(seed, 'seed', int, 'an integer')
        if num_threads is None:
            num_threads = os.cpu_count()
        else:
            if not pytorch and num_threads > 1:
                error_message = \
                    'setting num_threads is only supported when pytorch=True'
                raise ValueError(error_message)
            check_type(num_threads, 'num_threads', int, 'a positive integer')
            check_bounds(num_threads, 'num_threads', 1)
        check_type(verbose, 'verbose', bool, 'Boolean')
        # Concatenate PCs across datasets; get labels indicating which rows of
        # these concatenated PCs come from each dataset or batch
        if QC_column is None:
            PCs = [dataset._obsm[PC_key] for dataset in datasets]
        else:
            PCs = [dataset._obsm[PC_key][QCed]
                   for dataset, QCed in zip(datasets, QC_columns_NumPy)]
        num_cells_per_dataset = np.array(list(map(len, PCs)))
        if batch_column is None:
            batch_labels = np.repeat(np.arange(len(num_cells_per_dataset)),
                                     num_cells_per_dataset)
        else:
            batch_labels = []
            batch_index = 0
            for dataset, QC_col, batch_col in \
                    zip(datasets, QC_columns, batch_columns):
                if batch_col is not None:
                    if QC_col is not None:
                        batch_col = batch_col.filter(QC_col)
                    if batch_col.dtype in (pl.String, pl.Categorical, pl.Enum):
                        if batch_col.dtype != pl.Enum:
                            batch_col = batch_col\
                                .cast(pl.Enum(batch_col.unique().drop_nulls()))
                        batch_col = batch_col.to_physical()
                    batch_labels.append(batch_col.to_numpy() + batch_index)
                    batch_index += batch_col.n_unique()
                else:
                    batch_labels.append(np.full(batch_index,
                                                len(dataset) if QC_col is None
                                                else QC_col.sum()))
                    batch_index += 1
            batch_labels = np.concatenate(batch_labels)
        batch_labels = pd.Series(batch_labels, dtype='category')
        PCs = np.concatenate(PCs)
        # Run Harmony
        if pytorch:
            from harmony import harmonize
            Harmony_embedding = harmonize(
                PCs, pd.DataFrame({'batch': batch_labels}), 'batch',
                max_iter_harmony=max_iter_harmony,
                max_iter_clustering=max_iter_kmeans, random_state=seed,
                n_jobs=num_threads, verbose=verbose, **kwargs)
        else:
            from harmonypy import run_harmony
            Harmony_embedding = run_harmony(
                PCs, pd.DataFrame({'batch': batch_labels}), 'batch',
                max_iter_harmony=max_iter_harmony,
                max_iter_kmeans=max_iter_kmeans,
                verbose=verbose, random_state=seed, **kwargs).Z_corr.T
        # Store each dataset's Harmony embedding in its obsm
        for dataset_index, (dataset, QC_col, num_cells, end_index) in \
                enumerate(zip(datasets, QC_columns_NumPy,
                              num_cells_per_dataset,
                              num_cells_per_dataset.cumsum())):
            start_index = end_index - num_cells
            dataset_Harmony_embedding = \
                Harmony_embedding[start_index:end_index]
            # If `QC_col` is not None for this dataset, back-project from
            # QCed cells to all cells, filling with NaN
            if QC_col is not None:
                dataset_Harmony_embedding_QCed = dataset_Harmony_embedding
                dataset_Harmony_embedding = np.full(
                    (len(dataset), dataset_Harmony_embedding_QCed.shape[1]),
                    np.nan)
                # noinspection PyUnboundLocalVariable
                dataset_Harmony_embedding[QC_col] = \
                    dataset_Harmony_embedding_QCed
            datasets[dataset_index] = SingleCell(
                X=dataset._X, obs=dataset._obs, var=dataset._var,
                obsm=dataset._obsm | {Harmony_key: dataset_Harmony_embedding},
                varm=self._varm, uns=self._uns)
        return tuple(datasets) if others else datasets[0]
    
    def label_transfer_from(
            self,
            other: SingleCell,
            original_cell_type_column: SingleCellColumn,
            *,
            QC_column: SingleCellColumn | None = 'passed_QC',
            other_QC_column: SingleCellColumn | None = 'passed_QC',
            Harmony_key: str = 'Harmony_PCs',
            cell_type_column: str = 'cell_type',
            cell_type_confidence_column: str = 'cell_type_confidence',
            min_cell_type_confidence: int | float | np.integer | np.floating |
                                      None = None,
            num_neighbors: int | np.integer = 20,
            num_index_neighbors: int | np.integer = 60,
            num_trees: int | np.integer | None = 100,
            seed: int | np.integer | None = 0,
            num_threads: int | np.integer | None = 1,
            verbose: bool = True) -> SingleCell:
        """
        Transfer cell-type labels from another dataset to this one.
        
        Runs approximate k-nearest neighbors on the Harmony embeddings from
        `harmonize()` via nndescent, a C++ port of the popular pynndescent
        package.
        
        Install nndescent with:
        
        mamba install -y pybind11 && \
          pip install --no-deps --no-build-isolation \
          git+https://github.com/Wainberg/nndescent
        
        For each cell in `self`, the transferred cell-type label is the most
        common cell-type label among the `num_neighbors` cells in `other` with
        the nearest Harmony embeddings. The cell-type confidence is the
        fraction of these neighbors that share this most common cell-type
        label.
        
        Args:
            other: the dataset to transfer cell-type labels from
            original_cell_type_column: a column of `other.obs` containing
                                       cell-type labels. Can be a column name, 
                                       a polars expression, a polars Series, a
                                       1D NumPy array, or a function that takes
                                       in `other` and returns a polars Series
                                       or 1D NumPy array.
            QC_column: an optional Boolean column of `self.obs` indicating
                       which cells passed QC. Can be a column name, a polars 
                       expression, a polars Series, a 1D NumPy array, or a
                       function that takes in `self` and returns a polars 
                       Series or 1D NumPy array. Set to None to include all
                       cells. Cells failing QC will be ignored during the label
                       transfer.
            other_QC_column: an optional Boolean column of `other.obs`
                             indicating which cells passed QC. Can be a column
                             name, a polars expression, a polars Series, a 1D
                             NumPy array, or a function that takes in `other`
                             and returns a polars Series or 1D NumPy array. Set
                             to None to include all cells. Cells failing QC 
                             will be ignored during the label transfer.
            Harmony_key: the key of `self.obsm` and `other.obsm` containing the
                         Harmony embeddings for each dataset
            cell_type_column: the name of a column to be added to `self.obs`
                              indicating each cell's cell type, i.e. the most
                              common cell-type label among the cell's
                              `num_neighbors` nearest neighbors in obs
            cell_type_confidence_column: the name of a column to be added to
                                         `self.obs` indicating each cell's
                                         cell-type confidence, i.e. the
                                         fraction of the cell's `num_neighbors`
                                         nearest neighbors in obs that share
                                         the most common cell-type label. If
                                         multiple cell types are equally common
                                         among the nearest neighbors, tiebreak
                                         based on which of them is most common
                                         in `other`.
            min_cell_type_confidence: if not None, exclude cells from `self`
                                      with cell-type confidences less than this
                                      threshold from downstream analyses, by
                                      setting `QC_column` to False for these
                                      cells. Must be None if `QC_column` is
                                      None.
            num_neighbors: the number of nearest neighbors to use when
                           determining a cell's label
            num_index_neighbors: the number of nearest neighbors to use when
                                 building the index used for the
                                 nearest-neighbors search. Larger values (e.g.
                                 50-100) will lead to more accurate
                                 determination of a cell's true k-nearest
                                 neighbors at the cost of increased runtime.
                                 Defaults to 60, the number used by default in
                                 umap-learn, rather than nndescent's faster but
                                 less conservative default of 30.
            num_trees: the number of random projection trees to use when
                       building the index used for nearest-neighbor search.
                       Paradoxically, increasing the number of trees actually
                       reduces runtime (up to a point) by reducing the number
                       of subsequent graph updates required. As a result, we
                       use a default of 100 trees instead of the smaller
                       default of `min(32, 5 + int(round(num_cells ** 0.25)))`
                       used by nndescent and pynndescent (this default can be
                       re-enabled by setting `num_trees=None`).
            seed: the random seed for (py)nndescent to use when finding nearest
                  neighbors; if None, do not set a seed
            num_threads: the number of threads to use for the nearest-neighbors
                         tree construction and search. Set `num_threads=None`
                         to use all available cores (as determined by
                         `os.cpu_count()`).
            verbose: whether to print details of the nearest-neighbor search
                     from (py)nndescent, and whether to print the number of
                     cells filtered out if `min_cell_type_confidence=True`
        
        Returns:
            `self`, but with two additional columns: `cell_type_column`,
            containing the transferred cell-type labels, and
            `cell_type_confidence_column`, containing the cell-type
            confidences.
        """
        # Check that `other` is a SingleCell dataset
        check_type(other, 'other', SingleCell, 'a SingleCell dataset')
        # Get `original_cell_type_column` from `other`
        original_cell_type_column = other._get_column(
            'obs', original_cell_type_column, 'original_cell_type_column',
            (pl.Categorical, pl.Enum, pl.String))
        # Get `QC_column` from `self` and `other_QC_column` from `other`;
        # if `other_QC_column` is not None, filter cell type labels to cells
        # passing QC
        if QC_column is not None:
            QC_column = self._get_column(
                'obs', QC_column, 'QC_column', pl.Boolean,
                allow_missing=QC_column == 'passed_QC')
        if other_QC_column is not None:
            other_QC_column = other._get_column(
                'obs', other_QC_column, 'other_QC_column', pl.Boolean,
                allow_missing=QC_column == 'passed_QC')
            original_cell_type_column = \
                original_cell_type_column.filter(other_QC_column)
        # Check that `original_cell_type_column` has at least two distinct cell
        # types
        most_common_cell_types = \
            original_cell_type_column.value_counts(sort=True).to_series()
        if len(most_common_cell_types) == 1:
            error_message = (
                f'original_cell_type_column must have at least two distinct '
                f'cell types')
            if other_QC_column is not None:
                error_message += ' after filtering to cells passing QC'
            raise ValueError(error_message)
        # Check that `Harmony_key` is a string and in both `self.obsm` and
        # `other.obsm`
        check_type(Harmony_key, 'Harmony_key', str, 'a string')
        datasets = (self, 'self'), (other, 'other')
        for dataset, dataset_name in datasets:
            if Harmony_key not in dataset._obsm:
                error_message = (
                    f'Harmony_key {Harmony_key!r} is not a column of '
                    f'{dataset_name}.obs; did you forget to run harmonize() '
                    f'before label_transfer_from()?')
                raise ValueError(error_message)
        # Check that `cell_type_column` and `cell_type_confidence_column` are
        # strings and not already columns of `self.obs`
        for column, column_name in (cell_type_column, 'cell_type_column'), \
                (cell_type_confidence_column, 'cell_type_confidence_column'):
            check_type(column, column_name, str, 'a string')
            if column in self._obs:
                error_message = (
                    f'{column_name} {column!r} is already a column '
                    f'of obs; did you already run label_transfer_from()?')
                raise ValueError(error_message)
        # Check other inputs
        if min_cell_type_confidence is not None:
            check_type(min_cell_type_confidence, 'min_cell_type_confidence',
                       (int, float), 'a number between 0 and 1, inclusive')
            check_bounds(min_cell_type_confidence, 'min_cell_type_confidence',
                         0, 1)
        for variable, variable_name in (
                (num_neighbors, 'num_neighbors'),
                (num_index_neighbors, 'num_index_neighbors')):
            check_type(variable, variable_name, int, 'a positive integer')
            check_bounds(variable, variable_name, 1)
        if num_trees is not None:
            check_type(num_trees, 'num_trees', int, 'a positive integer')
            check_bounds(num_trees, 'num_trees', 1)
        if seed is not None:
            check_type(seed, 'seed', int, 'an integer')
        if num_threads is None:
            num_threads = os.cpu_count()
        else:
            check_type(num_threads, 'num_threads', int, 'a positive integer')
            check_bounds(num_threads, 'num_threads', 1)
        check_type(verbose, 'verbose', bool, 'Boolean')
        # Recode cell types so the most common is 0, the next-most common 1,
        # etc. This has the effect of breaking ties by taking the most common
        # cell type: NumPy's mode() picks the first element in case of ties.
        cell_type_to_code = dict(zip(most_common_cell_types, range(
            len(most_common_cell_types))))
        original_cell_type_column = original_cell_type_column\
            .replace_strict(cell_type_to_code, return_dtype=pl.Int32)
        # Get the Harmony embeddings for self and other
        self_Harmony_embeddings = self._obsm[Harmony_key] \
            if QC_column is None else \
            self._obsm[Harmony_key][QC_column.to_numpy()]
        other_Harmony_embeddings = other._obsm[Harmony_key] \
            if other_QC_column is None else \
            other._obsm[Harmony_key][other_QC_column.to_numpy()]
        # Use nndescent to get the indices of the num_neighbors nearest
        # neighbors in other for each cell in self
        # noinspection PyUnresolvedReferences
        from nndescent import NNDescent
        self_Harmony_embeddings = \
            self_Harmony_embeddings.astype(np.float32, copy=False)
        other_Harmony_embeddings = \
            other_Harmony_embeddings.astype(np.float32, copy=False)
        nn_indices = NNDescent(other_Harmony_embeddings,
                               n_neighbors=num_index_neighbors,
                               n_trees=num_trees, seed=seed,
                               n_threads=num_threads, verbose=verbose)\
            .query(self_Harmony_embeddings, k=num_neighbors)[0]
        # Get the cell-type labels of these nearest neighbors (using our
        # integer encoding where the most common cell type is 0, the next-most
        # common 1, etc.)
        nn_cell_types = original_cell_type_column.to_numpy()[nn_indices]
        # For each cell in self, get the most common cell type among the
        # nearest neighbors using scipy's mode(). As mentioned above, mode()
        # picks the first element in case of ties, which according to our
        # encoding is the most common cell type. mode() also returns the
        # frequency of the most common cell type, which we normalize by
        # num_neighbors to convert to the cell-type confidence.
        cell_type, cell_type_confidence = \
            mode(nn_cell_types, axis=1, keepdims=False)
        cell_type_confidence = pl.Series(cell_type_confidence_column,
                                         cell_type_confidence / num_neighbors)
        # Map the cell-type codes back to their labels by constructing a polars
        # Series from the codes, then casting it to an Enum
        cell_type = pl.Series(cell_type_column, cell_type)\
            .cast(pl.Enum(most_common_cell_types.to_list()))
        # Add `cell_type_column` and `cell_type_confidence_column` to obs.
        # If `QC_column` is not None, back-project from QCed cells to all
        # cells, filling with nulls.
        if QC_column is None:
            obs = self._obs.with_columns(cell_type, cell_type_confidence)
        else:
            # noinspection PyTypeChecker
            obs = self._obs.with_columns(
                pl.when(QC_column.name)
                .then(pl.lit(cell_type)
                      .gather(pl.col(QC_column.name).cum_sum() - 1)),
                pl.when(QC_column.name)
                .then(pl.lit(cell_type_confidence)
                      .gather(pl.col(QC_column.name).cum_sum() - 1)))
        # If `min_cell_type_confidence` is not None, filter to cells with
        # cell-type confidence greater than or equal to
        # `min_cell_type_confidence`
        if min_cell_type_confidence is not None:
            if verbose:
                print(f'Filtering to cells with '
                      f'≥{100 * min_cell_type_confidence}% cell-type '
                      f'confidence...')
            obs = obs.with_columns(pl.col(QC_column) &
                                   pl.col(cell_type_confidence_column)
                                   .ge(min_cell_type_confidence))
            if verbose:
                print(f'{obs[QC_column].sum():,} cells remain after filtering '
                      f'to cells with ≥{100 * min_cell_type_confidence}% '
                      f'cell-type confidence.')
        # Return a new SingleCell dataset containing the cell-type labels and
        # confidences
        return SingleCell(X=self._X, obs=obs, var=self._var, obsm=self._obsm,
                          varm=self._varm, uns=self._uns)
    
    def embed(self,
              *,
              QC_column: SingleCellColumn | None = 'passed_QC',
              PC_key: str = 'PCs',
              embedding_key: str | None = 'PaCMAP',
              num_neighbors: int | np.integer = 10,
              num_extra_neighbors: int | np.integer = 50,
              num_index_neighbors: int | np.integer | None = None,
              num_trees: int | np.integer | None = 100,
              num_mid_near_pairs: int | np.integer = 5,
              num_further_pairs: int | np.integer = 20,
              num_iterations: int | np.integer |
                              tuple[int | np.integer, int | np.integer,
                                    int | np.integer] | None = None,
              learning_rate: int | float | np.integer | np.floating = 1.0,
              seed: int | np.integer = 0,
              num_threads: int | np.integer | None = 1,
              verbose: bool = True) -> SingleCell:
        """
        Calculate a two-dimensional embedding of this SingleCell dataset
        suitable for plotting with `plot_embedding()`.
        
        Uses PaCMAP (Pairwise Controlled Manifold Approximation;
        github.com/YingfanWang/PaCMAP; arxiv.org/pdf/2012.04456), a faster UMAP
        alternative that also captures global structure better.
        
        Requires the pyglass nearest-neighbors package. Install it with:
        
        git clone https://github.com/Wainberg/pyglass && \
          cd pyglass && bash build.sh
        
        This function is intended to be run after `PCA()`; by default, it uses
        `obsm['PCs']` as the input to PaCMAP, and stores the output in
        `obsm['PaCMAP']` as a num_cells × 2 NumPy array.
        
        Args:
            QC_column: an optional Boolean column of obs indicating which cells
                       passed QC. Can be a column name, a polars expression, a 
                       polars Series, a 1D NumPy array, or a function that 
                       takes in this SingleCell dataset and returns a polars 
                       Series or 1D NumPy array. Set to None to include all
                       cells. Cells failing QC will be ignored and have their
                       embeddings set to NaN.
            PC_key: the key of obsm containing a NumPy array of the principal
                    components (or harmonized principal components) to embed
            embedding_key: the key of obsm where a NumPy array of the
                           embeddings will be stored
            num_neighbors: the number of nearest neighbors to consider for
                           local structure preservation
            num_extra_neighbors: the number of extra nearest neighbors (on top
                                 of `num_neighbors`) to search for initially,
                                 before pruning to the `num_neighbors` of these
                                 `num_neighbors + num_extra_neighbors` cells
                                 with the smallest scaled distances. For a pair
                                 of cells `i` and `j`, the scaled distance
                                 between `i` and `j` is its squared Euclidean
                                 distance, divided by `i`'s average Euclidean
                                 distance to its 3rd, 4th, and 5th nearest
                                 neighbors, divided by `j`'s average Euclidean
                                 distance to its 3rd, 4th, and 5th nearest
                                 neighbors. Must be a positive integer or 0.
            num_index_neighbors: the number of nearest neighbors to use when
                                 building the index used for the
                                 nearest-neighbors search. Larger values (e.g.
                                 50-100) will lead to more accurate
                                 determination of a cell's true k-nearest
                                 neighbors at the cost of increased runtime.
                                 Defaults to the number of neighbors being
                                 initially searched for, i.e.
                                 `num_neighbors + num_extra_neighbors + 1`,
                                 where the `+ 1` is for the cell itself. We do
                                 not recommend setting this to lower than
                                 `num_neighbors + num_extra_neighbors + 1`.
            num_trees: the number of random projection trees to use when
                       building the index used for nearest-neighbor search.
                       Paradoxically, increasing the number of trees actually
                       reduces runtime (up to a point) by reducing the number
                       of subsequent graph updates required. As a result, we
                       use a default of 100 trees instead of the smaller
                       default of `min(32, 5 + int(round(num_cells ** 0.25)))`
                       used by nndescent and pynndescent (this default can be
                       re-enabled by setting `num_trees=None`).
            num_mid_near_pairs: the number of mid-near pairs to consider for
                                global structure preservation
            num_further_pairs: the number of further pairs to consider for
                               local and global structure preservation
            num_iterations: the number of iterations/epochs to run PaCMAP for.
                            Can be a length-3 tuple of the number of iterations
                            for each of the 3 stages of PaCMAP, or a single
                            integer of the number of iterations for the third
                            stage (in which case the number of iterations for
                            the first two stages will be set to 100).
            learning_rate: the learning rate of the Adam optimizer for PaCMAP
            seed: the random seed to use for nearest-neighbor finding and
                  PaCMAP
            num_threads: the number of cores to run nearest-neighbor finding
                         and PaCMAP on. Set `num_threads=None` to use all
                         available cores (as determined by `os.cpu_count()`).
            verbose: whether to print details of the PaCMAP construction
        
        Returns:
            A new SingleCell dataset with the PaCMAP embedding added as
            `obsm[embedding_key]`.
        
        Note:
            PaCMAP's original implementation assumes generic input data, so it
            initializes the embedding by standardizing the input data, running
            PCA on it, and taking the first two PCs. Because our input data is
            already PCs (or harmonized PCs), we avoid redundancy by omitting
            this step and initializing the embedding with the first two columns
            of our input data, i.e. the first two PCs.
        """
        # noinspection PyUnresolvedReferences
        from nndescent import NNDescent
        # Get `QC_column`, if not None
        if QC_column is not None:
            QC_column = self._get_column(
                'obs', QC_column, 'QC_column', pl.Boolean,
                allow_missing=QC_column == 'passed_QC')
        # Check that `PC_key` is the name of a key in obsm
        check_type(PC_key, 'PC_key', str, 'a string')
        if PC_key not in self._obsm:
            error_message = f'PC_key {PC_key!r} is not a key of obsm'
            if PC_key == 'PCs':
                error_message += \
                    '; did you forget to run PCA() before embed()?'
            raise ValueError(error_message)
        # Get PCs, for QCed cells only if `QC_column` is not None
        if QC_column is None:
            X = self._obsm[PC_key]
        else:
            QCed_NumPy = self._obs[QC_column].to_numpy()
            X = self._obsm[PC_key][QCed_NumPy]
        # Check that `embedding_key` is a string and not already a key in obsm
        check_type(embedding_key, 'embedding_key', str, 'a string')
        if embedding_key in self._obsm:
            error_message = (
                f'embedding_key {embedding_key!r} is already a key of obsm; '
                f'did you already run embed()?')
            raise ValueError(error_message)
        # Check that `num_neighbors`, `num_trees`, `num_mid_near_pairs` and
        # `num_further_pairs` are positive integers (or None, for `num_trees`)
        for variable, variable_name in (
                (num_neighbors, 'num_neighbors'),
                (num_mid_near_pairs, 'num_mid_near_pairs'),
                (num_further_pairs, 'num_further_pairs')):
            check_type(variable, variable_name, int, 'a positive integer')
            check_bounds(variable, variable_name, 1)
        if num_trees is not None:
            check_type(num_trees, 'num_trees', int, 'a positive integer')
            check_bounds(num_trees, 'num_trees', 1)
        # Check that `num_extra_neighbors` is a positive integer or 0
        check_type(num_extra_neighbors, 'num_extra_neighbors', int,
                   'a positive integer or 0')
        check_bounds(num_extra_neighbors, 'num_extra_neighbors', 0)
        # Get the number of total neighbors to search for initially. We want
        # each cell's `num_neighbors + num_extra_neighbors`-nearest neighbors,
        # but we add 1 extra neighbor for the cell itself.
        num_total_neighbors = num_neighbors + num_extra_neighbors + 1
        # Check that `num_index_neighbors` is a positive integer, or None in
        # which case set it to `num_total_neighbors`
        if num_index_neighbors is None:
            num_index_neighbors = num_total_neighbors
        else:
            check_type(num_index_neighbors, 'num_index_neighbors', int,
                       'a positive integer')
            check_bounds(num_index_neighbors, 'num_index_neighbors', 1)
        # Calculate the number of nearest neighbors, if not specified
        num_cells = len(X)
        if num_neighbors is None:
            num_neighbors = 10 if num_cells <= 10_000 else \
                int(round(10 + 15 * (np.log10(num_cells) - 4)))
        if num_cells <= num_neighbors:
            error_message = (
                f'the number of cells ({num_cells:,}) must be greater than '
                f'num_neighbors ({num_neighbors:,})')
            raise ValueError(error_message)
        # Check that `num_iterations` is an integer or length-3 tuple of
        # integers, or None
        if num_iterations is not None:
            check_type(num_iterations, 'num_iterations', (int, tuple),
                       'a positive integer or length-3 tuple of positive '
                       'integers')
            if isinstance(num_iterations, tuple):
                if len(num_iterations) != 3:
                    error_message = (
                        f'num_iterations must be a positive integer or '
                        f'length-3 tuple of positive integers, but has length '
                        f'{len(num_iterations):,}')
                    raise ValueError(error_message)
                for step, step_num_iterations in enumerate(num_iterations):
                    check_type(step_num_iterations,
                               f'num_iterations[{step!r}]', int,
                               'a positive integer')
                    check_bounds(step_num_iterations,
                                 f'num_iterations[{step!r}]', 1)
            else:
                check_bounds(num_iterations, 'num_iterations', 1)
                num_iterations = 100, 100, num_iterations
        else:
            num_iterations = 100, 100, 250
        # Check that `learning_rate` is a positive floating-point number
        check_type(learning_rate, 'learning_rate', (int, float),
                   'a positive number')
        check_bounds(learning_rate, 'learning_rate', 0, left_open=True)
        # Check that `seed` is an integer
        check_type(seed, 'seed', int, 'an integer')
        # Check that `num_threads` is a positive integer or None; if None, set
        # to `os.cpu_count()`
        if num_threads is None:
            num_threads = os.cpu_count()
        else:
            check_type(num_threads, 'num_threads', int, 'a positive integer')
            check_bounds(num_threads, 'num_threads', 1)
        # Check that `verbose` is Boolean
        check_type(verbose, 'verbose', bool, 'Boolean')
        # Define Cython functions
        cython_functions = cython_inline(rf'''
        from cython.parallel cimport threadid, prange
        from libc.math cimport sqrt
        from libc.stdlib cimport malloc, free
        from libc.string cimport memcpy
        
        cdef extern from "limits.h":
            cdef int INT_MAX
        
        cdef int rand(long* state) noexcept nogil:
            cdef long x = state[0]
            state[0] = x * 6364136223846793005L + 1442695040888963407L
            cdef int s = (x ^ (x >> 18)) >> 27
            cdef int rot = x >> 59
            return (s >> rot) | (s << ((-rot) & 31))
        
        cdef long srand(long seed) noexcept nogil:
            cdef long state = seed + 1442695040888963407L
            rand(&state)
            return state
        
        cdef int randint(int bound, long* state) noexcept nogil:
            cdef int r, threshold = -bound % bound
            while True:
                r = rand(state)
                if r >= threshold:
                    return r % bound

        cdef void quicksort(const double[::1] arr,
                            int[::1] indices,
                            int left,
                            int right) noexcept nogil:
            cdef double pivot_value
            cdef int pivot_index, mid, i, temp
    
            while left < right:
                mid = left + (right - left) // 2
                if arr[indices[mid]] < arr[indices[left]]:
                    temp = indices[left]
                    indices[left] = indices[mid]
                    indices[mid] = temp
                if arr[indices[right]] < arr[indices[left]]:
                    temp = indices[left]
                    indices[left] = indices[right]
                    indices[right] = temp
                if arr[indices[right]] < arr[indices[mid]]:
                    temp = indices[mid]
                    indices[mid] = indices[right]
                    indices[right] = temp
    
                pivot_value = arr[indices[mid]]
                temp = indices[mid]
                indices[mid] = indices[right]
                indices[right] = temp
                pivot_index = left
    
                for i in range(left, right):
                    if arr[indices[i]] < pivot_value:
                        temp = indices[i]
                        indices[i] = indices[pivot_index]
                        indices[pivot_index] = temp
                        pivot_index += 1
    
                temp = indices[right]
                indices[right] = indices[pivot_index]
                indices[pivot_index] = temp
    
                if pivot_index - left < right - pivot_index:
                    quicksort(arr, indices, left, pivot_index - 1)
                    left = pivot_index + 1
                else:
                    quicksort(arr, indices, pivot_index + 1, right)
                    right = pivot_index - 1
    
        cdef inline void argsort(const double[::1] arr,
                                 int[::1] indices) noexcept nogil:
            cdef int i
            for i in range(indices.shape[0]):
                indices[i] = i
            quicksort(arr, indices, 0, indices.shape[0] - 1)
        
        def remove_self_neighbors(int[:, ::1] neighbors,
                                  const unsigned num_threads):
            cdef int i, j
            for i in {prange('neighbors.shape[0]', num_threads)}:
                # If the cell is its own nearest neighbor (almost always), skip
                if neighbors[i, 0] == i:
                    continue
                # Find the position where the cell is listed as its own
                # self-neighbor
                for j in range(1, neighbors.shape[1]):
                    if neighbors[i, j] == i:
                        break
                # Shift all neighbors before it to the right, overwriting it
                while j > 0:
                    neighbors[i, j] = neighbors[i, j - 1]
                    j = j - 1
        
        def get_scaled_distances(const double[:, ::1] X,
                                 const int[:, :] neighbors,
                                 double[:, ::1] scaled_distances,
                                 const unsigned num_threads):
            cdef int i, j, k
            cdef double* sig = \
                <double*> malloc(scaled_distances.shape[0] * sizeof(double))
            try:
                for i in {prange('scaled_distances.shape[0]', num_threads)}:
                    for j in range(scaled_distances.shape[1]):
                        scaled_distances[i, j] = 0
                        for k in range(X.shape[1]):
                            scaled_distances[i, j] = scaled_distances[i, j] + \
                                                     (X[i, k] - X[j, k]) ** 2
                
                for i in {prange('scaled_distances.shape[0]', num_threads)}:
                    sig[i] = (sqrt(scaled_distances[i, 3]) +
                              sqrt(scaled_distances[i, 4]) +
                              sqrt(scaled_distances[i, 5])) / 3
                    if sig[i] < 1e-10:
                        sig[i] = 1e-10
                
                for i in {prange('scaled_distances.shape[0]', num_threads)}:
                    for j in range(scaled_distances.shape[1]):
                        scaled_distances[i, j] = scaled_distances[i, j] / \
                            sig[i] / sig[neighbors[i, j]]
            finally:
                free(sig)
    
        def get_neighbor_pairs(const double[:, ::1] X,
                               const double[:, ::1] scaled_distances,
                               const int[:, :] neighbors,
                               int[:, ::1] neighbor_pairs,
                               const unsigned num_threads):
            cdef int i, j, thread_id
            cdef int num_neighbors = neighbor_pairs.shape[1]
            cdef int num_total_neighbors = scaled_distances.shape[1]
            cdef int* indices_buffer = <int*> malloc(
                num_total_neighbors * num_threads * sizeof(int))
            cdef int[::1] indices = \
                <int[:num_total_neighbors * num_threads]> indices_buffer
            
            try:
                if num_threads == 1:
                    for i in range(neighbor_pairs.shape[0]):
                        argsort(scaled_distances[i], indices)
                        for j in range(num_neighbors):
                            neighbor_pairs[i, j] = neighbors[i, indices[j]]
                else:
                    for i in prange(X.shape[0], num_threads=num_threads,
                                    nogil=True):
                        thread_id = threadid()
                        argsort(scaled_distances[i],
                                indices[thread_id * num_total_neighbors:
                                        thread_id * num_total_neighbors +
                                        num_total_neighbors])
                        for j in range(num_neighbors):
                            neighbor_pairs[i, j] = neighbors[i, indices[
                                thread_id * num_total_neighbors + j]]
            finally:
                free(indices_buffer)
        
        def sample_mid_near_pairs(const double[:, ::1] X,
                                  int[:, ::1] mid_near_pairs,
                                  const int seed,
                                  const unsigned num_threads):
            cdef int i, j, k, l, thread_id, n = X.shape[0], \
                closest_cell = -1, second_closest_cell = -1
            cdef double squared_distance, smallest, second_smallest
            cdef long state
            cdef int* sampled = <int*> malloc(6 * num_threads * sizeof(int))
            try:
                if num_threads == 1:
                    for i in range(n):
                        state = srand(seed + i)
                        for j in range(mid_near_pairs.shape[1]):
                            # Randomly sample 6 cells (which are not the
                            # current cell) and select the 2nd-closest
                            smallest = INT_MAX
                            second_smallest = INT_MAX
                            for k in range(6):
                                while True:
                                    # Sample a random cell...
                                    sampled[k] = randint(n, &state)
                                    # ...that is not this cell...
                                    if sampled[k] == i:
                                        continue
                                    # ...nor a previously sampled cell
                                    for l in range(k):
                                        if sampled[k] == sampled[l]:
                                            break
                                    else:
                                        break
                            for k in range(6):
                                squared_distance = 0
                                for l in range(X.shape[1]):
                                    squared_distance = squared_distance + \
                                        (X[i, l] - X[sampled[k], l]) ** 2
                                if squared_distance < smallest:
                                    second_smallest = smallest
                                    second_closest_cell = closest_cell
                                    smallest = squared_distance
                                    closest_cell = sampled[k]
                                elif squared_distance < second_smallest:
                                    second_smallest = squared_distance
                                    second_closest_cell = sampled[k]
                            mid_near_pairs[i, j] = second_closest_cell
                else:
                    for i in prange(n, num_threads=num_threads, nogil=True):
                        thread_id = threadid()
                        state = srand(seed + i)
                        for j in range(mid_near_pairs.shape[1]):
                            smallest = INT_MAX
                            second_smallest = INT_MAX
                            for k in range(6 * thread_id, 6 * thread_id + 6):
                                while True:
                                    sampled[k] = randint(n, &state)
                                    if sampled[k] == i:
                                        continue
                                    for l in range(6 * thread_id, k):
                                        if sampled[k] == sampled[l]:
                                            break
                                    else:
                                        break
                            for k in range(6 * thread_id, 6 * thread_id + 6):
                                squared_distance = 0
                                for l in range(X.shape[1]):
                                    squared_distance = squared_distance + \
                                        (X[i, l] - X[sampled[k], l]) ** 2
                                if squared_distance < smallest:
                                    second_smallest = smallest
                                    second_closest_cell = closest_cell
                                    smallest = squared_distance
                                    closest_cell = sampled[k]
                                elif squared_distance < second_smallest:
                                    second_smallest = squared_distance
                                    second_closest_cell = sampled[k]
                            mid_near_pairs[i, j] = second_closest_cell
            finally:
                free(sampled)
        
        def sample_further_pairs(const double[:, ::1] X,
                                 const int[:, ::1] neighbor_pairs,
                                 int[:, ::1] further_pairs,
                                 const int seed,
                                 const unsigned num_threads):
            """Sample Further pairs using the given seed."""
            cdef int i, j, k, n = X.shape[0], further_pair_index
            cdef long state
            for i in {prange('n', num_threads)}:
                state = srand(seed + i)
                for j in range(further_pairs.shape[1]):
                    while True:
                        # Sample a random cell...
                        further_pair_index = randint(n, &state)
                        # ...that is not this cell...
                        if further_pair_index == i:
                            continue
                        # ...nor one of its nearest neighbors...
                        for k in range(neighbor_pairs.shape[1]):
                            if further_pair_index == neighbor_pairs[i, k]:
                                break
                        else:
                            # ...nor a previously sampled cell
                            for k in range(j):
                                if further_pair_index == further_pairs[i, k]:
                                    break
                            else:
                                break
                    further_pairs[i, j] = further_pair_index
        
        def reformat_for_parallel(const int[:, ::1] pairs,
                                  int[::1] pair_indices, int[::1] pair_indptr):
            cdef int i, j, k, dest_index
            cdef int *dest_indices
            # Tabulate how often each cell appears in pairs; at a minimum, it
            # will appear `pairs.shape[1]` times (i.e. the number of
            # neighbors), as the `i` in the pair, but it will also appear a
            # variable number of times as the `j` in the pair.
            pair_indptr[0] = 0
            pair_indptr[1:] = pairs.shape[1]
            for i in range(pairs.shape[0]):
                for k in range(pairs.shape[1]):
                    j = pairs[i, k]
                    pair_indptr[j + 1] += 1
            # Take the cumulative sum of the values in `pair_indptr`
            for i in range(2, pair_indptr.shape[0]):
                pair_indptr[i] += pair_indptr[i - 1]
            # Now that we know how many pairs each cell is a part of, do a
            # second pass over `pairs` to populate `pair_indices` with the
            # pairs' indices. Use a temporary buffer, `dest_indices`, to keep
            # track of the index within `pair_indptr` to write each cell's next
            # pair to.
            dest_indices = <int*> malloc(pairs.shape[0] * sizeof(int))
            memcpy(dest_indices, &pair_indptr[0], pairs.shape[0] * sizeof(int))
            try:
                for i in range(pairs.shape[0]):
                    for k in range(pairs.shape[1]):
                        j = pairs[i, k]
                        pair_indices[dest_indices[i]] = j
                        pair_indices[dest_indices[j]] = i
                        dest_indices[i] += 1
                        dest_indices[j] += 1
            finally:
                free(dest_indices)

        def get_gradients(const double[:, ::1] embedding,
                          const int[:, ::1] neighbor_pairs,
                          const int[:, ::1] mid_near_pairs,
                          const int[:, ::1] further_pairs,
                          const double w_neighbors,
                          const double w_mid_near,
                          double[:, ::1] gradients):
            cdef int i, j, k
            cdef double distance_ij, embedding_ij_0, embedding_ij_1, w
            gradients[:] = 0
            # Nearest-neighbor pairs
            for i in range(neighbor_pairs.shape[0]):
                for k in range(neighbor_pairs.shape[1]):
                    j = neighbor_pairs[i, k]
                    embedding_ij_0 = embedding[i, 0] - embedding[j, 0]
                    embedding_ij_1 = embedding[i, 1] - embedding[j, 1]
                    distance_ij = 1 + embedding_ij_0 ** 2 + embedding_ij_1 ** 2
                    w = w_neighbors * (20 / (10 + distance_ij) ** 2)
                    gradients[i, 0] += w * embedding_ij_0
                    gradients[j, 0] -= w * embedding_ij_0
                    gradients[i, 1] += w * embedding_ij_1
                    gradients[j, 1] -= w * embedding_ij_1
            # Mid-near pairs
            for i in range(mid_near_pairs.shape[0]):
                for k in range(mid_near_pairs.shape[1]):
                    j = mid_near_pairs[i, k]
                    embedding_ij_0 = embedding[i, 0] - embedding[j, 0]
                    embedding_ij_1 = embedding[i, 1] - embedding[j, 1]
                    distance_ij = 1 + embedding_ij_0 ** 2 + embedding_ij_1 ** 2
                    w = w_mid_near * 20000 / (10000 + distance_ij) ** 2
                    gradients[i, 0] += w * embedding_ij_0
                    gradients[j, 0] -= w * embedding_ij_0
                    gradients[i, 1] += w * embedding_ij_1
                    gradients[j, 1] -= w * embedding_ij_1
            # Further pairs
            for i in range(further_pairs.shape[0]):
                for k in range(further_pairs.shape[1]):
                    j = further_pairs[i, k]
                    embedding_ij_0 = embedding[i, 0] - embedding[j, 0]
                    embedding_ij_1 = embedding[i, 1] - embedding[j, 1]
                    distance_ij = 1 + embedding_ij_0 ** 2 + embedding_ij_1 ** 2
                    w = 2 / (1 + distance_ij) ** 2
                    gradients[i, 0] -= w * embedding_ij_0
                    gradients[j, 0] += w * embedding_ij_0
                    gradients[i, 1] -= w * embedding_ij_1
                    gradients[j, 1] += w * embedding_ij_1
        
        def get_gradients_parallel(const double[:, ::1] embedding,
                                  const int[::1] neighbor_pair_indices,
                                  const int[::1] neighbor_pair_indptr,
                                  const int[::1] mid_near_pair_indices,
                                  const int[::1] mid_near_pair_indptr,
                                  const int[::1] further_pair_indices,
                                  const int[::1] further_pair_indptr,
                                  const double w_neighbors,
                                  const double w_mid_near,
                                  double[:, ::1] gradients,
                                  const int num_threads):
            cdef int i, j, k
            cdef double distance_ij, embedding_ij_0, embedding_ij_1, w
            for i in prange(embedding.shape[0], nogil=True,
                            num_threads=num_threads):
                gradients[i, 0] = 0
                gradients[i, 1] = 0
                # Nearest-neighbor pairs
                for k in range(neighbor_pair_indptr[i],
                               neighbor_pair_indptr[i + 1]):
                    j = neighbor_pair_indices[k]
                    embedding_ij_0 = embedding[i, 0] - embedding[j, 0]
                    embedding_ij_1 = embedding[i, 1] - embedding[j, 1]
                    distance_ij = 1 + embedding_ij_0 ** 2 + embedding_ij_1 ** 2
                    w = w_neighbors * (20 / (10 + distance_ij) ** 2)
                    gradients[i, 0] = gradients[i, 0] + w * embedding_ij_0
                    gradients[i, 1] = gradients[i, 1] + w * embedding_ij_1
                # Mid-near pairs
                for k in range(mid_near_pair_indptr[i],
                               mid_near_pair_indptr[i + 1]):
                    j = mid_near_pair_indices[k]
                    embedding_ij_0 = embedding[i, 0] - embedding[j, 0]
                    embedding_ij_1 = embedding[i, 1] - embedding[j, 1]
                    distance_ij = 1 + embedding_ij_0 ** 2 + embedding_ij_1 ** 2
                    w = w_mid_near * 20000 / (10000 + distance_ij) ** 2
                    gradients[i, 0] = gradients[i, 0] + w * embedding_ij_0
                    gradients[i, 1] = gradients[i, 1] + w * embedding_ij_1
                # Further pairs
                for k in range(further_pair_indptr[i],
                               further_pair_indptr[i + 1]):
                    j = further_pair_indices[k]
                    embedding_ij_0 = embedding[i, 0] - embedding[j, 0]
                    embedding_ij_1 = embedding[i, 1] - embedding[j, 1]
                    distance_ij = 1 + embedding_ij_0 ** 2 + embedding_ij_1 ** 2
                    w = 2 / (1 + distance_ij) ** 2
                    gradients[i, 0] = gradients[i, 0] - w * embedding_ij_0
                    gradients[i, 1] = gradients[i, 1] - w * embedding_ij_1
        
        def update_embedding_adam(double[:, ::1] embedding,
                                  const double[:, ::1] gradients,
                                  double[:, ::1] momentum,
                                  double[:, ::1] velocity,
                                  const double beta1,
                                  const double beta2,
                                  double learning_rate,
                                  const int iteration,
                                  const unsigned num_threads):
            cdef int i
            learning_rate = \
                learning_rate * sqrt(1 - beta2 ** (iteration + 1)) / \
                (1 - beta1 ** (iteration + 1))
            for i in {prange('embedding.shape[0]', num_threads)}:
                momentum[i, 0] += \
                    (1 - beta1) * (gradients[i, 0] - momentum[i, 0])
                velocity[i, 0] += \
                    (1 - beta2) * (gradients[i, 0] ** 2 - velocity[i, 0])
                embedding[i, 0] -= learning_rate * momentum[i, 0] / \
                                   (sqrt(velocity[i, 0]) + 1e-7)
                momentum[i, 1] += \
                    (1 - beta1) * (gradients[i, 1] - momentum[i, 1])
                velocity[i, 1] += \
                    (1 - beta2) * (gradients[i, 1] ** 2 - velocity[i, 1])
                embedding[i, 1] -= learning_rate * momentum[i, 1] / \
                                   (sqrt(velocity[i, 1]) + 1e-7)
            ''')
        remove_self_neighbors = cython_functions['remove_self_neighbors']
        get_scaled_distances = cython_functions['get_scaled_distances']
        get_neighbor_pairs = cython_functions['get_neighbor_pairs']
        sample_mid_near_pairs = cython_functions['sample_mid_near_pairs']
        sample_further_pairs = cython_functions['sample_further_pairs']
        update_embedding_adam = cython_functions['update_embedding_adam']
        # Calculate each cell's `num_neighbors + num_extra_neighbors`-nearest
        # neighbors. (`num_total_neighbors` is
        # `num_neighbors + num_extra_neighbors + 1`, where the `+ 1` is for the
        # cell itself. We exclude the cell itself below.)
        X_float32 = X.astype(np.float32, copy=False)
        neighbors = \
            NNDescent(X_float32, n_neighbors=num_index_neighbors,
                      n_trees=num_trees, seed=seed, n_threads=num_threads,
                      verbose=verbose)\
                .query(X_float32, k=num_total_neighbors)[0]
        if verbose:
            percent = (neighbors[:, 0] == range(num_cells)).mean()
            print(f'{100 * percent:.3f}% of cells are correctly detected '
                  f'as their own nearest neighbors (a measure of the '
                  f'quality of the k-nearest neighbors search)')
        # Remove self-neighbors from each cell's list of nearest neighbors.
        # These are almost always in the 0th column, but occasionally later due
        # to the inaccuracy of the nearest-neighbors search. This leaves us
        # with `num_neighbors + num_extra_neighbors` nearest neighbors.
        remove_self_neighbors(neighbors, num_threads)
        neighbors = neighbors[:, 1:]
        # Get scaled distances between each cell and its nearest neighbors
        scaled_distances = np.empty_like(neighbors, dtype=float)
        get_scaled_distances(X, neighbors, scaled_distances, num_threads)
        # Select the `num_neighbors` of the `num_total_neighbors`
        # nearest-neighbor pairs with the lowest scaled distances
        neighbor_pairs = np.empty((num_cells, num_neighbors), dtype=np.int32)
        get_neighbor_pairs(X, scaled_distances, neighbors, neighbor_pairs,
                           num_threads)
        del scaled_distances, neighbors
        # Sample mid-near pairs
        mid_near_pairs = np.empty((num_cells, num_mid_near_pairs),
                                  dtype=np.int32)
        sample_mid_near_pairs(X, mid_near_pairs, seed, num_threads)
        # Sample further pairs
        further_pairs = np.empty((num_cells, num_further_pairs),
                                 dtype=np.int32)
        sample_further_pairs(X, neighbor_pairs, further_pairs,
                             seed + mid_near_pairs.size, num_threads)
        # If multithreaded, reformat the three lists of pairs to allow
        # deterministic parallelism. Specifically, transform pairs of cell
        # indices from the original format of a 2D array `pairs` where
        # `pairs[i]` contains all js for which (i, j) is a pair, to a pair of
        # 1D arrays `pair_indices` and `pair_indptr` forming a sparse matrix,
        # where `pair_indices[pair_indptr[i]:pair_indptr[i + 1]]` contains all
        # js for which (i, j) is a pair or (j, i) is a pair. `pair_indices`
        # must have length `2 * pairs.size`, since each pair will appear twice,
        # once for (i, j) and once for (j, i). `pair_indptr` must have length
        # equal to the number of cells plus one, just like for scipy sparse
        # matrices.
        if num_threads > 1:
            reformat_for_parallel = cython_functions['reformat_for_parallel']
            
            neighbor_pair_indices = np.empty(2 * neighbor_pairs.size,
                                              dtype=np.int32)
            neighbor_pair_indptr = np.empty(num_cells + 1, dtype=np.int32)
            reformat_for_parallel(neighbor_pairs, neighbor_pair_indices,
                                  neighbor_pair_indptr)
            del neighbor_pairs
            mid_near_pair_indices = \
                np.empty(2 * mid_near_pairs.size, dtype=np.int32)
            mid_near_pair_indptr = np.empty(num_cells + 1, dtype=np.int32)
            reformat_for_parallel(mid_near_pairs, mid_near_pair_indices,
                                  mid_near_pair_indptr)
            del mid_near_pairs
            further_pair_indices = \
                np.empty(2 * further_pairs.size, dtype=np.int32)
            further_pair_indptr = np.empty(num_cells + 1, dtype=np.int32)
            reformat_for_parallel(further_pairs, further_pair_indices,
                                  further_pair_indptr)
            del further_pairs
            get_gradients = cython_functions['get_gradients_parallel']
        else:
            get_gradients = cython_functions['get_gradients']
        # Initialize the embedding, gradients, and other optimizer parameters
        embedding = 0.01 * X[:, :2]
        gradients = np.zeros_like(embedding, dtype=float)
        momentum = np.zeros_like(embedding, dtype=float)
        velocity = np.zeros_like(embedding, dtype=float)
        w_mid_near_init = 1000
        beta1 = 0.9
        beta2 = 0.999
        # Optimize the embedding
        for iteration in range(sum(num_iterations)):
            num_phase_1_iterations, num_phase_2_iterations = num_iterations[:2]
            if iteration < num_phase_1_iterations:
                w_mid_near = \
                    (1 - iteration / num_phase_1_iterations) * \
                    w_mid_near_init + iteration / num_phase_1_iterations * 3
                w_neighbors = 2
            elif iteration < num_phase_1_iterations + num_phase_2_iterations:
                w_mid_near = 3
                w_neighbors = 3
            else:
                w_mid_near = 0
                w_neighbors = 1
            # Calculate gradients
            if num_threads == 1:
                # noinspection PyUnboundLocalVariable
                get_gradients(embedding, neighbor_pairs, mid_near_pairs,
                              further_pairs, w_neighbors, w_mid_near,
                              gradients)
            else:
                # noinspection PyUnboundLocalVariable
                get_gradients(embedding, neighbor_pair_indices,
                              neighbor_pair_indptr, mid_near_pair_indices,
                              mid_near_pair_indptr, further_pair_indices,
                              further_pair_indptr, w_neighbors, w_mid_near,
                              gradients, num_threads)
            # Update the embedding based on the gradients, via the Adam
            # optimizer
            update_embedding_adam(embedding, gradients, momentum, velocity,
                                  beta1, beta2, learning_rate, iteration,
                                  num_threads)
        # If `QC_column` is not None, back-project from QCed cells to all
        # cells, filling with NaN
        if QC_column is not None:
            embedding_QCed = embedding
            embedding = np.full((len(self), embedding_QCed.shape[1]), np.nan)
            # noinspection PyUnboundLocalVariable
            embedding[QCed_NumPy] = embedding_QCed
        # noinspection PyTypeChecker
        return SingleCell(X=self._X, obs=self._obs, var=self._var,
                          obsm=self._obsm | {embedding_key: embedding},
                          varm=self._varm, uns=self._uns)
    
    # noinspection PyUnresolvedReferences
    def plot_embedding(
            self,
            color_column: str | None,
            filename: str | Path | None = None,
            *,
            cells_to_plot_column: str | None = 'passed_QC',
            embedding_key: str = 'PaCMAP',
            ax: 'Axes' | None = None,
            point_size: int | float | np.integer | np.floating | str |
                        None = None,
            sort_by_frequency: bool = False,
            palette: str | 'Colormap' | dict[Any, Color] = None,
            palette_kwargs: dict[str, Any] | None = None,
            default_color: Color = 'lightgray',
            scatter_kwargs: dict[str, Any] | None = None,
            label: bool = False,
            label_kwargs: dict[str, Any] | None = None,
            legend: bool = True,
            legend_kwargs: dict[str, Any] | None = None,
            colorbar: bool = True,
            colorbar_kwargs: dict[str, Any] | None = None,
            title: str | None = None,
            title_kwargs: dict[str, Any] | None = None,
            xlabel: str | None = 'Component 1',
            xlabel_kwargs: dict[str, Any] | None = None,
            ylabel: str | None = 'Component 2',
            ylabel_kwargs: dict[str, Any] | None = None,
            xlim: tuple[int | float | np.integer | np.floating,
                        int | float | np.integer | np.floating] | None = None,
            ylim: tuple[int | float | np.integer | np.floating,
                        int | float | np.integer | np.floating] | None = None,
            despine: bool = True,
            savefig_kwargs: dict[str, Any] | None = None) -> None:
        """
        Plot an embedding created by embed(), using Matplotlib.
        
        Requires the colorspacious package. Install via:
        mamba install -y colorspacious
        
        Args:
            filename: the file to save to. If None, generate the plot but do
                      not save it, which allows it to be shown interactively or
                      modified further (e.g. by adding a title or axis labels)
                      before saving.
            color_column: an optional column of obs indicating how to color
                          each cell in the plot. Can be a column name, a polars 
                          expression, a polars Series, a 1D NumPy array, or a 
                          function that takes in this SingleCell dataset and 
                          returns a polars Series or 1D NumPy array. Can be
                          discrete (e.g. cell-type labels), specified as a
                          String/Categorical/Enum column, or quantitative (e.g.
                          the number of UMIs per cell), specified as an
                          integer/floating-point column. Missing (null) cells
                          will be plotted with the color `default_color`. Set
                          to None to use `default_color` for all cells.
            cells_to_plot_column: an optional Boolean column of obs indicating
                                  which cells to plot. Can be a column name, a
                                  polars expression, a polars Series, a 1D
                                  NumPy array, or a function that takes in this
                                  SingleCell dataset and returns a polars
                                  Series or 1D NumPy array. Set to None to plot
                                  all cells passing QC.
            embedding_key: the key of obsm containing a NumPy array of the
                           embedding to plot
            ax: the Matplotlib axes to save the plot onto; if None, create a
                new figure with Matpotlib's constrained layout and plot onto it
            point_size: the size of the points for each cell; defaults to
                        30,000 divided by the number of cells, one quarter of
                        scanpy's default. Can be a single number, or the name
                        of a column of obs to make each point a different size.
            sort_by_frequency: if True, assign colors and sort the legend in
                               order of decreasing frequency; if False (the
                               default), use natural sorted order
                               (en.wikipedia.org/wiki/Natural_sort_order).
                               Cannot be True unless `palette` is None and
                               `color_column` is discrete; if `palette` is
                               not None, the plot order is determined by the
                               order of the keys in `palette`.
            palette: a string or Colormap object indicating the Matplotlib
                     colormap to use; or, if `color_column` is discrete, a
                     dictionary mapping values in `color_column` to Matplotlib
                     colors (cells with values of `color_column` that are not
                     in the dictionary will be plotted in the color
                     `default_color`). Defaults to `plt.rcParams['image.cmap']`
                     (`'viridis'` by default) if `color_column` is continous,
                     or the colors from `generate_palette()` if `color_column`
                     is discrete (with colors assigned in decreasing order of
                     frequency). Cannot be specified if `color_column` is None.
            palette_kwargs: a dictionary of keyword arguments to be passed to
                            `generate_palette()`. Can only be specified when
                            `color_column` is discrete and `palette` is None.
            default_color: the default color to plot cells in when
                           `color_column` is None, or when certain cells have
                           missing (null) values for `color_column`, or when
                           `palette` is a dictionary and some cells have values
                           of `color_column` that are not in the dictionary
            scatter_kwargs: a dictionary of keyword arguments to be passed to
                            `ax.scatter()`, such as:
                            - `rasterized`: whether to convert the scatter plot
                              points to a raster (bitmap) image when saving to
                              a vector format like PDF. Defaults to True,
                              instead of the Matplotlib default of False.
                            - `marker`: the shape to use for plotting each cell
                            - `norm`, `vmin`, and `vmax`: control how the
                              numbers in `color_column` are converted to
                              colors, if `color_column` is numeric
                            - `alpha`: the transparency of each point
                            - `linewidths` and `edgecolors`: the width and
                              color of the borders around each marker. These
                              are absent by default (`linewidths=0`), unlike
                              Matplotlib's default. Both arguments can be
                              either single values or sequences.
                            - `zorder`: the order in which the cells are
                              plotted, with higher values appearing on top of
                              lower ones.
                            Specifying `s`, `c`/`color`, or `cmap` will raise
                            an error, since these arguments conflict with the
                            `point_size`, `color_column`, and `colormap`
                            arguments, respectively.
            label: whether to label cells with each distinct value of
                   `color_column`. Labels will be placed at the median x and y
                   position of the points with that color. Can only be True
                   when `color_column` is discrete. When set to True, you may
                   also want to set `legend=False` to avoid redundancy.
            label_kwargs: a dictionary of keyword arguments to be passed to
                          `ax.text()` when adding labels to control the text
                          properties, such as:
                           - `color` and `size` to modify the text color/size
                           - `verticalalignment` and `horizontalalignment` to
                             control vertical and horizontal alignment. By
                             default, unlike Matplotlib's default, these are
                             both set to `'center'`.
                           - `path_effects` to set properties for the border
                             around the text. By default, set to
                             `matplotlib.patheffects.withStroke(
                                  linewidth=3, foreground='white', alpha=0.75)`
                             instead of Matplotlib's default of None, to put a
                             semi-transparent white border around the labels
                             for better contrast.
                          Can only be specified when `label=True`.
            legend: whether to add a legend for each value in `color_column`.
                    Ignored unless `color_column` is discrete.
            legend_kwargs: a dictionary of keyword arguments to be passed to
                           `ax.legend()` to modify the legend, such as:
                           - `loc`, `bbox_to_anchor`, and `bbox_transform` to
                             set its location. By default, `loc` is set to
                             `center left` and `bbox_to_anchor` to `(1, 0.5)`
                             to put the legend to the right of the plot,
                             anchored at the middle.
                           - `ncols` to set its number of columns. By
                             default, set to
                             `obs[color_column].n_unique() // 16 + 1` to have
                             at most 16 items per column.
                           - `prop`, `fontsize`, and `labelcolor` to set its
                             font properties
                           - `facecolor` and `framealpha` to set its background
                             color and transparency
                           - `frameon=True` or `edgecolor` to add or color its
                             border (`frameon` is False by default, unlike
                             Matplotlib's default of True)
                           - `title` to add a legend title
                           Can only be specified when `color_column` is
                           discrete and `legend=True`.
            colorbar: whether to add a colorbar. Ignored unless `color_column`
                      is quantitative.
            colorbar_kwargs: a dictionary of keyword arguments to be passed to
                             `ax.colorbar()`, such as:
                             - `location`: `'left'`, `'right'`, `'top'`, or
                               `'bottom'`
                             - `orientation`: `'vertical'` or `'horizontal'`
                             - `fraction`: the fraction of the axes to
                               allocate to the colorbar (default: 0.15)
                             - `shrink`: the fraction to multiply the size of
                               the colorbar by (default: 0.5, unlike
                               Matplotlib's default of 1)
                             - `aspect`: the ratio of the colorbar's long to
                               short dimensions (default: 20)
                             - `pad`: the fraction of the axes between the
                               colorbar and the rest of the figure (default:
                               0.05 if vertical, 0.15 if horizontal)
                             Can only be specified when `color_column` is
                             quantitative and `colorbar=True`.
            title: the title of the plot, or None to not add a title
            title_kwargs: a dictionary of keyword arguments to be passed to
                          ax.title() to modify the title; see `label_kwargs`
                          for examples
            xlabel: the x-axis label, or None to not label the x-axis
            xlabel_kwargs: a dictionary of keyword arguments to be passed to
                           ax.set_xlabel() to modify the x-axis label
            ylabel: the y-axis label, or None to not label the y-axis
            ylabel_kwargs: a dictionary of keyword arguments to be passed to
                           ay.set_ylabel() to modify the y-axis label
            xlim: a length-2 tuple of the left and right x-axis limits, or None
                  to set the limits based on the data
            ylim: a length-2 tuple of the bottom and top y-axis limits, or None
                  to set the limits based on the data
            despine: whether to remove the top and right spines (borders of the
                     plot area) from the plot
            savefig_kwargs: a dictionary of keyword arguments to be passed to
                            `plt.savefig()`, such as:
                            - `dpi`: defaults to 300 instead of Matplotlib's
                              default of 150
                            - `bbox_inches`: the bounding box of the portion of
                              the figure to save; defaults to `'tight'` (crop
                              out any blank borders) instead of Matplotlib's
                              default of None (save the entire figure)
                            - `pad_inches`: the number of inches of padding to
                              add on each of the four sides of the figure when
                              saving. Defaults to `'layout'` (use the padding
                              from the constrained layout engine, when `ax` is
                              not None) instead of Matplotlib's default of 0.1.
                            - `transparent`: whether to save with a transparent
                              background; defaults to True if saving to a PDF
                              (i.e. when `filename` ends with `'.pdf'`) and
                              False otherwise, instead of Matplotlib's default
                              of always being False.
                            Can only be specified when `filename` is not None.
        """
        import matplotlib.pyplot as plt
        # If `color_column` is not None, check that it either discrete
        # (Categorical, Enum, or String) or quantitative (integer or
        # floating-point). If discrete, require at least two distinct values.
        if color_column is not None:
            color_column = self._get_column(
                'obs', color_column, 'color_column',
                (pl.Categorical, pl.Enum, pl.String, 'integer',
                 'floating-point'), allow_null=True)
            unique_color_labels = color_column.unique()
            dtype = color_column.dtype
            discrete = dtype in (pl.Categorical, pl.Enum, pl.String)
            if discrete and len(unique_color_labels) == 1:
                error_message = (
                    f'color_column {color_column!r} must have at least two '
                    f'distinct values when its data type is '
                    f'{dtype.base_type()!r}')
                raise ValueError(error_message)
        # If `filename` is not None, check that it is a string or pathlib.Path
        # and that its base directory exists; if `filename` is None, make sure
        # `savefig_kwargs` is also None
        if filename is not None:
            check_type(filename, 'filename', (str, Path),
                       'a string or pathlib.Path')
            directory = os.path.dirname(filename)
            if directory and not os.path.isdir(directory):
                error_message = (
                    f'{filename} refers to a file in the directory '
                    f'{directory!r}, but this directory does not exist')
                raise NotADirectoryError(error_message)
            filename = str(filename)
        elif savefig_kwargs is not None:
            error_message = 'savefig_kwargs must be None when filename is None'
            raise ValueError(error_message)
        # Check that `embedding_key` is the name of a key in obsm
        check_type(embedding_key, 'embedding_key', str, 'a string')
        if embedding_key not in self._obsm:
            error_message = (
                f'embedding_key {embedding_key!r} is not a key of obsm; '
                f'did you forget to run embed() before plot_embedding()?')
            raise ValueError(error_message)
        # Check that the embedding `embedding_key` references is 2D.
        embedding = self._obsm[embedding_key]
        if embedding.shape[1] != 2:
            error_message = (
                f'the embedding at obsm[{embedding_key!r}] is '
                f'{embedding.shape[1]:,}-dimensional, but must be '
                f'2-dimensional to be plotted')
            raise ValueError(error_message)
        # If `cells_to_plot_column` is not None, subset to these cells
        if cells_to_plot_column is not None:
            cells_to_plot_column = self._get_column(
                'obs', cells_to_plot_column, 'cells_to_plot_column',
                pl.Boolean,
                custom_error='cells_to_plot_column {} is not a column of obs; '
                             'set cells_to_plot_column=None to include all '
                             'cells')
            embedding = embedding[cells_to_plot_column.to_numpy()]
            if color_column is not None:
                # noinspection PyUnboundLocalVariable
                color_labels = color_labels.filter(cells_to_plot_column)
                unique_color_labels = color_labels.unique()
        # Check that the embedding does not contain NaNs
        if np.isnan(embedding).any():
            error_message = \
                f'the embedding at obsm[{embedding_key!r}] contains NaNs; '
            if cells_to_plot_column is None and QC_column is not None:
                error_message += (
                    'did you forget to set QC_column to None in embed(), to '
                    'match the fact that you set cells_to_plot_column to '
                    'None in plot_embedding()?')
            else:
                error_message += (
                    'does your cells_to_plot_column contain cells that were '
                    'excluded by the QC_column used in embed()?')
            raise ValueError(error_message)
        # For each of the kwargs arguments, if the argument is not None, check
        # that it is a dictionary and that all its keys are strings.
        for kwargs, kwargs_name in (
                (palette_kwargs, 'palette_kwargs'),
                (scatter_kwargs, 'scatter_kwargs'),
                (label_kwargs, 'label_kwargs'),
                (legend_kwargs, 'legend_kwargs'),
                (colorbar_kwargs, 'colorbar_kwargs'),
                (title_kwargs, 'title_kwargs'),
                (xlabel_kwargs, 'xlabel_kwargs'),
                (ylabel_kwargs, 'ylabel_kwargs'),
                (savefig_kwargs, 'savefig_kwargs')):
            if kwargs is not None:
                check_type(kwargs, kwargs_name, dict, 'a dictionary')
                for key in kwargs:
                    if not isinstance(key, str):
                        error_message = (
                            f'all keys of {kwargs_name} must be strings, but '
                            f'it contains a key of type '
                            f'{type(key).__name__!r}')
                        raise TypeError(error_message)
        # If point_size is None, default to 30,000 / num_cells; otherwise,
        # require it to be a positive number or the name of a numeric column of
        # obs with all-positive numbers
        num_cells = \
            len(self) if cells_to_plot_column is None else len(embedding)
        if point_size is None:
            # noinspection PyUnboundLocalVariable
            point_size = 30_000 / num_cells
        else:
            check_type(point_size, 'point_size', (int, float, str),
                       'a positive number or string')
            if isinstance(point_size, (int, float)):
                check_bounds(point_size, 'point_size', 0, left_open=True)
            else:
                if point_size not in self._obs:
                    error_message = \
                        f'point_size {point_size!r} is not a column of obs'
                    raise ValueError(error_message)
                point_size = self._obs[point_size]
                if not (point_size.dtype.is_integer() or
                        point_size.dtype.is_float()):
                    error_message = (
                        f'the point_size column, obs[{point_size!r}], must '
                        f'have an integer or floating-point data type, but '
                        f'has data type {point_size.dtype.base_type()!r}')
                    raise TypeError(error_message)
                if point_size.min() <= 0:
                    error_message = (
                        f'the point_size column, obs[{point_size!r}], does '
                        f'not have all-positive elements')
                    raise ValueError(error_message)
        # If `sort_by_frequency=True`, ensure `palette` is None and
        # `color_column` is discrete
        check_type(sort_by_frequency, 'sort_by_frequency', bool, 'Boolean')
        if sort_by_frequency:
            if palette is not None:
                error_message = (
                    f'sort_by_frequency must be False when palette is '
                    f'specified')
                raise ValueError(error_message)
            # noinspection PyUnboundLocalVariable
            if color_column is None or not discrete:
                error_message = (
                    f'sort_by_frequency must be False when color_column is '
                    f'{"None" if color_column is None else "continuous"}')
                raise ValueError(error_message)
        # If `palette` is not None, check that it is a string, Colormap
        # object, or dictionary where all keys are in `color_column` and all
        # values are valid Matplotlib colors (and normalize these to hex
        # codes). If None and `color_column` is discrete, assign colors via
        # `generate_palette()`, in natural sort order, or decreasing order of
        # frequency if `sort_by_frequency=True`. Also make sure `palette` and
        # `palette_kwargs` are None when `color_column` is None.
        if palette is not None:
            if color_column is None:
                error_message = \
                    'palette must be None when color_column is None'
                raise ValueError(error_message)
            if palette_kwargs is not None:
                error_message = \
                    'palette_kwargs must be None when palette is specified'
                raise ValueError(error_message)
            check_type(palette, 'palette',
                       (str, plt.matplotlib.colors.Colormap, dict),
                       'a string, matplotlib Colormap object, or dictionary')
            if isinstance(palette, str):
                palette = plt.colormaps[palette]
            elif isinstance(palette, dict):
                # noinspection PyUnboundLocalVariable
                if not discrete:
                    error_message = (
                        'palette cannot be a dictionary when color_column is '
                        'continuous')
                    raise ValueError(error_message)
                for key, value in palette.items():
                    if not isinstance(key, str):
                        error_message = (
                            f'all keys of palette must be strings, but it '
                            f'contains a key of type {type(key).__name__!r}')
                        raise TypeError(error_message)
                    # noinspection PyUnboundLocalVariable
                    if key not in unique_color_labels:
                        error_message = (
                            f'palette is a dictionary containing the key '
                            f'{key!r}, which is not one of the values in '
                            f'obs[{color_column!r}]')
                        raise ValueError(error_message)
                    if not plt.matplotlib.colors.is_color_like(value):
                        error_message = \
                            f'palette[{key!r}] is not a valid Matplotlib color'
                        raise ValueError(error_message)
                    palette[key] = plt.matplotlib.colors.to_hex(value)
        else:
            # noinspection PyUnboundLocalVariable
            if color_column is not None and discrete:
                if palette_kwargs is None:
                    palette_kwargs = {}
                # noinspection PyUnboundLocalVariable
                color_order = \
                    color_labels.value_counts(sort=True).to_series() \
                    if sort_by_frequency else \
                        sorted(color_labels.unique(),
                               key=lambda color_label: [
                                   int(text) if text.isdigit() else
                                   text.lower() for text in
                                   re.split('([0-9]+)', color_label)])
                palette = generate_palette(len(color_order), **palette_kwargs)
                palette = dict(zip(color_order, palette))
            elif palette_kwargs is not None:
                error_message = (
                    f'palette_kwargs must be None when color_column is '
                    f'{"None" if color_column is None else "continuous"}')
                raise ValueError(error_message)
        # Check that `default_color` is a valid Matplotlib color, and convert
        # it to hex
        if not plt.matplotlib.colors.is_color_like(default_color):
            error_message = 'default_color is not a valid Matplotlib color'
            raise ValueError(error_message)
        default_color = plt.matplotlib.colors.to_hex(default_color)
        # Override the defaults for certain keys of `scatter_kwargs`
        default_scatter_kwargs = dict(rasterized=True, linewidths=0)
        scatter_kwargs = default_scatter_kwargs | scatter_kwargs \
            if scatter_kwargs is not None else default_scatter_kwargs
        # Check that `scatter_kwargs` does not contain the `s`, `c`/`color`, or
        # `cmap` keys
        if 's' in scatter_kwargs:
            error_message = (
                "'s' cannot be specified as a key in scatter_kwargs; specify "
                "the point_size argument instead")
            raise ValueError(error_message)
        for key in 'c', 'color', 'cmap':
            if key in scatter_kwargs:
                error_message = (
                    f'{key!r} cannot be specified as a key in '
                    f'scatter_kwargs; specify the color_column, palette, '
                    f'palette_kwargs, and/or default_color arguments instead')
                raise ValueError(error_message)
        # If `label=True`, require `color_column` to be discrete.
        # If `label=False`, `label_kwargs` must be None.
        check_type(label, 'label', bool, 'Boolean')
        if label:
            if color_column is None:
                error_message = 'color_column cannot be None when label=True'
                raise ValueError(error_message)
            if not discrete:
                error_message = \
                    'color_column cannot be continuous when label=True'
                raise ValueError(error_message)
        elif label_kwargs is not None:
            error_message = 'label_kwargs must be None when label=False'
            raise ValueError(error_message)
        # Only add a legend if `legend=True` and `color_column` is discrete.
        # If not adding a legend, `legend_kwargs` must be None.
        check_type(legend, 'legend', bool, 'Boolean')
        add_legend = legend and color_column is not None and discrete
        if not add_legend and legend_kwargs is not None:
            error_message = (
                f'legend_kwargs must be None when color_column is '
                f'{"None" if color_column is None else "continuous"}')
            raise ValueError(error_message)
        # Only add a colorbar if `colorbar=True` and `color_column` is
        # continuous. If not adding a colorbar, `colorbar_kwargs` must be None.
        check_type(colorbar, 'colorbar', bool, 'Boolean')
        add_colorbar = colorbar and color_column is not None and not discrete
        if not add_colorbar and colorbar_kwargs is not None:
            error_message = (
                f'colorbar_kwargs must be None when color_column is '
                f'{"None" if color_column is None else "discrete"}')
            raise ValueError(error_message)
        # `title` must be a string or None; if None, `title_kwargs` must be as
        # well; ditto for `xlabel` and `ylabel`
        for arg, arg_name, arg_kwargs in (
                (title, 'title', title_kwargs),
                (xlabel, 'xlabel', xlabel_kwargs),
                (ylabel, 'ylabel', ylabel_kwargs)):
            if arg is not None:
                check_type(arg, arg_name, str, 'a string')
            elif arg_kwargs is not None:
                error_message = \
                    f'{arg_name}_kwargs must be None when {arg_name} is None'
                raise ValueError(error_message)
        # `xlim` and `ylim` must be length-2 tuples or None, with the first
        # element less than the second
        for arg, arg_name in (xlim, 'xlim'), (ylim, 'ylim'):
            if arg is not None:
                check_type(arg, arg_name, tuple, 'a length-2 tuple')
                if len(arg) != 2:
                    error_message = (
                        f'{arg_name} must be a length-2 tuple, but has length '
                        f'{len(arg):,}')
                    raise ValueError(error_message)
                if arg[0] >= arg[1]:
                    error_message = \
                        f'{arg_name}[0] must be less than {arg_name}[1]'
                    raise ValueError(error_message)
        # If `color_column` is None, plot all cells in `default_color`. If
        # `palette` is a dictionary, generate an explicit list of colors to
        # plot each cell in. If `palette` is a Colormap, just pass it as the
        # cmap` argument. If `palette` is missing and `color_column` is
        # continuous, set it to `plt.rcParams['image.cmap']` ('viridis' by
        # default)
        if color_column is None:
            c = default_color
            cmap = None
        elif isinstance(palette, dict):
            # Note: `replace_strict(..., default=default_color)` fills both
            # missing values and values missing from `palette` with
            # `default_color`
            c = color_labels\
                .replace_strict(palette, default=default_color,
                                return_dtype=pl.String)\
                .to_numpy()
            cmap = None
        else:
            # Need to `copy()` because `set_bad()` is in-place
            c = color_labels.to_numpy()
            if palette is not None:
                cmap = palette.copy()
                # noinspection PyUnresolvedReferences
                cmap.set_bad(default_color)
            else:  # `color_column` is continuous
                cmap = plt.rcParams['image.cmap']
        # If `ax` is None, create a new figure with `constrained_layout=True`;
        # otherwise, check that it is a Matplotlib axis
        make_new_figure = ax is None
        try:
            if make_new_figure:
                plt.figure(constrained_layout=True)
                ax = plt.gca()
            else:
                check_type(ax, 'ax', plt.Axes, 'a Matplotlib axis')
            # Make a scatter plot of the embedding with equal x-y aspect ratios
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                                 s=point_size, c=c, cmap=cmap,
                                 **scatter_kwargs)
            ax.set_aspect('equal')
            # Add the title, axis labels and axis limits
            if title is not None:
                if title_kwargs is None:
                    ax.set_title(title)
                else:
                    ax.set_title(title, **title_kwargs)
            if xlabel is not None:
                if xlabel_kwargs is None:
                    ax.set_xlabel(xlabel)
                else:
                    ax.set_xlabel(xlabel, **xlabel_kwargs)
            if ylabel is not None:
                if ylabel_kwargs is None:
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_ylabel(ylabel, **ylabel_kwargs)
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            # Add the legend; override the defaults for certain values of
            # `legend_kwargs`
            if add_legend:
                default_legend_kwargs = dict(
                    loc='center left', bbox_to_anchor=(1, 0.5), frameon=False,
                    ncols=len(unique_color_labels) // 16 + 1)
                legend_kwargs = default_legend_kwargs | legend_kwargs \
                    if legend_kwargs is not None else default_legend_kwargs
                if isinstance(palette, dict):
                    for color_label, color in palette.items():
                        ax.scatter([], [], c=color, label=color_label,
                                   **scatter_kwargs)
                    plt.legend(**legend_kwargs)
                else:
                    plt.legend(*scatter.legend_elements(), **legend_kwargs)
            # Add the colorbar; override the defaults for certain keys of
            # `colorbar_kwargs`
            if add_colorbar:
                default_colorbar_kwargs = dict(shrink=0.5)
                colorbar_kwargs = default_colorbar_kwargs | colorbar_kwargs \
                    if colorbar_kwargs is not None else default_colorbar_kwargs
                plt.colorbar(scatter, ax=ax, **colorbar_kwargs)
            # Label cells; override the defaults for certain keys of
            # `label_kwargs`
            if label:
                from matplotlib.patheffects import withStroke
                if label_kwargs is None:
                    label_kwargs = {}
                # noinspection PyUnresolvedReferences
                label_kwargs |= dict(
                    horizontalalignment=label_kwargs.pop(
                        'horizontalalignment',
                        label_kwargs.pop('ha', 'center')),
                    verticalalignment=label_kwargs.pop(
                        'verticalalignment',
                        label_kwargs.pop('va', 'center')),
                    path_effects=[withStroke(linewidth=3, foreground='white',
                                             alpha=0.75)])
                for color_label in unique_color_labels:
                    ax.text(*np.median(embedding[color_labels ==
                                                      color_label], axis=0),
                            color_label, **label_kwargs)
            # Despine, if specified
            if despine:
                spines = plt.gca().spines
                spines['top'].set_visible(False)
                spines['right'].set_visible(False)
            # Save; override the defaults for certain keys of `savefig_kwargs`
            if filename is not None:
                default_savefig_kwargs = \
                    dict(dpi=300, bbox_inches='tight', pad_inches='layout',
                         transparent=filename is not None and
                                     filename.endswith('.pdf'))
                savefig_kwargs = default_savefig_kwargs | savefig_kwargs \
                    if savefig_kwargs is not None else default_savefig_kwargs
                plt.savefig(filename, **savefig_kwargs)
                if make_new_figure:
                    plt.close()
        except:
            # If we made a new figure, make sure to close it if there's an
            # exception (but not if there was no error and `filename` is None,
            # in case the user wants to modify it further before saving)
            if make_new_figure:
                plt.close()
            raise


class Pseudobulk:
    """
    A pseudobulked single-cell dataset resulting from calling `pseudobulk()`
    on a SingleCell dataset.
    
    Has slots for:
    - X: a dict of NumPy arrays of counts per cell and gene for each cell type
    - obs: a dict of polars DataFrames of sample metadata for each cell type
    - var: a dict of polars DataFrames of gene metadata for each cell type
    as well as `obs_names` and `var_names`, aliases for a dict of obs[:, 0] and
    var[:, 0] for each cell type, and `cell_types`, a tuple of cell types.
    
    Supports iteration:
    - `for cell_type in pseudobulk:` yields the cell type names, as does
      `for cell_type in pseudobulk.keys():`
    - `for X, obs, var in pseudobulk.values():` yields the X, obs and var for
       each cell type
    - `for cell_type, (X, obs, var) in pseudobulk.items():` yields both the
      name and the X, obs and var for each cell type
    - `for X in pseudobulk.iter_X():` yields just the X for each cell type
    - `for X in pseudobulk.iter_obs():` yields just the obs for each cell type
    - `for X in pseudobulk.iter_var():` yields just the var for each cell type
    """
    def __init__(self,
                 X: dict[str, np.ndarray[2, np.integer | np.floating]] |
                    str | Path,
                 obs: dict[str, pl.DataFrame] = None,
                 var: dict[str, pl.DataFrame] = None) -> None:
        """
        Load a saved Pseudobulk dataset, or create one from an in-memory count
        matrix + metadata for each cell type. The latter functionality is
        mainly for internal use; most users will create new pseudobulk datasets
        by calling `pseudobulk()` on a SingleCell dataset.
        
        Args:
            X: a {cell type: NumPy array} dict of counts or log CPMs, or a
               directory to load a saved Pseudobulk dataset from (see save())
            obs: a {cell type: polars DataFrame} dict of metadata per sample.
                 The first column must be String, Enum or Categorical.
            var: a {cell type: polars DataFrame} dict of metadata per gene.
                 The first column must be String, Enum or Categorical.
        """
        if isinstance(X, dict):
            if obs is None:
                error_message = (
                    'obs is None, but since X is a dictionary, obs must also '
                    'be a dictionary')
                raise TypeError(error_message)
            if var is None:
                error_message = (
                    'var is None, but since X is a dictionary, var must also '
                    'be a dictionary')
                raise TypeError(error_message)
            if not X:
                error_message = 'X is an empty dictionary'
                raise ValueError(error_message)
            if X.keys() != obs.keys():
                error_message = \
                    'X and obs must have the same keys (cell types)'
                raise ValueError(error_message)
            if X.keys() != var.keys():
                error_message = \
                    'X and var must have the same keys (cell types)'
                raise ValueError(error_message)
            for cell_type in X:
                if not isinstance(cell_type, str):
                    error_message = (
                        f'all keys of X (cell types) must be strings, but X '
                        f'contains a key of type {type(cell_type).__name__!r}')
                    raise TypeError(error_message)
                check_type(X[cell_type], f'X[{cell_type!r}]', np.ndarray,
                           'a NumPy array')
                if X[cell_type].ndim != 2:
                    error_message = (
                        f'X[{cell_type!r}] is a {X[cell_type].ndim:,}-'
                        f'dimensional NumPy array, but must be 2-dimensional')
                    raise ValueError(error_message)
                check_type(obs[cell_type], f'obs[{cell_type!r}]', pl.DataFrame,
                           'a polars DataFrame')
                check_type(var[cell_type], f'var[{cell_type!r}]', pl.DataFrame,
                           'a polars DataFrame')
            self._X = X
            self._obs = obs
            self._var = var
        elif isinstance(X, (str, Path)):
            X = str(X)
            if not os.path.exists(X):
                error_message = f'Pseudobulk directory {X!r} does not exist'
                raise FileNotFoundError(error_message)
            cell_types = [line.rstrip('\n') for line in
                          open(f'{X}/cell_types.txt')]
            self._X = {cell_type: np.load(
                os.path.join(X, f'{cell_type.replace("/", "-")}.X.npy'))
                for cell_type in cell_types}
            self._obs = {cell_type: pl.read_parquet(
                os.path.join(X, f'{cell_type.replace("/", "-")}.obs.parquet'))
                for cell_type in cell_types}
            self._var = {cell_type: pl.read_parquet(
                os.path.join(X, f'{cell_type.replace("/", "-")}.var.parquet'))
                for cell_type in cell_types}
        else:
            error_message = (
                f'X must be a dictionary of NumPy arrays or a directory '
                f'containing a saved Pseudobulk dataset, but has type '
                f'{type(X).__name__!r}')
            raise ValueError(error_message)
        for cell_type in self:
            if len(self._obs[cell_type]) == 0:
                error_message = \
                    f'len(obs[{cell_type!r}]) is 0: no samples remain'
                raise ValueError(error_message)
            if len(self._var[cell_type]) == 0:
                error_message = \
                    f'len(var[{cell_type!r}]) is 0: no genes remain'
                raise ValueError(error_message)
            if len(self._obs[cell_type]) != len(self._X[cell_type]):
                error_message = (
                    f'len(obs[{cell_type!r}]) is '
                    f'{len(self._obs[cell_type]):,}, but '
                    f'len(X[{cell_type!r}]) is {len(X[cell_type]):,}')
                raise ValueError(error_message)
            if len(self._var[cell_type]) != self._X[cell_type].shape[1]:
                error_message = (
                    f'len(var[{cell_type!r}]) is '
                    f'{len(self._var[cell_type]):,}, but '
                    f'X[{cell_type!r}].shape[1] is '
                    f'{self._X[cell_type].shape[1]:,}')
                raise ValueError(error_message)
            if self._obs[cell_type][:, 0].dtype not in \
                    (pl.String, pl.Categorical, pl.Enum):
                error_message = (
                    f'the first column of obs[{cell_type!r}] '
                    f'({self._obs[cell_type].columns[0]!r}) must be String, '
                    f'Categorical or Enum, but has data type '
                    f'{self._obs[cell_type][:, 0].dtype.base_type()!r}')
                raise ValueError(error_message)
            if self._var[cell_type][:, 0].dtype not in \
                    (pl.String, pl.Categorical, pl.Enum):
                error_message = (
                    f'the first column of var[{cell_type!r}] '
                    f'({self._var[cell_type].columns[0]!r}) must be String, '
                    f'Categorical or Enum, but has data type '
                    f'{self._var[cell_type][:, 0].dtype.base_type()!r}')
                raise ValueError(error_message)
    
    @staticmethod
    def _setter_check(new: dict[str, np.ndarray[2, np.integer | np.floating]
                                     | pl.DataFrame],
                      old: dict[str, np.ndarray[2, np.integer | np.floating]
                                     | pl.DataFrame],
                      name: str) -> None:
        """
        When setting X, obs or var, raise an error if the new value is not a
        dictionary, the new keys (cell types) differ from the old ones, or
        the new values differ in length (or shape, in the case of X) from the
        old ones. For obs and var, also check that the first column is String,
        Categorical or Enum.
        
        Args:
            new: the new X, obs or var
            old: the old X, obs or var
            name: the name of the field: 'X', 'obs' or 'var'
        """
        if not isinstance(new, dict):
            error_message = (
                f'new {name} must be a dictionary, but has type '
                f'{type(new).__name__!r}')
            raise TypeError(error_message)
        if new.keys() != old.keys():
            error_message = (
                f'new {name} has different cell types (keys) from the old '
                f'{name}')
            raise ValueError(error_message)
        if name == 'X':
            for cell_type in new:
                check_type(new[cell_type], f'X[{cell_type!r}]', np.ndarray,
                           'a NumPy array')
                new_shape = new[cell_type].shape
                old_shape = old[cell_type].shape
                if new_shape != old_shape:
                    error_message = (
                        f'new X is {new_shape.shape[0]:,} × '
                        f'{new_shape.shape[1]:,}, but old X is '
                        f'{old_shape.shape[0]:,} × {old_shape.shape[1]:,}')
                    raise ValueError(error_message)
        else:
            for cell_type in new:
                check_type(new[cell_type], f'{name}[{cell_type!r}]',
                           pl.DataFrame, 'a polars DataFrame')
                if new[cell_type][:, 0].dtype not in (
                        pl.String, pl.Categorical, pl.Enum):
                    error_message = (
                        f'the first column of {name}[{cell_type!r}] '
                        f'({new[cell_type].columns[0]!r}) must be String, '
                        f'Categorical or Enum, but the first column of the '
                        f'new {name} has data type '
                        f'{new[cell_type][:, 0].dtype.base_type()!r}')
                    raise ValueError(error_message)
                if len(new) != len(old):
                    error_message = (
                        f'new {name} has length {len(new):,}, but old {name} '
                        f'has length {len(old):,}')
                    raise ValueError(error_message)
    
    @property
    def X(self) -> dict[str, np.ndarray[2, np.integer | np.floating]]:
        return self._X
    
    @X.setter
    def X(self, X: dict[str, np.ndarray[2, np.integer | np.floating]]) -> \
            None:
        self._setter_check(X, self._X, 'X')
        self._X = X
    
    @property
    def obs(self) -> dict[str, pl.DataFrame]:
        return self._obs
    
    @obs.setter
    def obs(self, obs: dict[str, pl.DataFrame]) -> None:
        self._setter_check(obs, self._obs, 'obs')
        self._obs = obs

    @property
    def var(self) -> dict[str, pl.DataFrame]:
        return self._var
    
    @var.setter
    def var(self, var: dict[str, pl.DataFrame]) -> None:
        self._setter_check(var, self._var, 'var')
        self._var = var

    @property
    def obs_names(self) -> dict[str, pl.Series]:
        return {cell_type: obs[:, 0] for cell_type, obs in self._obs.items()}
    
    @property
    def var_names(self) -> dict[str, pl.Series]:
        return {cell_type: var[:, 0] for cell_type, var in self._var.items()}
    
    def set_obs_names(self, column: str) -> Pseudobulk:
        """
        Sets a column as the new first column of obs, i.e. the obs_names.
        
        Args:
            column: the column name in obs; must have String, Categorical, or
                    Enum data type

        Returns:
            A new Pseudobulk dataset with `column` as the first column of each
            cell type's obs. If `column` is already the first column for every
            cell type, return this dataset unchanged.
        """
        check_type(column, 'column', str, 'a string')
        if all(column == cell_type_obs.columns[0]
               for cell_type_obs in self._obs.values()):
            return self
        obs = {}
        for cell_type, cell_type_obs in self._obs.items():
            if column not in cell_type_obs:
                error_message = \
                    f'{column!r} is not a column of obs[{cell_type!r}]'
                raise ValueError(error_message)
            check_dtype(cell_type_obs, f'obs[{column!r}]',
                        (pl.String, pl.Categorical, pl.Enum))
            obs[cell_type] = cell_type_obs.select(column, pl.exclude(column))
        return Pseudobulk(X=self._X, obs=obs, var=self._var)
    
    def set_var_names(self, column: str) -> Pseudobulk:
        """
        Sets a column as the new first column of var, i.e. the var_names.
        
        Args:
            column: the column name in var; must have String, Categorical, or
                    Enum data type

        Returns:
            A new Pseudobulk dataset with `column` as the first column of each
            cell type's var. If `column` is already the first column for every
            cell type, return this dataset unchanged.
        """
        check_type(column, 'column', str, 'a string')
        if all(column == cell_type_var.columns[0]
               for cell_type_var in self._var.values()):
            return self
        var = {}
        for cell_type, cell_type_var in self._var.items():
            if column not in cell_type_var:
                error_message = \
                    f'{column!r} is not a column of var[{cell_type!r}]'
                raise ValueError(error_message)
            check_dtype(cell_type_var, f'var[{column!r}]',
                        (pl.String, pl.Categorical, pl.Enum))
            var[cell_type] = cell_type_var.select(column, pl.exclude(column))
        return Pseudobulk(X=self._X, obs=self._obs, var=var)

    def keys(self) -> KeysView[str]:
        """
        Get a KeysView (like you would get from `dict.keys()`) of this
        Pseudobulk dataset's cell types. `for cell_type in pb.keys()` is
        equivalent to `for cell_type in pb`.
        
        Returns:
            A KeysView of the cell types.
        """
        return self._X.keys()
    
    def values(self) -> ValuesView[tuple[np.ndarray[2, np.integer |
                                                       np.floating],
                                       pl.DataFrame, pl.DataFrame]]:
        """
        Get a ValuesView (like you would get from `dict.values()`) of
        `(X, obs, var)` tuples for each cell type in this Pseudobulk dataset.
        
        Returns:
            A ValuesView of `(X, obs, var)` tuples for each cell type.
        """
        return {cell_type: (self._X[cell_type], self._obs[cell_type],
                            self._var[cell_type])
                for cell_type in self._X}.values()
    
    def items(self) -> ItemsView[str, tuple[np.ndarray[2, np.integer |
                                                          np.floating],
                                            pl.DataFrame, pl.DataFrame]]:
        """
        Get an ItemsView (like you would get from `dict.items()`) of
        `(cell_type, (X, obs, var))` tuples for each cell type in this
        Pseudobulk dataset.
        
        Yields:
            An ItemsView of `(cell_type, (X, obs, var))` tuples for each cell
            type.
        """
        return {cell_type: (self._X[cell_type], self._obs[cell_type],
                            self._var[cell_type])
                for cell_type in self._X}.items()
    
    def iter_X(self) -> Iterable[np.ndarray[2, np.integer | np.floating]]:
        """
        Iterate over each cell type's X.
        
        Yields:
            X for each cell type.
        """
        for cell_type in self:
            yield self._X[cell_type]
    
    def iter_obs(self) -> Iterable[pl.DataFrame]:
        """
        Iterate over each cell type's obs.
        
        Yields:
            obs for each cell type.
        """
        for cell_type in self:
            yield self._obs[cell_type]
    
    def iter_var(self) -> Iterable[pl.DataFrame]:
        """
        Iterate over each cell type's var.
        
        Yields:
            var for each cell type.
        """
        for cell_type in self:
            yield self._var[cell_type]
    
    def map_X(self, function: Callable[[np.ndarray[2, np.integer |
                                                      np.floating], ...],
                                        np.ndarray[2, np.integer |
                                                      np.floating]],
              *args: Any, **kwargs: Any) -> Pseudobulk:
        """
        Apply a function to each cell type's X.
        
        Args:
            function: the function to apply
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new Pseudobulk dataset where the function has been applied to
            each cell type's X.
        """
        return Pseudobulk(X={cell_type: function(self._X[cell_type], *args,
                                                 **kwargs)
                             for cell_type in self},
                          obs=self._obs, var=self._var)
    
    def map_obs(self, function: Callable[[pl.DataFrame, ...], pl.DataFrame],
                *args: Any, **kwargs: Any) -> Pseudobulk:
        """
        Apply a function to each cell type's obs.
        
        Args:
            function: the function to apply
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new Pseudobulk dataset where the function has been applied to
            each cell type's obs.
        """
        return Pseudobulk(X=self._X,
                          obs={cell_type: function(self._obs[cell_type], *args,
                                                   **kwargs)
                               for cell_type in self},
                          var=self._var)
    
    def map_var(self, function: Callable[[pl.DataFrame, ...], pl.DataFrame],
                *args: Any, **kwargs: Any) -> Pseudobulk:
        """
        Apply a function to each cell type's var.
        
        Args:
            function: the function to apply
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new Pseudobulk dataset where the function has been applied to
            each cell type's var.
        """
        return Pseudobulk(X=self._X, obs=self._obs,
                          var={cell_type: function(self._var[cell_type], *args,
                                                   **kwargs)
                               for cell_type in self})
        
    def __eq__(self, other: Pseudobulk) -> bool:
        """
        Test for equality with another Pseudobulk dataset.
        
        Args:
            other: the other Pseudobulk dataset to test for equality with

        Returns:
            Whether the two Pseudobulk datasets are identical.
        """
        if not isinstance(other, Pseudobulk):
            error_message = (
                f'the left-hand operand of `==` is a Pseudobulk dataset, but '
                f'the right-hand operand has type {type(other).__name__!r}')
            raise TypeError(error_message)
        # noinspection PyUnresolvedReferences
        return tuple(self.keys()) == tuple(other.keys()) and \
            all(obs.equals(other_obs) for obs, other_obs in
                zip(self._obs.values(), other._obs.values())) and \
            all(var.equals(other_var) for var, other_var in
                zip(self._var.values(), other._var.values())) and \
            all((X.ravel() == other_X.ravel()).all() for X, other_X in
                zip(self._X.values(), other._X.values()))
    
    def __or__(self, other: Pseudobulk) -> Pseudobulk:
        """
        Combine the cell types of this Pseudobulk dataset with another. The
        two datasets must have non-overlapping cell types.
        
        Args:
            other: the other Pseudobulk dataset to combine with this one

        Returns:
            A Pseudobulk dataset with each of the cell types in the first
            Pseudobulk dataset, followed by each of the cell types in the
            second.
        """
        if not isinstance(other, Pseudobulk):
            error_message = (
                f'the left-hand operand of `|` is a Pseudobulk dataset, but '
                f'the right-hand operand has type {type(other).__name__!r}')
            raise TypeError(error_message)
        if self.keys() & other.keys():
            error_message = (
                'the left- and right-hand operands of `|` are Pseudobulk '
                'datasets that share some cell types')
            raise ValueError(error_message)
        return Pseudobulk(X=self._X | other._X, obs=self._obs | other._obs,
                          var=self._var | other._var)
    
    @staticmethod
    def _getitem_error(item: Indexer) -> None:
        """
        Raise an error if the indexer is invalid.
        
        Args:
            item: the indexer
        """
        types = tuple(type(elem).__name__ for elem in to_tuple(item))
        if len(types) == 1:
            types = types[0]
        error_message = (
            f'Pseudobulk indices must be a cell-type string, a length-1 tuple '
            f'of (cell_type,), a length-2 tuple of (cell_type, samples), or a '
            f'length-3 tuple of (cell_type, samples, genes). Samples and '
            f'genes must each be a string or integer; a slice of strings or '
            f'integers; or a list, NumPy array, or polars Series of strings, '
            f'integers, or Booleans. You indexed with: {types}.')
        raise ValueError(error_message)
    
    @staticmethod
    def _getitem_by_string(df: pl.DataFrame, string: str) -> int:
        """
        Get the index where df[:, 0] == string, raising an error if no rows or
        multiple rows match.
        
        Args:
            df: a DataFrame (obs or var)
            string: the string to find the index of in the first column of df

        Returns:
            The integer index of the string within the first column of df.
        """
        first_column = df.columns[0]
        try:
            return df\
                .select(first_column)\
                .with_row_index('__Pseudobulk_getitem')\
                .row(by_predicate=pl.col(first_column) == string)\
                [0]
        except pl.exceptions.NoRowsReturnedError:
            raise KeyError(string)
    
    @staticmethod
    def _getitem_process(item: Indexer, index: int, df: pl.DataFrame) -> \
            list[int] | slice | pl.Series:
        """
        Process an element of an item passed to __getitem__().
        
        Args:
            item: the item
            index: the index of the element to process
            df: the DataFrame (obs or var) to process the element with respect
                to

        Returns:
            A new indexer indicating the rows/columns to index.
        """
        subitem = item[index]
        if is_integer(subitem):
            return [subitem]
        elif isinstance(subitem, str):
            return [Pseudobulk._getitem_by_string(df, subitem)]
        elif isinstance(subitem, slice):
            start = subitem.start
            stop = subitem.stop
            step = subitem.step
            if isinstance(start, str):
                start = Pseudobulk._getitem_by_string(df, start)
            elif start is not None and not is_integer(start):
                Pseudobulk._getitem_error(item)
            if isinstance(stop, str):
                stop = Pseudobulk._getitem_by_string(df, stop)
            elif stop is not None and not is_integer(stop):
                Pseudobulk._getitem_error(item)
            if step is not None and not is_integer(step):
                Pseudobulk._getitem_error(item)
            return slice(start, stop, step)
        elif isinstance(subitem, (list, np.ndarray, pl.Series)):
            if not isinstance(subitem, pl.Series):
                subitem = pl.Series(subitem)
            if subitem.is_null().any():
                error_message = (
                    'indexer contains null entries; this may happen when '
                    'indexing with a mix of strings and integers, due to a '
                    'bug in Polars: github.com/pola-rs/polars/issues/11156')
                raise ValueError(error_message)
            if subitem.dtype == pl.String or subitem.dtype == \
                    pl.Categorical or subitem.dtype == pl.Enum:
                indices = subitem\
                    .to_frame(df.columns[0])\
                    .join(df.with_row_index('_Pseudobulk_index'),
                          on=df.columns[0], how='left')\
                    ['_Pseudobulk_index']
                if indices.null_count():
                    error_message = subitem.filter(indices.is_null())[0]
                    raise KeyError(error_message)
                return indices
            elif subitem.dtype.is_integer() or subitem.dtype == pl.Boolean:
                return subitem
            else:
                Pseudobulk._getitem_error(item)
        else:
            Pseudobulk._getitem_error(item)
    
    def __contains__(self, cell_type: str) -> bool:
        """
        Check if this Pseudobulk dataset contains the specified cell type.
        
        Args:
            cell_type: the cell type

        Returns:
            Whether the cell type is present in the Pseudobulk dataset.
        """
        check_type(cell_type, 'cell_type', str, 'a string')
        return cell_type in self._X
    
    def __getitem__(self, item: Indexer | tuple[str, Indexer, Indexer]) -> \
            Pseudobulk:
        """
        Subset to specific cell type(s), sample(s), and/or gene(s).
        
        Index with a tuple of `(cell_types, samples, genes)`. If `samples` and
        `genes` are integers, arrays/lists/slices of integers, or arrays/lists
        of Booleans, the result will be a Pseudobulk dataset subset to
        `X[samples, genes]`, `obs[samples]`, and `var[genes]` for each of the
        cell types in `cell_types`. However, `samples` and/or `genes` can
        instead be strings (or arrays or slices of strings), in which case they
        refer to the first column of obs and/or var, respectively.
        
        Examples:
        - Subset to one cell type:
          pseudobulk['Astro']
        - Subset to multiple cell types:
          pseudobulk[['Astro', 'Micro']]
        - Subset to one cell type and sample, for all genes:
          pb['Astro', 'H19.30.002']
          pb['Astro', 2]
        - Subset to one gene, for all cell types and samples:
          pb[:, :, 'APOE']
          pb[:, :, 13196]
        - Subset to one cell type, sample and gene:
          pb['Astro', 'H18.30.002', 'APOE']
          pb['Astro', 2, 13196]
        - Subset to one cell type and a range of samples and genes:
          pb['Astro', 'H18.30.002':'H19.33.004', 'APOE':'TREM2']
          pb['Astro', 'H18.30.002':'H19.33.004', 13196:34268]
        - Subset to one a cell type and specific samples and genes:
          pb['Astro', ['H18.30.002', 'H19.33.004']]
          pb['Astro', :, pl.Series(['APOE', 'TREM2'])]
          pb['Astro', ('H18.30.002', 'H19.33.004'),
             np.array(['APOE', 'TREM2'])]
        
        Args:
            item: the item to index with

        Returns:
            A new Pseudobulk dataset subset to the specified cell types,
            samples, and/or genes.
        """
        if isinstance(item, tuple):
            if not 1 <= len(item) <= 3:
                self._getitem_error(item)
            cell_types = to_tuple(item[0])
        elif isinstance(item, list):
            cell_types = to_tuple(item)
        elif isinstance(item, str):
            cell_types = item,
        else:
            self._getitem_error(item)
        # noinspection PyUnboundLocalVariable
        for cell_type in cell_types:
            if cell_type not in self:
                if isinstance(cell_type, str):
                    error_message = (
                        f'tried to select {cell_type!r}, which is not a cell '
                        f'type in this Pseudobulk')
                    raise ValueError(error_message)
                else:
                    error_message = (
                        f'tried to select a non-existent cell type of type '
                        f'{type(cell_type).__name__!r}')
                    raise TypeError(error_message)
        if not isinstance(item, tuple) or len(item) == 1:
            return Pseudobulk(X={cell_type: self._X[cell_type]
                                 for cell_type in cell_types},
                              obs={cell_type: self._obs[cell_type]
                                   for cell_type in cell_types},
                              var={cell_type: self._var[cell_type]
                                   for cell_type in cell_types})
        X, obs, var = {}, {}, {}
        for cell_type in cell_types:
            rows = self._getitem_process(item, 1, self._obs[cell_type])
            if isinstance(rows, pl.Series):
                obs[cell_type] = self._obs[cell_type].filter(rows) \
                    if rows.dtype == pl.Boolean else self._obs[cell_type][rows]
                rows = rows.to_numpy()
            else:
                obs[cell_type] = self._obs[cell_type][rows]
            if len(item) == 2:
                X[cell_type] = self._X[cell_type][rows]
                var[cell_type] = self._var[cell_type]
            else:
                columns = self._getitem_process(item, 2, self._var[cell_type])
                if isinstance(columns, pl.Series):
                    var[cell_type] = self._var[cell_type].filter(columns) \
                        if columns.dtype == pl.Boolean \
                        else self._var[cell_type][columns]
                    columns = columns.to_numpy()
                else:
                    var[cell_type] = self._var[cell_type][columns]
                X[cell_type] = self._X[cell_type][rows, columns] \
                    if isinstance(rows, slice) or \
                       isinstance(columns, slice) else \
                    self._X[cell_type][np.ix_(rows, columns)]
        return Pseudobulk(X=X, obs=obs, var=var)
    
    def __iter__(self) -> Iterable[str]:
        """
        Iterate over the cell types of this Pseudobulk dataset.
        `for cell_type in pb` is equivalent to `for cell_type in pb.keys()`.
        
        Returns:
            An iterator over the cell types.
        """
        return iter(self._X)
    
    def __len__(self) -> dict[str, int]:
        """
        Get the number of samples in each cell type of this Pseudobulk dataset.
        
        Returns:
            A dictionary mapping each cell type to its number of samples.
        """
        return {cell_type: len(X_cell_type)
                for cell_type, X_cell_type in self._X.items()}
    
    def __repr__(self) -> str:
        """
        Get a string representation of this Pseudobulk dataset.
        
        Returns:
            A string summarizing the dataset.
        """
        min_num_samples = min(len(obs) for obs in self._obs.values())
        max_num_samples = max(len(obs) for obs in self._obs.values())
        min_num_genes = min(len(var) for var in self._var.values())
        max_num_genes = max(len(var) for var in self._var.values())
        samples_string = \
            f'{min_num_samples:,} {plural("sample", max_num_samples)}' \
            if min_num_samples == max_num_samples else \
            f'{min_num_samples:,}-{max_num_samples:,} samples'
        genes_string = \
            f'{min_num_genes:,} {plural("gene", max_num_genes)}' \
            if min_num_genes == max_num_genes else \
            f'{min_num_genes:,}-{max_num_genes:,} genes'
        return f'Pseudobulk dataset with {len(self._X):,} cell ' \
               f'{"types, each" if len(self._X) > 1 else "type,"} with ' \
               f'{samples_string} (obs) and {genes_string} (var)\n' + \
            fill(f'    Cell types: {", ".join(self._X)}',
                 width=os.get_terminal_size().columns,
                 subsequent_indent=' ' * 17)
    
    @property
    def shape(self) -> dict[str, tuple[int, int]]:
        """
        Get the shape of each cell type in this Pseudobulk dataset.
        
        Returns:
            A dictionary mapping each cell type to a length-2 tuple where the
            first element is the number of samples, and the second is the
            number of genes.
        """
        return {cell_type: X_cell_type.shape
                for cell_type, X_cell_type in self._X.items()}
    
    def save(self, directory: str | Path, overwrite: bool = False) -> None:
        """
        Saves a Pseudobulk dataset to `directory` (which must not exist unless
        `overwrite=True`, and will be created) with three files per cell type:
        the X at f'{cell_type}.X.npy', the obs at f'{cell_type}.obs.parquet',
        and the var at f'{cell_type}.var.parquet'. Also saves a text file,
        cell_types.txt, containing the cell types.
        
        Args:
            directory: the directory to save the Pseudobulk dataset to
            overwrite: if False, raises an error if the directory exists; if
                       True, overwrites files inside it as necessary
        """
        check_type(directory, 'directory', (str, Path),
                   'a string or pathlib.Path')
        directory = str(directory)
        if not overwrite and os.path.exists(directory):
            error_message = (
                f'directory {directory!r} already exists; set overwrite=True '
                f'to overwrite')
            raise FileExistsError(error_message)
        os.makedirs(directory, exist_ok=overwrite)
        with open(os.path.join(directory, 'cell_types.txt'), 'w') as f:
            print('\n'.join(self._X), file=f)
        for cell_type in self._X:
            escaped_cell_type = cell_type.replace('/', '-')
            np.save(os.path.join(directory, f'{escaped_cell_type}.X.npy'),
                    self._X[cell_type])
            self._obs[cell_type].write_parquet(
                os.path.join(directory, f'{escaped_cell_type}.obs.parquet'))
            self._var[cell_type].write_parquet(
                os.path.join(directory, f'{escaped_cell_type}.var.parquet'))
    
    def copy(self, deep: bool = False) -> Pseudobulk:
        """
        Make a deep (if deep=True) or shallow copy of this Pseudobulk dataset.
        
        Returns:
            A copy of the Pseudobulk dataset. Since polars DataFrames are
            immutable, obs[cell_type] and var[cell_type] will always point to
            the same underlying data as the original for all cell types. The
            only difference when deep=True is that X[cell_type] will point to a
            fresh copy of the data, rather than the same data. Watch out: when
            deep=False, any modifications to X[cell_type] will modify both
            copies!
        """
        check_type(deep, 'deep', bool, 'Boolean')
        return Pseudobulk(X={cell_type: cell_type_X.copy()
                             for cell_type, cell_type_X in self._X.items()}
                            if deep else self._X, obs=self._obs, var=self._var)

    def concat_obs(self,
                   datasets: Pseudobulk,
                   *more_datasets: Pseudobulk,
                   flexible: bool = False) -> Pseudobulk:
        """
        Concatenate the samples of multiple Pseudobulk datasets. All datasets
        must have the same cell types.
        
        By default, all datasets must have the same var. They must also have
        the same columns in obs, with the same data types.
        
        Conversely, if `flexible=True`, subset to genes present in all datasets
        (according to the first column of var, i.e. `var_names`) before
        concatenating. Subset to columns of var that are identical in all
        datasets after this subsetting. Also, subset to columns of obs that are
        present in all datasets, and have the same data types. All datasets'
        `obs_names` must have the same name and dtype, and similarly for
        `var_names`.
        
        The one exception to the obs "same data type" rule: if a column is Enum
        in some datasets and Categorical in others, or Enum in all datasets but
        with different categories in each dataset, that column will be retained
        as an Enum column (with the union of the categories) in the
        concatenated obs.
        
        Args:
            datasets: one or more Pseudobulk datasets to concatenate with this
                      one
            *more_datasets: additional Pseudobulk datasets to concatenate with
                            this one, specified as positional arguments
            flexible: whether to subset to genes and columns of obs and var
                      common to all datasets before concatenating, rather than
                      raising an error on any mismatches
        
        Returns:
            The concatenated Pseudobulk dataset.
        """
        # Check inputs
        if isinstance(datasets, Pseudobulk):
            datasets = datasets,
        datasets = (self,) + datasets + more_datasets
        if len(datasets) == 1:
            error_message = \
                'need at least one other Pseudobulk dataset to concatenate'
            raise ValueError(error_message)
        check_types(datasets[1:], 'datasets', Pseudobulk,
                    'Pseudobulk datasets')
        check_type(flexible, 'flexible', bool, 'Boolean')
        # Check that cell types match across all datasets
        if not all(self.keys() == dataset.keys() for dataset in datasets[1:]):
            error_message = \
                'not all Pseudobulk datasets have the same cell types'
            raise ValueError(error_message)
        # Perform either flexible or non-flexible concatenation
        X = {}
        obs = {}
        var = {}
        for cell_type in self:
            if flexible:
                # Check that `obs_names` and `var_names` have the same name and
                # data type for each cell type across all datasets
                obs_names_name = self._obs[cell_type][:, 0].name
                if not all(dataset._obs[cell_type][:, 0] == obs_names_name
                           for dataset in datasets[1:]):
                    error_message = (
                        f'[{cell_type!r}] not all Pseudobulk datasets have '
                        f'the same name for the first column of obs (the '
                        f'obs_names column)')
                    raise ValueError(error_message)
                var_names_name = self._var[cell_type][:, 0].name
                if not all(dataset._var[cell_type][:, 0].name == var_names_name
                           for dataset in datasets[1:]):
                    error_message = (
                        f'[{cell_type!r}] not all Pseudobulk datasets have '
                        f'the same name for the first column of var (the '
                        f'var_names column)')
                    raise ValueError(error_message)
                obs_names_dtype = self._obs[cell_type][:, 0].dtype
                if not all(dataset._obs[cell_type][:, 0].dtype ==
                           obs_names_dtype for dataset in datasets[1:]):
                    error_message = (
                        f'[{cell_type!r}] not all Pseudobulk datasets have '
                        f'the same data type for the first column of obs (the '
                        f'obs_names column)')
                    raise TypeError(error_message)
                var_names_dtype = self._var[cell_type][:, 0].dtype
                if not all(dataset._var[cell_type][:, 0].dtype ==
                           var_names_dtype for dataset in datasets[1:]):
                    error_message = (
                        f'[{cell_type!r}] not all Pseudobulk datasets have '
                        f'the same data type for the first column of var (the '
                        f'var_names column)')
                    raise TypeError(error_message)
                # Subset to genes in common across all datasets
                genes_in_common = self._var[cell_type][:, 0]\
                    .filter(self._var[cell_type][:, 0]
                            .is_in(pl.concat([dataset._var[cell_type][:, 0]
                                              for dataset in datasets[1:]])))
                if len(genes_in_common) == 0:
                    error_message = (
                        f'[{cell_type!r}] no genes are shared across all '
                        f'Pseudobulk datasets')
                    raise ValueError(error_message)
                cell_type_X = []
                cell_type_var = []
                for dataset in datasets:
                    gene_indices = dataset._getitem_process(
                        genes_in_common, 1, dataset._var[cell_type])
                    cell_type_X.append(
                        dataset._X[cell_type][:, gene_indices.to_numpy()])
                    cell_type_var.append(dataset._var[cell_type][gene_indices])
                # Subset to columns of var that are identical in all datasets
                # after this subsetting
                var_columns_in_common = [
                    column.name for column in cell_type_var[0][:, 1:]
                    if all(column.name in dataset_cell_type_var and
                           dataset_cell_type_var[column.name].equals(column)
                           for dataset_cell_type_var in cell_type_var[1:])]
                cell_type_var = cell_type_var[0]
                cell_type_var = cell_type_var.select(cell_type_var.columns[0],
                                                     var_columns_in_common)
                # Subset to columns of obs that are present in all datasets,
                # and have the same data types. Also include columns of obs
                # that are Enum in some datasets and Categorical in others, or
                # Enum in all datasets but with different categories in each
                # dataset; cast these to Categorical.
                obs_mismatched_categoricals = {
                    column for column, dtype in self._obs[cell_type][:, 1:]
                    .select(pl.col(pl.Categorical, pl.Enum)).schema.items()
                    if all(column in dataset._obs[cell_type] and
                           dataset._obs[cell_type][column].dtype in
                           (pl.Categorical, pl.Enum)
                           for dataset in datasets[1:]) and
                       not all(dataset._obs[cell_type][column].dtype == dtype
                               for dataset in datasets[1:])}
                obs_columns_in_common = [
                    column
                    for column, dtype in islice(
                        self._obs[cell_type].schema.items(), 1)
                    if column in obs_mismatched_categoricals or
                       all(column in dataset[cell_type]._obs and
                           dataset._obs[cell_type][column].dtype == dtype
                           for dataset in datasets[1:])]
                cast_dict = {column: pl.Enum(
                    pl.concat([dataset._obs[cell_type][column]
                              .cat.get_categories() for dataset in datasets])
                    .unique(maintain_order=True))
                    for column in obs_mismatched_categoricals}
                cell_type_obs = [
                    dataset._obs[cell_type]
                    .cast(cast_dict)
                    .select(obs_columns_in_common) for dataset in datasets]
            else:  # non-flexible
                # Check that all var are identical
                cell_type_var = self._var[cell_type]
                for dataset in datasets[1:]:
                    if not dataset._var[cell_type].equals(cell_type_var):
                        error_message = (
                            f'[{cell_type!r}] all Pseudobulk datasets must '
                            f'have the same var, unless flexible=True')
                        raise ValueError(error_message)
                # Check that all obs have the same columns and data types
                schema = self._obs[cell_type].schema
                for dataset in datasets[1:]:
                    if dataset._obs[cell_type].schema != schema:
                        error_message = (
                            f'[{cell_type!r}] all Pseudobulk datasets must '
                            f'have the same columns in obs, with the same '
                            f'data types, unless flexible=True')
                        raise ValueError(error_message)
                cell_type_X = [dataset._X[cell_type] for dataset in datasets]
                cell_type_obs = [dataset._obs[cell_type]
                                 for dataset in datasets]
            # Concatenate
            X[cell_type] = np.vstack(cell_type_X)
            obs[cell_type] = pl.concat(cell_type_obs)
            var[cell_type] = cell_type_var
        return Pseudobulk(X=X, obs=obs, var=var)

    def concat_var(self,
                   datasets: Pseudobulk,
                   *more_datasets: Pseudobulk,
                   flexible: bool = False) -> Pseudobulk:
        """
        Concatenate the genes of multiple Pseudobulk datasets. All datasets
        must have the same cell types.
        
        By default, all datasets must have the same obs. They must also have
        the same columns in var, with the same data types.
        
        Conversely, if `flexible=True`, subset to cells present in all
        datasets (according to the first column of obs, i.e. `obs_names`)
        before concatenating. Subset to columns of obs that are identical in
        all datasets after this subsetting. Also, subset to columns of var that
        are present in all datasets, and have the same data types. All
        datasets' `obs_names` must have the same name and dtype, and similarly
        for `var_names`.
        
        The one exception to the var "same data type" rule: if a column is Enum
        in some datasets and Categorical in others, or Enum in all datasets but
        with different categories in each dataset, that column will be retained
        as an Enum column (with the union of the categories) in the
        concatenated var.
        
        Args:
            datasets: one or more Pseudobulk datasets to concatenate with this
                      one
            *more_datasets: additional Pseudobulk datasets to concatenate with
                            this one, specified as positional arguments
            flexible: whether to subset to cells and columns of obs and var
                      common to all datasets before concatenating, rather than
                      raising an error on any mismatches
        
        Returns:
            The concatenated Pseudobulk dataset.
        """
        # Check inputs
        if isinstance(datasets, Pseudobulk):
            datasets = datasets,
        datasets = (self,) + datasets + more_datasets
        if len(datasets) == 1:
            error_message = \
                'need at least one other Pseudobulk dataset to concatenate'
            raise ValueError(error_message)
        check_types(datasets[1:], 'datasets', Pseudobulk,
                    'Pseudobulk datasets')
        check_type(flexible, 'flexible', bool, 'Boolean')
        # Check that cell types match across all datasets
        if not all(self.keys() == dataset.keys() for dataset in datasets[1:]):
            error_message = \
                'not all Pseudobulk datasets have the same cell types'
            raise ValueError(error_message)
        # Perform either flexible or non-flexible concatenation
        X = {}
        obs = {}
        var = {}
        for cell_type in self:
            if flexible:
                # Check that `var_names` and `obs_names` have the same name and
                # data type for each cell type across all datasets
                var_names_name = self._var[cell_type][:, 0].name
                if not all(dataset._var[cell_type][:, 0] == var_names_name
                           for dataset in datasets[1:]):
                    error_message = (
                        f'[{cell_type!r}] not all Pseudobulk datasets have '
                        f'the same name for the first column of var (the '
                        f'var_names column)')
                    raise ValueError(error_message)
                obs_names_name = self._obs[cell_type][:, 0].name
                if not all(dataset._obs[cell_type][:, 0].name == obs_names_name
                           for dataset in datasets[1:]):
                    error_message = (
                        f'[{cell_type!r}] not all Pseudobulk datasets have '
                        f'the same name for the first column of obs (the '
                        f'obs_names column)')
                    raise ValueError(error_message)
                var_names_dtype = self._var[cell_type][:, 0].dtype
                if not all(dataset._var[cell_type][:, 0].dtype ==
                           var_names_dtype for dataset in datasets[1:]):
                    error_message = (
                        f'[{cell_type!r}] not all Pseudobulk datasets have '
                        f'the same data type for the first column of var (the '
                        f'var_names column)')
                    raise TypeError(error_message)
                obs_names_dtype = self._obs[cell_type][:, 0].dtype
                if not all(dataset._obs[cell_type][:, 0].dtype ==
                           obs_names_dtype for dataset in datasets[1:]):
                    error_message = (
                        f'[{cell_type!r}] not all Pseudobulk datasets have '
                        f'the same data type for the first column of obs (the '
                        f'obs_names column)')
                    raise TypeError(error_message)
                # Subset to genes in common across all datasets
                genes_in_common = self._obs[cell_type][:, 0]\
                    .filter(self._obs[cell_type][:, 0]
                            .is_in(pl.concat([dataset._obs[cell_type][:, 0]
                                              for dataset in datasets[1:]])))
                if len(genes_in_common) == 0:
                    error_message = (
                        f'[{cell_type!r}] no genes are shared across all '
                        f'Pseudobulk datasets')
                    raise ValueError(error_message)
                cell_type_X = []
                cell_type_obs = []
                for dataset in datasets:
                    gene_indices = dataset._getitem_process(
                        genes_in_common, 1, dataset._obs[cell_type])
                    cell_type_X.append(
                        dataset._X[cell_type][:, gene_indices.to_numpy()])
                    cell_type_obs.append(dataset._obs[cell_type][gene_indices])
                # Subset to columns of obs that are identical in all datasets
                # after this subsetting
                obs_columns_in_common = [
                    column.name for column in cell_type_obs[0][:, 1:]
                    if all(column.name in dataset_cell_type_obs and
                           dataset_cell_type_obs[column.name].equals(column)
                           for dataset_cell_type_obs in cell_type_obs[1:])]
                cell_type_obs = cell_type_obs[0]
                cell_type_obs = cell_type_obs.select(cell_type_obs.columns[0],
                                                     obs_columns_in_common)
                # Subset to columns of var that are present in all datasets,
                # and have the same data types. Also include columns of var
                # that are Enum in some datasets and Categorical in others, or
                # Enum in all datasets but with different categories in each
                # dataset; cast these to Categorical.
                var_mismatched_categoricals = {
                    column for column, dtype in self._var[cell_type][:, 1:]
                    .select(pl.col(pl.Categorical, pl.Enum)).schema.items()
                    if all(column in dataset._var[cell_type] and
                           dataset._var[cell_type][column].dtype in
                           (pl.Categorical, pl.Enum)
                           for dataset in datasets[1:]) and
                       not all(dataset._var[cell_type][column].dtype == dtype
                               for dataset in datasets[1:])}
                var_columns_in_common = [
                    column
                    for column, dtype in islice(
                        self._var[cell_type].schema.items(), 1)
                    if column in var_mismatched_categoricals or
                       all(column in dataset[cell_type]._var and
                           dataset._var[cell_type][column].dtype == dtype
                           for dataset in datasets[1:])]
                cast_dict = {column: pl.Enum(
                    pl.concat([dataset._var[cell_type][column]
                              .cat.get_categories() for dataset in datasets])
                    .unique(maintain_order=True))
                    for column in var_mismatched_categoricals}
                cell_type_var = [
                    dataset._var[cell_type]
                    .cast(cast_dict)
                    .select(var_columns_in_common) for dataset in datasets]
            else:  # non-flexible
                # Check that all obs are identical
                cell_type_obs = self._obs[cell_type]
                for dataset in datasets[1:]:
                    if not dataset._obs[cell_type].equals(cell_type_obs):
                        error_message = (
                            f'[{cell_type!r}] all Pseudobulk datasets must '
                            f'have the same obs, unless flexible=True')
                        raise ValueError(error_message)
                # Check that all var have the same columns and data types
                schema = self._var[cell_type].schema
                for dataset in datasets[1:]:
                    if dataset._var[cell_type].schema != schema:
                        error_message = (
                            f'[{cell_type!r}] all Pseudobulk datasets must '
                            f'have the same columns in var, with the same '
                            f'data types, unless flexible=True')
                        raise ValueError(error_message)
                cell_type_X = [dataset._X[cell_type] for dataset in datasets]
                cell_type_var = [dataset._var[cell_type]
                                 for dataset in datasets]
            # Concatenate
            X[cell_type] = np.hstack(cell_type_X)
            var[cell_type] = pl.concat(cell_type_var)
            obs[cell_type] = cell_type_obs
        return Pseudobulk(X=X, obs=obs, var=var)
    
    def _get_columns(self,
                     obs_or_var_name: Literal['obs', 'var'],       
                     columns: PseudobulkColumn | None |
                              Sequence[PseudobulkColumn | None],
                     variable_name: str,
                     dtypes: pl.datatypes.classes.DataTypeClass | str |
                             tuple[pl.datatypes.classes.DataTypeClass | str,
                                   ...],
                     custom_error: str | None = None,
                     allow_None: bool = True,
                     allow_null: bool = False,
                     cell_types: Sequence[str] | None = None) -> \
            dict[str, pl.Series | None]:
        """
        Get a column of the same length as obs or var for each cell type.
        
        Args:
            obs_or_var_name: the name of the DataFrame the column is with
                             respect to, i.e. `'obs'` or `'var'`
            columns: a string naming a column of each cell type's obs/var, a
                     polars expression that evaluates to a single column when 
                     applied to each cell type's obs/var, a polars Series or 
                     NumPy array of the same length as each cell type's 
                     obs/var, or a function that takes in two arguments, `self`
                     and a cell type, and returns a polars Series or NumPy 
                     array of the same length as obs/var. Or, a sequence of
                     any combination of these for each cell type. May also be
                     None (or a Sequence containing None) if `allow_None=True`.
            variable_name: the name of the variable corresponding to `columns`
            dtypes: the required dtype(s) of the column
            custom_error: a custom error message for when (an element of)
                          `columns` is a string and is not found in obs/var;
                          use `{}` as a placeholder for the name of the column
            allow_None: whether to allow `columns` or its elements to be None
            allow_null: whether to allow `columns` to contain null values
            cell_types: a list of cell types; if None, use all cell types. If
                        specified and `column` is a Sequence, `column` and
                        `cell_types` should have the same length.
        
        Returns:
            A dictionary mapping each cell type to a polars Series of the same
            length as the cell type's obs/var. Or, if `columns` is None (or if
            some elements are None), a dict where some or all values are None.
        """
        obs_or_var = self._obs if obs_or_var_name == 'obs' else self._var
        if cell_types is None:
            cell_types = self._X
        if columns is None:
            if not allow_None:
                error_message = f'{variable_name} is None'
                raise TypeError(error_message)
            return {cell_type: None for cell_type in cell_types}
        columns_dict = {}
        if isinstance(columns, str):
            for cell_type in cell_types:
                if columns not in obs_or_var[cell_type]:
                    error_message = (
                        f'{columns!r} is not a column of '
                        f'{obs_or_var_name}[{cell_type!r}]'
                        if custom_error is None else
                        custom_error.format(f'{columns!r}'))
                    raise ValueError(error_message)
                columns_dict[cell_type] = obs_or_var[cell_type][columns]
        elif isinstance(columns, pl.Expr):
            for cell_type in cell_types:
                columns_dict[cell_type] = obs_or_var[cell_type].select(columns)
                if columns_dict[cell_type].width > 1:
                    error_message = (
                        f'{variable_name} is a polars expression that expands '
                        f'to {columns_dict[cell_type].width:,} columns rather '
                        f'than 1 for cell type {cell_type!r}')
                    raise ValueError(error_message)
                columns_dict[cell_type] = columns_dict[cell_type].to_series()
        elif isinstance(columns, pl.Series):
            for cell_type in cell_types:
                if len(columns) != len(obs_or_var[cell_type]):
                    error_message = (
                        f'{variable_name} is a polars Series of length '
                        f'{len(columns):,}, which differs from the length of '
                        f'{obs_or_var_name}[{cell_type!r}] '
                        f'({len(obs_or_var[cell_type]):,})')
                    raise ValueError(error_message)
                columns_dict[cell_type] = columns
        elif isinstance(columns, np.ndarray):
            for cell_type in cell_types:
                if len(columns) != len(obs_or_var[cell_type]):
                    error_message = (
                        f'{variable_name} is a NumPy array of length '
                        f'{len(columns):,}, which differs from the length of '
                        f'{obs_or_var_name}[{cell_type!r}] '
                        f'({len(obs_or_var[cell_type]):,})')
                    raise ValueError(error_message)
                columns_dict[cell_type] = pl.Series(variable_name, columns)
        elif callable(columns):
            function = columns
            for cell_type in cell_types:
                columns = function(self, cell_type)
                if isinstance(columns, np.ndarray):
                    if columns.ndim != 1:
                        error_message = (
                            f'{variable_name} is a function that returns a '
                            f'{columns.ndim:,}D NumPy array, but must return '
                            f'a polars Series or 1D NumPy array')
                        raise ValueError(error_message)
                    columns = pl.Series(variable_name, columns)
                elif not isinstance(columns, pl.Series):
                    error_message = (
                        f'{variable_name} is a function that returns a '
                        f'variable of type {type(columns).__name__}, but must '
                        f'return a polars Series or 1D NumPy array')
                    raise TypeError(error_message)
                if len(columns) != len(obs_or_var[cell_type]):
                    error_message = (
                        f'{variable_name} is a function that returns a column '
                        f'of length {len(columns):,} for cell type '
                        f'{cell_type!r}, which differs from the length of '
                        f'{obs_or_var_name}[{cell_type!r}] '
                        f'({len(obs_or_var[cell_type]):,})')
                    raise ValueError(error_message)
                columns_dict[cell_type] = columns
        elif isinstance(columns, Sequence):
            if len(columns) != len(cell_types):
                error_message = (
                    f'{variable_name} is a sequence of length '
                    f'{len(columns):,}, which differs from the number of cell '
                    f'types ({len(cell_types):,})')
                raise ValueError(error_message)
            if not allow_None and any(column is None for column in columns):
                error_message = \
                    f'{variable_name} contains an element that is None'
                raise TypeError(error_message)
            for index, (column, cell_type) in \
                    enumerate(zip(columns, cell_types)):
                if isinstance(column, str):
                    if column not in obs_or_var[cell_type]:
                        error_message = (
                            f'{column!r} is not a column of '
                            f'{obs_or_var_name}[{cell_type!r}]'
                            if custom_error is None else
                            custom_error.format(f'{column!r}'))
                        raise ValueError(error_message)
                    columns_dict[cell_type] = obs_or_var[cell_type][column]
                elif isinstance(column, pl.Expr):
                    columns_dict[cell_type] = \
                        obs_or_var[cell_type].select(column)
                    if columns[cell_type].width > 1:
                        error_message = (
                            f'{variable_name}[{index}] is a polars expression '
                            f'that expands to {columns[cell_type].width:,} '
                            f'columns rather than 1 for cell type '
                            f'{cell_type!r}')
                        raise ValueError(error_message)
                    columns_dict[cell_type] = \
                        columns_dict[cell_type].to_series()
                elif isinstance(column, pl.Series):
                    if len(column) != len(obs_or_var[cell_type]):
                        error_message = (
                            f'{variable_name}[{index}] is a polars Series of '
                            f'length {len(column):,}, which differs from the '
                            f'length of {obs_or_var_name}[{cell_type!r}] '
                            f'({len(obs_or_var[cell_type]):,})')
                        raise ValueError(error_message)
                    columns_dict[cell_type] = column
                elif isinstance(column, np.ndarray):
                    if len(column) != len(obs_or_var[cell_type]):
                        error_message = (
                            f'{variable_name}[{index}] is a NumPy array of '
                            f'length {len(column):,}, which differs from the '
                            f'length of {obs_or_var_name}[{cell_type!r}] '
                            f'({len(obs_or_var[cell_type]):,})')
                        raise ValueError(error_message)
                    columns_dict[cell_type] = pl.Series(variable_name, column)
                elif callable(columns):
                    column = column(self, cell_type)
                    if isinstance(column, np.ndarray):
                        if column.ndim != 1:
                            error_message = (
                                f'{variable_name}[{index}] is a function that '
                                f'returns a {column.ndim:,}D NumPy array, but '
                                f'must return a polars Series or 1D NumPy '
                                f'array')
                            raise ValueError(error_message)
                        column = pl.Series(variable_name, column)
                    elif not isinstance(column, pl.Series):
                        error_message = (
                            f'{variable_name}[{index}] is a function that '
                            f'returns a variable of type '
                            f'{type(column).__name__}, but must return a '
                            f'polars Series or 1D NumPy array')
                        raise TypeError(error_message)
                    if len(column) != len(obs_or_var[cell_type]):
                        error_message = (
                            f'{variable_name}[{index}] is a function that '
                            f'returns a column of length {len(column):,} for '
                            f'cell type {cell_type!r}, which differs from the '
                            f'length of {obs_or_var_name}[{cell_type!r}] '
                            f'({len(obs_or_var[cell_type]):,})')
                        raise ValueError(error_message)
                    columns_dict[cell_type] = column
                else:
                    error_message = (
                        f'{variable_name}[{index}] must be a string column '
                        f'name, a polars expression or Series, a 1D NumPy '
                        f'array, or a function that returns any of these when '
                        f'applied to this Pseudobulk dataset and a given cell '
                        f'type, but has type {type(column).__name__!r}')
                    raise TypeError(error_message)
        else:
            error_message = (
                f'{variable_name} must be a string column name, a polars '
                f'expression or Series, a 1D NumPy array, or a function that '
                f'returns any of these when applied to this Pseudobulk '
                f'dataset and a given cell type, but has type '
                f'{type(columns).__name__!r}')
            raise TypeError(error_message)
        # Check dtypes
        if not isinstance(dtypes, tuple):
            dtypes = dtypes,
        for cell_type, column in columns_dict.items():
            base_type = column.dtype.base_type()
            for expected_type in dtypes:
                if base_type == expected_type or expected_type == 'integer' \
                        and base_type in pl.INTEGER_DTYPES or \
                        expected_type == 'floating-point' and \
                        base_type in pl.FLOAT_DTYPES:
                    break
            else:
                if len(dtypes) == 1:
                    dtypes = str(dtypes[0])
                elif len(dtypes) == 2:
                    dtypes = ' or '.join(map(str, dtypes))
                else:
                    dtypes = ', '.join(map(str, dtypes[:-1])) + ', or ' + \
                             str(dtypes[-1])
                error_message = (
                    f'{variable_name} must be {dtypes}, but has data type '
                    f'{base_type!r} for cell type {cell_type!r}')
                raise TypeError(error_message)
        # Check nulls, if `allow_null=False`
        if not allow_null:
            for cell_type, column in columns_dict.items():
                null_count = column.null_count()
                if null_count > 0:
                    error_message = (
                        f'{variable_name} contains {null_count:,} '
                        f'{plural("null value", null_count)} for cell type '
                        f'{cell_type!r}, but must not contain any')
                    raise ValueError(error_message)
        return columns_dict
    
    def filter_obs(self,
                   *predicates: str | pl.Expr | pl.Series |
                                Iterable[str | pl.Expr | pl.Series] | bool |
                                list[bool] | np.ndarray[1, np.bool_],
                   **constraints: Any) -> Pseudobulk:
        """
        Equivalent to `df.filter()` from polars, but applied to both obs and X
        for each cell type.
        
        Args:
            *predicates: one or more column names, expressions that evaluate to
                         Boolean Series, Boolean Series, lists of Booleans,
                         and/or 1D Boolean NumPy arrays
            **constraints: column filters: `name=value` filters to samples
                           where the column named `name` has the value `value`
        
        Returns:
            A new Pseudobulk dataset filtered to samples passing all the
            Boolean filters in `predicates` and `constraints`.
        """
        X = {}
        obs = {}
        for cell_type in self:
            obs[cell_type] = self._obs[cell_type]\
                .with_row_index('__Pseudobulk_index')\
                .filter(*predicates, **constraints)
            X[cell_type] = self._X[cell_type][
                obs[cell_type]['__Pseudobulk_index'].to_numpy()]
            obs[cell_type] = obs[cell_type].drop('__Pseudobulk_index')
        return Pseudobulk(X=X, obs=obs, var=self._var)
    
    def filter_var(self,
                   *predicates: pl.Expr | pl.Series | str |
                                Iterable[pl.Expr | pl.Series | str] | bool |
                                list[bool] | np.ndarray[1, np.bool_],
                   **constraints: Any) -> Pseudobulk:
        """
        Equivalent to `df.filter()` from polars, but applied to both var and X
        for each cell type.
        
        Args:
            *predicates: one or more column names, expressions that evaluate to
                         Boolean Series, Boolean Series, lists of Booleans,
                         and/or 1D Boolean NumPy arrays
            **constraints: column filters: `name=value` filters to genes
                           where the column named `name` has the value `value`
        
        Returns:
            A new Pseudobulk dataset filtered to genes passing all the
            Boolean filters in `predicates` and `constraints`.
        """
        X = {}
        var = {}
        for cell_type in self:
            var[cell_type] = self._var[cell_type]\
                .with_row_index('__Pseudobulk_index')\
                .filter(*predicates, **constraints)
            X[cell_type] = self._X[cell_type][
                :, var[cell_type]['__Pseudobulk_index'].to_numpy()]
            var[cell_type] = var[cell_type].drop('__Pseudobulk_index')
        return Pseudobulk(X=X, obs=self._obs, var=var)
    
    def select_obs(self,
                   *exprs: Scalar | pl.Expr | pl.Series |
                           Iterable[Scalar | pl.Expr | pl.Series],
                   **named_exprs: Scalar | pl.Expr | pl.Series) -> Pseudobulk:
        """
        Equivalent to `df.select()` from polars, but applied to each cell
        type's obs. obs_names will be automatically included as the first
        column, if not included explicitly.
        
        Args:
            *exprs: column(s) to select, specified as positional arguments.
                    Accepts expression input. Strings are parsed as column
                    names, other non-expression inputs are parsed as literals.
            **named_exprs: additional columns to select, specified as keyword
                           arguments. The columns will be renamed to the
                           keyword used.
        
        Returns:
            A new Pseudobulk dataset with
            obs[cell_type]=obs[cell_type].select(*exprs, **named_exprs) for all
            cell types in obs, and obs_names as the first column unless already
            included explicitly.
        """
        obs = {}
        for cell_type, cell_type_obs in self._obs.items():
            new_cell_type_obs = cell_type_obs.select(*exprs, **named_exprs)
            if cell_type_obs.columns[0] not in new_cell_type_obs:
                new_cell_type_obs = \
                    new_cell_type_obs.select(cell_type_obs[:, 0], pl.all())
            obs[cell_type] = new_cell_type_obs
        return Pseudobulk(X=self._X, obs=obs, var=self._var)
    
    def select_var(self,
                   *exprs: Scalar | pl.Expr | pl.Series |
                           Iterable[Scalar | pl.Expr | pl.Series],
                   **named_exprs: Scalar | pl.Expr | pl.Series) -> Pseudobulk:
        """
        Equivalent to `df.select()` from polars, but applied to each cell
        type's var. var_names will be automatically included as the first
        column, if not included explicitly.
        
        Args:
            *exprs: column(s) to select, specified as positional arguments.
                    Accepts expression input. Strings are parsed as column
                    names, other non-expression inputs are parsed as literals.
            **named_exprs: additional columns to select, specified as keyword
                           arguments. The columns will be renamed to the
                           keyword used.
        
        Returns:
            A new Pseudobulk dataset with
            var[cell_type]=var[cell_type].select(*exprs, **named_exprs) for all
            cell types in var, and var_names as the first column unless already
            included explicitly.
        """
        var = {}
        for cell_type, cell_type_var in self._var.items():
            new_cell_type_var = cell_type_var.select(*exprs, **named_exprs)
            if cell_type_var.columns[0] not in new_cell_type_var:
                new_cell_type_var = \
                    new_cell_type_var.select(cell_type_var[:, 0], pl.all())
            var[cell_type] = new_cell_type_var
        return Pseudobulk(X=self._X, obs=self._obs, var=var)
    
    def select_cell_types(self, cell_types: str, *more_cell_types: str) -> \
            Pseudobulk:
        """
        Create a new Pseudobulk dataset subset to the cell type(s) in
        `cell_types` and `more_cell_types`.
        
        Args:
            cell_types: cell type(s) to select
            *more_cell_types: additional cell types to select, specified as
                              positional arguments
        
        Returns:
            A new Pseudobulk dataset subset to the specified cell type(s).
        """
        cell_types = to_tuple(cell_types) + more_cell_types
        check_types(cell_types, 'cell_types', str, 'strings')
        for cell_type in cell_types:
            if cell_type not in self:
                error_message = (
                    f'tried to select {cell_type!r}, which is not a cell type '
                    f'in this Pseudobulk')
                raise ValueError(error_message)
        return Pseudobulk(X={cell_type: self._X[cell_type]
                             for cell_type in cell_types},
                          obs={cell_type: self._obs[cell_type]
                               for cell_type in cell_types},
                          var={cell_type: self._var[cell_type]
                               for cell_type in cell_types})
    
    def with_columns_obs(self,
                         *exprs: Scalar | pl.Expr | pl.Series |
                                 Iterable[Scalar | pl.Expr | pl.Series],
                         **named_exprs: Scalar | pl.Expr | pl.Series) -> \
            Pseudobulk:
        """
        Equivalent to `df.with_columns()` from polars, but applied to each cell
        type's obs.
        
        Args:
            *exprs: column(s) to add, specified as positional arguments.
                    Accepts expression input. Strings are parsed as column
                    names, other non-expression inputs are parsed as literals.
            **named_exprs: additional columns to add, specified as keyword
                           arguments. The columns will be renamed to the
                           keyword used.
        
        Returns:
            A new Pseudobulk dataset with
            obs[cell_type]=obs[cell_type].with_columns(*exprs, **named_exprs)
            for all cell types in obs.
        """
        return Pseudobulk(X=self._X, obs={
            cell_type: obs.with_columns(*exprs, **named_exprs)
            for cell_type, obs in self._obs.items()}, var=self._var)
    
    def with_columns_var(self,
                         *exprs: Scalar | pl.Expr | pl.Series |
                                 Iterable[Scalar | pl.Expr | pl.Series],
                         **named_exprs: Scalar | pl.Expr | pl.Series) -> \
            Pseudobulk:
        """
        Equivalent to `df.with_columns()` from polars, but applied to each cell
        type's var.
        
        Args:
            *exprs: column(s) to add, specified as positional arguments.
                    Accepts expression input. Strings are parsed as column
                    names, other non-expression inputs are parsed as literals.
            **named_exprs: additional columns to add, specified as keyword
                           arguments. The columns will be renamed to the
                           keyword used.
        
        Returns:
            A new Pseudobulk dataset with
            var[cell_type]=var[cell_type].with_columns(*exprs, **named_exprs)
            for all cell types in var.
        """
        return Pseudobulk(X=self._X, obs=self._obs, var={
            cell_type: var.with_columns(*exprs, **named_exprs)
            for cell_type, var in self._var.items()})

    def drop_obs(self,
                 columns: pl.type_aliases.ColumnNameOrSelector |
                          Iterable[pl.type_aliases.ColumnNameOrSelector],
                 *more_columns: pl.type_aliases.ColumnNameOrSelector) -> \
            Pseudobulk:
        """
        Create a new Pseudobulk dataset with `columns` and `more_columns`
        removed from obs.
        
        Args:
            columns: columns(s) to drop
            *more_columns: additional columns to drop, specified as
                           positional arguments
        
        Returns:
            A new Pseudobulk dataset with the column(s) removed.
        """
        columns = to_tuple(columns) + more_columns
        return Pseudobulk(X=self._X,
                          obs={cell_type: self._obs[cell_type].drop(columns)
                               for cell_type in self}, var=self._var)

    def drop_var(self,
                 columns: pl.type_aliases.ColumnNameOrSelector |
                          Iterable[pl.type_aliases.ColumnNameOrSelector],
                 *more_columns: pl.type_aliases.ColumnNameOrSelector) -> \
            Pseudobulk:
        """
        Create a new Pseudobulk dataset with `columns` and `more_columns`
        removed from var.
        
        Args:
            columns: columns(s) to drop
            *more_columns: additional columns to drop, specified as
                           positional arguments
        
        Returns:
            A new Pseudobulk dataset with the column(s) removed.
        """
        columns = to_tuple(columns) + more_columns
        return Pseudobulk(X=self._X, obs=self._obs,
                          var={cell_type: self._var[cell_type].drop(columns)
                               for cell_type in self})
    
    def drop_cell_types(self, cell_types: str, *more_cell_types: str) -> \
            Pseudobulk:
        """
        Create a new Pseudobulk dataset with `cell_types` and `more_cell_types`
        removed. Raises an error if all cell types would be dropped.
        
        Args:
            cell_types: cell type(s) to drop
            *more_cell_types: additional cell types to drop, specified as
                              positional arguments
        
        Returns:
            A new Pseudobulk dataset with the cell type(s) removed.
        """
        cell_types = set(to_tuple(cell_types)) | set(more_cell_types)
        check_types(cell_types, 'cell_types', str, 'strings')
        # noinspection PyTypeChecker
        original_cell_types = set(self)
        if not cell_types < original_cell_types:
            if cell_types == original_cell_types:
                error_message = 'all cell types would be dropped'
                raise ValueError(error_message)
            for cell_type in cell_types:
                if cell_type not in original_cell_types:
                    error_message = (
                        f'tried to drop {cell_type!r}, which is not a cell '
                        f'type in this Pseudobulk')
                    raise ValueError(error_message)
        new_cell_types = \
            [cell_type for cell_type in self if cell_type not in cell_types]
        return Pseudobulk(X={cell_type: self._X[cell_type]
                             for cell_type in new_cell_types},
                          obs={cell_type: self._obs[cell_type]
                               for cell_type in new_cell_types},
                          var={cell_type: self._var[cell_type]
                               for cell_type in new_cell_types})
    
    def rename_obs(self, mapping: dict[str, str] | Callable[[str], str]) -> \
            Pseudobulk:
        """
        Create a new Pseudobulk dataset with column(s) of obs renamed for each
        cell type.
        
        Args:
            mapping: the renaming to apply, either as a dictionary with the old
                     names as keys and the new names as values, or a function
                     that takes an old name and returns a new name
        
        Returns:
            A new Pseudobulk dataset with the column(s) of obs renamed.
        """
        return Pseudobulk(X=self._X, obs={
            cell_type: self._obs[cell_type].rename(mapping)
            for cell_type in self}, var=self._var)
    
    def rename_var(self, mapping: dict[str, str] | Callable[[str], str]) -> \
            Pseudobulk:
        """
        Create a new Pseudobulk dataset with column(s) of var renamed for each
        cell type.
        
        Args:
            mapping: the renaming to apply, either as a dictionary with the old
                     names as keys and the new names as values, or a function
                     that takes an old name and returns a new name
        
        Returns:
            A new Pseudobulk dataset with the column(s) of var renamed.
        """
        return Pseudobulk(X=self._X, obs=self._obs, var={
            cell_type: self._var[cell_type].rename(mapping)
            for cell_type in self})
    
    def rename_cell_types(self,
                          mapping: dict[str, str] | Callable[[str], str]) -> \
            Pseudobulk:
        """
        Create a new Pseudobulk dataset with cell type(s) renamed.
        
        Args:
            mapping: the renaming to apply, either as a dictionary with the old
                     cell type names as keys and the new names as values, or a
                     function that takes an old name and returns a new name
        
        Returns:
            A new Pseudobulk dataset with the cell type(s) renamed.
        """
        if isinstance(mapping, dict):
            new_cell_types = [mapping.get(cell_type, cell_type)
                              for cell_type in self._X]
        elif callable(mapping):
            new_cell_types = [mapping(cell_type) for cell_type in self._X]
        else:
            raise RuntimeError(f'mapping must be a dictionary or function, '
                               f'but has type {type(mapping).__name__!r}')
        return Pseudobulk(X={new_cell_type: X
                             for new_cell_type, X in
                             zip(new_cell_types, self._X.values())},
                          obs={new_cell_type: obs
                               for new_cell_type, obs in
                               zip(new_cell_types, self._obs.values())},
                          var={new_cell_type: var
                               for new_cell_type, var in
                               zip(new_cell_types, self._var.values())})
    
    def cast_X(self, dtype: np._typing.DTypeLike) -> Pseudobulk:
        """
        Cast each cell type's X to the specified data type.
        
        Args:
            dtype: a NumPy data type

        Returns:
            A new Pseudobulk dataset with each cell type's X cast to the
            specified data type.
        """
        return Pseudobulk(X={cell_type: self._X[cell_type].astype(dtype)
                             for cell_type in self},
                          obs=self._obs, var=self._var)
    
    def cast_obs(self,
                 dtypes: Mapping[pl.type_aliases.ColumnNameOrSelector |
                                 pl.type_aliases.PolarsDataType,
                                 pl.type_aliases.PolarsDataType] |
                         pl.type_aliases.PolarsDataType,
                 *,
                 strict: bool = True) -> Pseudobulk:
        """
        Cast column(s) of each cell type's obs to the specified data type(s).
        
        Args:
            dtypes: a mapping of column names (or selectors) to data types, or
                    a single data type to which all columns will be cast
            strict: whether to raise an error if a cast could not be done (for
                    instance, due to numerical overflow)

        Returns:
            A new Pseudobulk dataset with column(s) of each cell type's obs
            cast to the specified data type(s).
        """
        return Pseudobulk(X=self._X,
                          obs={cell_type: self._obs[cell_type].cast(
                              dtypes, strict=strict) for cell_type in self},
                          var=self._var)
    
    def cast_var(self,
                 dtypes: Mapping[pl.type_aliases.ColumnNameOrSelector |
                                 pl.type_aliases.PolarsDataType,
                                 pl.type_aliases.PolarsDataType] |
                         pl.type_aliases.PolarsDataType,
                 *,
                 strict: bool = True) -> Pseudobulk:
        """
        Cast column(s) of each cell type's var to the specified data type(s).
        
        Args:
            dtypes: a mapping of column names (or selectors) to data types, or
                    a single data type to which all columns will be cast
            strict: whether to raise an error if a cast could not be done (for
                    instance, due to numerical overflow)

        Returns:
            A new Pseudobulk dataset with column(s) of each cell type's var
            cast to the specified data type(s).
        """
        return Pseudobulk(X=self._X,
                          obs=self._obs,
                          var={cell_type: self._var[cell_type].cast(
                              dtypes, strict=strict) for cell_type in self})
    
    def join_obs(self,
                 other: pl.DataFrame,
                 on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
                 *,
                 left_on: str | pl.Expr | Sequence[str | pl.Expr] |
                          None = None,
                 right_on: str | pl.Expr | Sequence[str | pl.Expr] |
                           None = None,
                 suffix: str = '_right',
                 validate: Literal['m:m', 'm:1', '1:m', '1:1'] = 'm:m',
                 join_nulls: bool = False,
                 coalesce: bool = True) -> Pseudobulk:
        """
        Left join each cell type's obs with another DataFrame.
        
        Args:
            other: a polars DataFrame to join each cell type's obs with
            on: the name(s) of the join column(s) in both DataFrames
            left_on: the name(s) of the join column(s) in obs
            right_on: the name(s) of the join column(s) in `other`
            suffix: a suffix to append to columns with a duplicate name
            validate: checks whether the join is of the specified type. Can be:
                      - 'm:m' (many-to-many): the default, no checks performed.
                      - '1:1' (one-to-one): check that none of the values in
                        the join column(s) appear more than once in obs or more
                        than once in `other`.
                      - '1:m' (one-to-many): check that none of the values in
                        the join column(s) appear more than once in obs.
                      - 'm:1' (many-to-one): check that none of the values in
                        the join column(s) appear more than once in `other`.
            join_nulls: whether to include null as a valid value to join on.
                        By default, null values will never produce matches.
            coalesce: if True, coalesce each of the pairs of join columns
                      (the columns in `on` or `left_on`/`right_on`) from obs
                      and `other` into a single column, filling missing values
                      from one with the corresponding values from the other.
                      If False, include both as separate columns, adding
                      `suffix` to the join columns from `other`.
        
        Returns:
            A new Pseudobulk dataset with the columns from `other` joined to
            each cell type's obs.
        
        Note:
            If a column of `on`, `left_on` or `right_on` is Enum in obs and
            Categorical in `other` (or vice versa), or Enum in both but with
            different categories in each, that pair of columns will be
            automatically cast to a common Enum data type (with the union of
            the categories) before joining.
        """
        # noinspection PyTypeChecker
        check_type(other, 'other', pl.DataFrame, 'a polars DataFrame')
        if on is None:
            if left_on is None and right_on is None:
                error_message = (
                    f"either 'on' or both of 'left_on' and 'right_on' must be "
                    f"specified")
                raise ValueError(error_message)
            elif left_on is None:
                error_message = \
                    'right_on is specified, so left_on must be specified'
                raise ValueError(error_message)
            elif right_on is None:
                error_message = \
                    'left_on is specified, so right_on must be specified'
                raise ValueError(error_message)
        else:
            if left_on is not None:
                error_message = "'on' is specified, so 'left_on' must be None"
                raise ValueError(error_message)
            if right_on is not None:
                error_message = "'on' is specified, so 'right_on' must be None"
                raise ValueError(error_message)
        obs = {}
        for cell_type in self:
            left = self._obs[cell_type]
            right = other
            if on is None:
                left_columns = left.select(left_on)
                right_columns = right.select(right_on)
            else:
                left_columns = left.select(on)
                right_columns = right.select(on)
            left_cast_dict = {}
            right_cast_dict = {}
            for left_column, right_column in zip(left_columns, right_columns):
                left_dtype = left_column.dtype
                right_dtype = right_column.dtype
                if left_dtype == right_dtype:
                    continue
                if (left_dtype == pl.Enum or left_dtype == pl.Categorical) \
                        and (right_dtype == pl.Enum or
                             right_dtype == pl.Categorical):
                    common_dtype = \
                        pl.Enum(pl.concat([left_column.cat.get_categories(),
                                           right_column.cat.get_categories()])
                                .unique(maintain_order=True))
                    left_cast_dict[left_column.name] = common_dtype
                    right_cast_dict[right_column.name] = common_dtype
                else:
                    error_message = (
                        f'obs[{cell_type!r}][{left_column.name!r}] has data '
                        f'type {left_dtype.base_type()!r}, but '
                        f'other[{cell_type!r}][{right_column.name!r}] has '
                        f'data type {right_dtype.base_type()!r}')
                    raise TypeError(error_message)
            if left_cast_dict is not None:
                left = left.cast(left_cast_dict)
                right = right.cast(right_cast_dict)
            obs[cell_type] = \
                left.join(right, on=on, how='left', left_on=left_on,
                          right_on=right_on, suffix=suffix, validate=validate,
                          join_nulls=join_nulls, coalesce=coalesce)
            if len(obs[cell_type]) > len(self._obs[cell_type]):
                other_on = to_tuple(right_on if right_on is not None else on)
                assert other.select(other_on).is_duplicated().any()
                duplicate_column = other_on[0] if len(other_on) == 1 else \
                    next(column for column in other_on
                         if other[column].is_duplicated().any())
                error_message = (
                    f'other[{duplicate_column!r}] contains duplicate values, '
                    f'so it must be deduplicated before being joined on')
                raise ValueError(error_message)
        return Pseudobulk(X=self._X, obs=obs, var=self._var)
    
    def join_var(self,
                 other: pl.DataFrame,
                 on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
                 *,
                 left_on: str | pl.Expr | Sequence[str | pl.Expr] |
                          None = None,
                 right_on: str | pl.Expr | Sequence[str | pl.Expr] |
                           None = None,
                 suffix: str = '_right',
                 validate: Literal['m:m', 'm:1', '1:m', '1:1'] = 'm:m',
                 join_nulls: bool = False,
                 coalesce: bool = True) -> Pseudobulk:
        """
        Join each cell type's var with another DataFrame.
        
        Args:
            other: a polars DataFrame to join each cell type's var with
            on: the name(s) of the join column(s) in both DataFrames
            left_on: the name(s) of the join column(s) in var
            right_on: the name(s) of the join column(s) in `other`
            suffix: a suffix to append to columns with a duplicate name
            validate: checks whether the join is of the specified type. Can be:
                      - 'm:m' (many-to-many): the default, no checks performed.
                      - '1:1' (one-to-one): check that none of the values in
                        the join column(s) appear more than once in var or more
                        than once in `other`.
                      - '1:m' (one-to-many): check that none of the values in
                        the join column(s) appear more than once in var.
                      - 'm:1' (many-to-one): check that none of the values in
                        the join column(s) appear more than once in `other`.
            join_nulls: whether to include null as a valid value to join on.
                        By default, null values will never produce matches.
            coalesce: if True, coalesce each of the pairs of join columns
                      (the columns in `on` or `left_on`/`right_on`) from obs
                      and `other` into a single column, filling missing values
                      from one with the corresponding values from the other.
                      If False, include both as separate columns, adding
                      `suffix` to the join columns from `other`.
        
        Returns:
            A new Pseudobulk dataset with the columns from `other` joined to
            each cell type's var.
        
        Note:
            If a column of `on`, `left_on` or `right_on` is Enum in obs and
            Categorical in `other` (or vice versa), or Enum in both but with
            different categories in each, that pair of columns will be
            automatically cast to a common Enum data type (with the union of
            the categories) before joining.
        """
        check_type(other, 'other', pl.DataFrame, 'a polars DataFrame')
        if on is None:
            if left_on is None and right_on is None:
                error_message = (
                    "either 'on' or both of 'left_on' and 'right_on' must be "
                    "specified")
                raise ValueError(error_message)
            elif left_on is None:
                error_message = \
                    'right_on is specified, so left_on must be specified'
                raise ValueError(error_message)
            elif right_on is None:
                error_message = \
                    'left_on is specified, so right_on must be specified'
                raise ValueError(error_message)
        else:
            if left_on is not None:
                error_message = "'on' is specified, so 'left_on' must be None"
                raise ValueError(error_message)
            if right_on is not None:
                error_message = "'on' is specified, so 'right_on' must be None"
                raise ValueError(error_message)
        var = {}
        for cell_type in self:
            left = self._var[cell_type]
            right = other
            if on is None:
                left_columns = left.select(left_on)
                right_columns = right.select(right_on)
            else:
                left_columns = left.select(on)
                right_columns = right.select(on)
            left_cast_dict = {}
            right_cast_dict = {}
            for left_column, right_column in zip(left_columns, right_columns):
                left_dtype = left_column.dtype
                right_dtype = right_column.dtype
                if left_dtype == right_dtype:
                    continue
                if (left_dtype == pl.Enum or left_dtype == pl.Categorical) \
                        and (right_dtype == pl.Enum or
                             right_dtype == pl.Categorical):
                    common_dtype = \
                        pl.Enum(pl.concat([left_column.cat.get_categories(),
                                           right_column.cat.get_categories()])
                                .unique(maintain_order=True))
                    left_cast_dict[left_column.name] = common_dtype
                    right_cast_dict[right_column.name] = common_dtype
                else:
                    error_message = (
                        f'var[{cell_type!r}][{left_column.name!r}] has data '
                        f'type {left_dtype.base_type()!r}, but '
                        f'other[{cell_type!r}][{right_column.name!r}] has '
                        f'data type {right_dtype.base_type()!r}')
                    raise TypeError(error_message)
            if left_cast_dict is not None:
                left = left.cast(left_cast_dict)
                right = right.cast(right_cast_dict)
            var[cell_type] = \
                left.join(right, on=on, how='left', left_on=left_on,
                          right_on=right_on, suffix=suffix, validate=validate,
                          join_nulls=join_nulls, coalesce=coalesce)
            if len(var[cell_type]) > len(self._var[cell_type]):
                other_on = to_tuple(right_on if right_on is not None else on)
                assert other.select(other_on).is_duplicated().any()
                duplicate_column = other_on[0] if len(other_on) == 1 else \
                    next(column for column in other_on
                         if other[column].is_duplicated().any())
                error_message = (
                    f'other[{duplicate_column!r}] contains duplicate values, '
                    f'so it must be deduplicated before being joined on')
                raise ValueError(error_message)
        return Pseudobulk(X=self._X, obs=self._obs, var=var)
    
    def peek_obs(self, cell_type: str | None = None, row: int = 0) -> None:
        """
        Print a row of obs (the first row, by default) for a cell type (the
        first cell type, by default) with each column on its own line.
        
        Args:
            cell_type: the cell type to print the row for, or None to use the
                       first cell type
            row: the index of the row to print
        """
        if cell_type is None:
            cell_type = next(iter(self._obs))
        with pl.Config(tbl_rows=-1):
            print(self._obs[cell_type][row].unpivot(variable_name='column'))
    
    def peek_var(self, cell_type: str | None = None, row: int = 0) -> None:
        """
        Print a row of var (the first row, by default) with each column on its
        own line.
        
        Args:
            cell_type: the cell type to print the row for, or None to use the
                       first cell type
            row: the index of the row to print
        """
        if cell_type is None:
            cell_type = next(iter(self._var))
        with pl.Config(tbl_rows=-1):
            print(self._var[cell_type][row].unpivot(variable_name='column'))
    
    def subsample_obs(self,
                      n: int | np.integer | None = None,
                      *,
                      fraction: int | float | np.integer | np.floating |
                                None = None,
                      by_column: PseudobulkColumn | None |
                                 Sequence[PseudobulkColumn | None] = None,
                      subsample_column: str | None = None,
                      seed: int | np.integer | None = 0) -> Pseudobulk:
        """
        Subsample a specific number or fraction of samples.
        
        Args:
            n: the number of samples to return; mutually exclusive with
               `fraction`
            fraction: the fraction of samples to return; mutually exclusive
                      with `n`
            by_column: an optional String, Categorical, Enum, or integer column
                       of obs to subsample by. Can be None, a column name, a
                       polars expression, a polars Series, a 1D NumPy array, or
                       a function that takes in this Pseudobulk dataset and a
                       cell type and returns a polars Series or 1D NumPy array.
                       Can also be a sequence of any combination of these for
                       each cell type. Specifying `by_column` ensures that the
                       same fraction of cells with each value of `by_column`
                       are subsampled. When combined with `n`, to make sure the
                       total number of samples is exactly `n`, some of the
                       smallest groups may be oversampled by one element, or
                       some of the largest groups can be undersampled by one
                       element. Can contain null entries: the corresponding
                       samples will not be included in the result.
            subsample_column: an optional name of a Boolean column to add to
                              obs indicating the subsampled genes; if None,
                              subset to these genes instead
            seed: the random seed to use when subsampling; if None, do not set
                  a seed
        
        Returns:
            A new Pseudobulk dataset subset to the subsampled cells, or if
            `subsample_column` is not None, the full dataset with
            `subsample_column` added to obs.
        """
        if n is not None:
            check_type(n, 'n', int, 'a positive integer')
            check_bounds(n, 'n', 1)
        elif fraction is not None:
            check_type(fraction, 'fraction', float,
                       'a floating-point number between 0 and 1')
            check_bounds(fraction, 'fraction', 0, 1, left_open=True,
                         right_open=True)
        else:
            error_message = 'one of n and fraction must be specified'
            raise ValueError(error_message)
        if n is not None and fraction is not None:
            error_message = 'only one of n and fraction must be specified'
            raise ValueError(error_message)
        by_column = self._get_columns(
            'obs', by_column, 'by_column',
            (pl.String, pl.Categorical, pl.Enum, 'integer'), allow_null=True)
        if subsample_column is not None:
            check_type(subsample_column, 'subsample_column', str, 'a string')
            for cell_type, obs in self._obs.items():
                if subsample_column in obs:
                    error_message = (
                        f'subsample_column {subsample_column!r} is already a '
                        f'column of obs[{cell_type!r}]')
                    raise ValueError(error_message)
        if seed is not None:
            check_type(seed, 'seed', int, 'an integer')
        by = lambda expr, cell_type: \
            expr if by_column[cell_type] is None else \
            expr.over(by_column[cell_type])
        if by_column is not None and n is not None:
            # Reassign n to be a vector of sample sizes per group, broadcast to
            # the length of obs. The total sample size should exactly match the
            # original n; if necessary, oversample the smallest groups or
            # undersample the largest groups to make this happen.
            cell_type_n = {}
            for cell_type, cell_type_by_column in by_column.items():
                if cell_type_by_column is None:
                    cell_type_n[cell_type] = n
                else:
                    by_frame = cell_type_by_column.to_frame()
                    by_name = cell_type_by_column.name
                    group_counts = by_frame\
                        .group_by(by_name)\
                        .agg(pl.len(), n=(n * pl.len() / len(by_column))
                                         .round().cast(pl.Int32))\
                        .drop_nulls(by_name)
                    diff = n - group_counts['n'].sum()
                    if diff != 0:
                        group_counts = group_counts\
                            .sort('len', descending=diff < 0)\
                            .with_columns(n=pl.col.n +
                                          pl.int_range(pl.len()).lt(abs(diff))
                                          .cast(pl.Int32) *
                                            pl.lit(diff).sign())
                    cell_type_n[cell_type] = \
                        group_counts.join(by_frame, on=by_name)['n']
        # noinspection PyUnboundLocalVariable,PyUnresolvedReferences
        expressions = {
            cell_type: pl.int_range(pl.len())
                       .shuffle(seed=seed)
                       .pipe(by, cell_type=cell_type)
                       .lt((cell_type_n[cell_type] if by_column is not None
                            else n) if fraction is None else
                           fraction * pl.len().pipe(by, cell_type=cell_type))
                       for cell_type in self}
        if subsample_column is None:
            X = {}
            obs = {}
            for cell_type in self:
                obs[cell_type] = self._obs[cell_type]\
                    .with_row_index('__Pseudobulk_index')\
                    .filter(expressions[cell_type])
                X[cell_type] = self._X[cell_type][
                    obs[cell_type]['__Pseudobulk_index'].to_numpy()]
                obs[cell_type] = obs[cell_type].drop('__Pseudobulk_index')
            return Pseudobulk(X=X, obs=obs, var=self._var)
        else:
            return Pseudobulk(X=self._X, obs={
                cell_type: obs.with_columns(expressions[cell_type]
                                            .alias(subsample_column))
                for cell_type, obs in self._obs.items()}, var=self._var)
    
    def subsample_var(self,
                      n: int | np.integer | None = None,
                      *,
                      fraction: int | float | np.integer | np.floating |
                                None = None,
                      by_column: PseudobulkColumn | None |
                                 Sequence[PseudobulkColumn | None] = None,
                      subsample_column: str | None = None,
                      seed: int | np.integer | None = 0) -> Pseudobulk:
        """
        Subsample a specific number or fraction of genes.
        
        Args:
            n: the number of genes to return; mutually exclusive with
               `fraction`
            fraction: the fraction of genes to return; mutually exclusive with
                      `n`
            by_column: an optional String, Categorical, Enum, or integer column
                       of var to subsample by. Can be None, a column name, a
                       polars expression, a polars Series, a 1D NumPy array, or
                       a function that takes in this Pseudobulk dataset and a
                       cell type and returns a polars Series or 1D NumPy array.
                       Can also be a sequence of any combination of these for
                       each cell type. Specifying `by_column` ensures that the
                       same fraction of genes with each value of `by_column`
                       are subsampled. When combined with `n`, to make sure the
                       total number of samples is exactly `n`, some of the
                       smallest groups may be oversampled by one element, or
                       some of the largest groups may be undersampled by one
                       element. Can contain null entries: the corresponding
                       genes will not be included in the result.
            subsample_column: an optional name of a Boolean column to add to
                              var indicating the subsampled genes; if None,
                              subset to these genes instead
            seed: the random seed to use when subsampling; if None, do not set
                  a seed

        Returns:
            A new Pseudobulk dataset subset to the subsampled genes, or if
            `subsample_column` is not None, the full dataset with
            `subsample_column` added to var.
        """
        if n is not None:
            check_type(n, 'n', int, 'a positive integer')
            check_bounds(n, 'n', 1)
        elif fraction is not None:
            check_type(fraction, 'fraction', float,
                       'a floating-point number between 0 and 1')
            check_bounds(fraction, 'fraction', 0, 1, left_open=True,
                         right_open=True)
        else:
            error_message = 'one of n and fraction must be specified'
            raise ValueError(error_message)
        if n is not None and fraction is not None:
            error_message = 'only one of n and fraction must be specified'
            raise ValueError(error_message)
        by_column = self._get_columns(
            'var', by_column, 'by_column',
            (pl.String, pl.Categorical, pl.Enum, 'integer'), allow_null=True)
        if subsample_column is not None:
            check_type(subsample_column, 'subsample_column', str, 'a string')
            for cell_type, var in self._var.items():
                if subsample_column in var:
                    error_message = (
                        f'subsample_column {subsample_column!r} is already a '
                        f'column of var[{cell_type!r}]')
                    raise ValueError(error_message)
        if seed is not None:
            check_type(seed, 'seed', int, 'an integer')
        by = lambda expr, cell_type: \
            expr if by_column[cell_type] is None else \
            expr.over(by_column[cell_type])
        if by_column is not None and n is not None:
            # Reassign n to be a vector of sample sizes per group, broadcast to
            # the length of var. The total sample size should exactly match the
            # original n; if necessary, oversample the smallest groups or
            # undersample the largest groups to make this happen.
            cell_type_n = {}
            for cell_type, cell_type_by_column in by_column.items():
                if cell_type_by_column is None:
                    cell_type_n[cell_type] = n
                else:
                    by_frame = cell_type_by_column.to_frame()
                    by_name = cell_type_by_column.name
                    group_counts = by_frame\
                        .group_by(by_name)\
                        .agg(pl.len(), n=(n * pl.len() / len(by_column))
                                         .round().cast(pl.Int32))\
                        .drop_nulls(by_name)
                    diff = n - group_counts['n'].sum()
                    if diff != 0:
                        group_counts = group_counts\
                            .sort('len', descending=diff < 0)\
                            .with_columns(n=pl.col.n +
                                          pl.int_range(pl.len()).lt(abs(diff))
                                          .cast(pl.Int32) *
                                            pl.lit(diff).sign())
                    cell_type_n[cell_type] = \
                        group_counts.join(by_frame, on=by_name)['n']
        # noinspection PyUnboundLocalVariable,PyUnresolvedReferences
        expressions = {
            cell_type: pl.int_range(pl.len())
                       .shuffle(seed=seed)
                       .pipe(by, cell_type=cell_type)
                       .lt((cell_type_n[cell_type] if by_column is not None
                            else n) if fraction is None else
                           fraction * pl.len().pipe(by, cell_type=cell_type))
                       for cell_type in self}
        if subsample_column is None:
            X = {}
            var = {}
            for cell_type in self:
                var[cell_type] = self._var[cell_type]\
                    .with_row_index('__Pseudobulk_index')\
                    .filter(expressions[cell_type])
                X[cell_type] = self._X[cell_type][
                    :, var[cell_type]['__Pseudobulk_index'].to_numpy()]
                var[cell_type] = var[cell_type].drop('__Pseudobulk_index')
            return Pseudobulk(X=X, obs=self._obs, var=var)
        else:
            return Pseudobulk(X=self._X, obs=self._obs, var={
                cell_type: var.with_columns(expressions[cell_type]
                                            .alias(subsample_column))
                for cell_type, var in self._var.items()})
    
    def pipe(self,
             function: Callable[[Pseudobulk, ...], Any],
             *args: Any,
             **kwargs: Any) -> Any:
        """
        Apply a function to a Pseudobulk dataset.
        
        Args:
            function: the function to apply
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            function(self, *args, **kwargs)
        """
        return function(self, *args, **kwargs)
    
    def pipe_X(self,
               function: Callable[[np.ndarray[2, np.integer | np.floating],
                                   ...],
                                  np.ndarray[2, np.integer | np.floating]],
               *args: Any,
               **kwargs: Any) -> Pseudobulk:
        """
        Apply a function to a Pseudobulk dataset's X.
        
        Args:
            function: the function to apply to X
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new Pseudobulk dataset where the function has been applied to X.
        """
        return Pseudobulk(X={cell_type: function(self._X, *args, **kwargs)
                             for cell_type in self},
                          obs=self._obs, var=self._var)
    
    def pipe_obs(self,
                 function: Callable[[pl.DataFrame, ...], pl.DataFrame],
                 *args: Any,
                 **kwargs: Any) -> Pseudobulk:
        """
        Apply a function to a Pseudobulk dataset's obs for each cell type.
        
        Args:
            function: the function to apply to each cell type's obs
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new Pseudobulk dataset where the function has been applied to
            each cell type's obs.
        """
        return Pseudobulk(X=self._X, obs={
            cell_type: function(self._obs[cell_type], *args, **kwargs)
            for cell_type in self}, var=self._var)
    
    def pipe_var(self,
                 function: Callable[[pl.DataFrame, ...], pl.DataFrame],
                 *args: Any,
                 **kwargs: Any) -> Pseudobulk:
        """
        Apply a function to a Pseudobulk dataset's var for each cell type.
        
        Args:
            function: the function to apply to each cell type's var
            *args: the positional arguments to the function
            **kwargs: the keyword arguments to the function

        Returns:
            A new Pseudobulk dataset where the function has been applied to
            each cell type's var.
        """
        return Pseudobulk(X=self._X, obs=self._obs, var={
            cell_type: function(self._var[cell_type], *args, **kwargs)
            for cell_type in self})
    
    def qc(self,
           case_control_column: PseudobulkColumn | None |
                                Sequence[PseudobulkColumn | None],
           *,
           custom_filter: PseudobulkColumn | None |
                          Sequence[PseudobulkColumn | None] = None,
           min_num_cells: int | np.integer | None = 10,
           max_standard_deviations: int | float | np.integer | np.floating |
                                    None = 3,
           min_nonzero_fraction: int | float | np.integer | np.floating |
                                 None = 0.8,
           error_if_negative_counts: bool = True,
           allow_float: bool = False,
           verbose: bool = True) -> Pseudobulk:
        """
        Subsets each cell type to samples passing quality control (QC).
        This is different from `SingleCell.qc()`, which (for memory efficiency)
        just adds a Boolean column to obs of which cells passed QC.
        
        For each cell type, filter, in order, to:
        - samples with at least `min_num_cells` cells of that type
          (default: 10)
        - samples for whom the number of genes with 0 counts is at most
          `max_standard_deviations` standard deviations above the mean
          (default: 3)
        - genes with at least 1 count in `100 * min_nonzero_fraction`% of
          controls AND `100 * min_nonzero_fraction`% of cases (default: 80%),
          or if `case_control_column` is None, at least one count in
          `100 * min_nonzero_fraction`% of samples
        
        Args:
            case_control_column: an optional column of obs with case-control
                                 labels; set to None for non-case-control data.
                                 Can be None, a column name, a polars
                                 expression, a polars Series, a 1D NumPy array,
                                 or a function that takes in this Pseudobulk
                                 dataset and a cell type and returns a polars
                                 Series or 1D NumPy array. Can also be a
                                 sequence of any combination of these for each
                                 cell type. Not used when
                                 `min_nonzero_fraction` is None. Must be
                                 Boolean, integer, floating-point, or Enum with
                                 cases = 1/True and controls = 0/False. Can
                                 contain null entries: the corresponding
                                 samples will fail QC.
            custom_filter: an optional Boolean column of obs containing a
                           filter to apply on top of the other QC filters; True
                           elements will be kept. Can be None, a column name, a
                           polars expression, a polars Series, a 1D NumPy
                           array, or a function that takes in this Pseudobulk
                           dataset and a cell type and returns a polars Series
                           or 1D NumPy array. Can also be a sequence of any
                           combination of these for each cell type.
            min_num_cells: if not None, filter to samples with ≥ this many
                           cells of each cell type
            max_standard_deviations: if not None, filter to samples for whom
                                     the number of genes with 0 counts is at
                                     most this many standard deviations above
                                     the mean
            min_nonzero_fraction: if not None, filter to genes with at least
                                  one count in this fraction of controls AND
                                  this fraction of cases (or if
                                  `case_control_column` is None, at least one
                                  count in this fraction of samples)
            error_if_negative_counts: if True, raise an error if any counts are
                                      negative
            allow_float: if False, raise an error if `X.dtype` is
                         floating-point (suggesting the user may not be using
                         the raw counts); if True, disable this sanity check
            verbose: whether to print how many samples and genes were filtered
                     out at each step of the QC process
        
        Returns:
            A new Pseudobulk dataset with each cell type's X, obs and var
            subset to samples and genes passing QC.
        """
        # Check inputs
        case_control_column = self._get_columns(
            'obs', case_control_column, 'case_control_column',
            (pl.Boolean, 'integer', 'floating-point', pl.Enum),
            allow_null=True)
        custom_filter = self._get_columns(
            'obs', custom_filter, 'custom_filter', pl.Boolean)
        if min_num_cells is not None:
            check_type(min_num_cells, 'min_num_cells', int,
                       'a positive integer')
            check_bounds(min_num_cells, 'min_num_cells', 1)
        if max_standard_deviations is not None:
            check_type(max_standard_deviations, 'max_standard_deviations',
                       (int, float), 'a positive number')
            check_bounds(max_standard_deviations, 'max_standard_deviations', 0,
                         left_open=True)
        if min_nonzero_fraction is not None:
            check_type(min_nonzero_fraction, 'min_nonzero_fraction',
                       (int, float), 'a number between 0 and 1, inclusive')
            check_bounds(min_nonzero_fraction, 'min_nonzero_fraction', 0, 1)
        check_type(error_if_negative_counts, 'error_if_negative_counts', bool,
                   'Boolean')
        check_type(allow_float, 'allow_float', bool, 'Boolean')
        check_type(verbose, 'verbose', bool, 'Boolean')
        # If `error_if_negative_counts=True`, raise an error if X has any
        # negative values
        if error_if_negative_counts:
            for cell_type in self:
                if self._X[cell_type].ravel().min() < 0:
                    error_message = f'X[{cell_type!r}] has negative counts'
                    raise ValueError(error_message)
        # If `allow_float=False`, raise an error if `X` is floating-point
        if not allow_float:
            for cell_type in self:
                dtype = self._X[cell_type].dtype
                if np.issubdtype(dtype, np.floating):
                    error_message = (
                        f"qc() requires raw counts but X[{cell_type!r}].dtype "
                        f"is {dtype!r}, a floating-point data type; if you "
                        f"are sure that all values are raw integer counts, "
                        f"i.e. that (X[{cell_type!r}].data == "
                        f"X[{cell_type!r}].data.astype(int)).all(), then set "
                        f"allow_float=True (or just cast X to an integer data "
                        f"type).")
                    raise TypeError(error_message)
        if verbose:
            print()
        X_qced, obs_qced, var_qced = {}, {}, {}
        for cell_type, (X, obs, var) in self.items():
            if verbose:
                print(f'[{cell_type}] Starting with {len(obs):,} samples and '
                      f'{len(var):,} genes.')
            # Apply the custom filter, if specified
            if custom_filter is not None:
                mask = custom_filter[cell_type]
                X = X[mask.to_numpy()]
                obs = obs.filter(mask)
                if verbose:
                    print(f'[{cell_type}] {len(obs):,} samples remain after '
                          f'applying the custom filter.')
            # Filter to samples with at least `min_num_cells` cells of this
            # cell type
            if min_num_cells is not None:
                if verbose:
                    print(f'[{cell_type}] Filtering to samples with at least '
                          f'{min_num_cells} {cell_type} cells...')
                sample_mask = obs['num_cells'] >= min_num_cells
                # noinspection PyUnresolvedReferences
                X = X[sample_mask.to_numpy()]
                obs = obs.filter(sample_mask)
                if verbose:
                    print(f'[{cell_type}] {len(obs):,} samples remain after '
                          f'filtering to samples with at least '
                          f'{min_num_cells} {cell_type} cells.')
            # Filter to samples where the number of genes with 0 counts is less
            # than `max_standard_deviations` standard deviations above the mean
            if max_standard_deviations is not None:
                if verbose:
                    print(f'[{cell_type}] Filtering to samples where the '
                          f'number of genes with 0 counts is '
                          f'<{max_standard_deviations} standard deviations '
                          f'above the mean...')
                num_zero_counts = (X == 0).sum(axis=1)
                sample_mask = num_zero_counts < num_zero_counts.mean() + \
                              max_standard_deviations * num_zero_counts.std()
                X = X[sample_mask]
                obs = obs.filter(sample_mask)
                if verbose:
                    print(f'[{cell_type}] {len(obs):,} samples remain after '
                          f'filtering to samples where the number of genes '
                          f'with 0 counts is <{max_standard_deviations} '
                          f'standard deviations above the mean.')
            # Filter to genes with at least 1 count in
            # `100 * min_nonzero_fraction`% of controls AND
            # `100 * min_nonzero_fraction`% of cases, or if
            # `case_control_column` is None (for this cell type), at least
            # one count in `100 * min_nonzero_fraction`% of samples
            if min_nonzero_fraction is not None:
                if case_control_column is not None and \
                        case_control_column[cell_type] is not None:
                    if verbose:
                        print(f'[{cell_type}] Filtering to genes with at '
                              f'least one count in '
                              f'{100 * min_nonzero_fraction}% of cases and '
                              f'{100 * min_nonzero_fraction}% of controls...')
                    case_control_mask = case_control_column[cell_type]
                    if case_control_mask.dtype != pl.Boolean:
                        if case_control_mask.dtype == pl.Enum:
                            categories = case_control_mask.cat.get_categories()
                            if len(categories) != 2:
                                suffix = 'y' if len(categories) == 1 else 'ies'
                                error_message = (
                                    f'case_control_column is an Enum column '
                                    f'with {len(categories):,} '
                                    f'categor{suffix} for cell type '
                                    f'{cell_type!r}, but must have 2 '
                                    f'(cases = 1, controls = 0)')
                                raise ValueError(error_message)
                            case_control_mask = case_control_mask.to_physical()
                        else:
                            unique_labels = case_control_mask.unique()
                            num_unique_labels = len(unique_labels)
                            if num_unique_labels != 2:
                                plural_string = \
                                    plural('unique value', num_unique_labels)
                                error_message = (
                                    f'case_control_column is a numeric column '
                                    f'with {num_unique_labels:,} '
                                    f'{plural_string} for cell type '
                                    f'{cell_type!r}, but must have 2 '
                                    f'(cases = 1, controls = 0)')
                                raise ValueError(error_message)
                            if not unique_labels.sort()\
                                    .equals(pl.Series([0, 1])):
                                error_message = (
                                    f'case_control_column is a numeric column '
                                    f'with 2 unique values for cell type '
                                    f'{cell_type!r}, {unique_labels[0]} and '
                                    f'{unique_labels[1]}, but must have '
                                    f'cases = 1 and controls = 0')
                                raise ValueError(error_message)
                        case_control_mask = case_control_mask.cast(pl.Boolean)
                    case_control_mask = case_control_mask.to_numpy()
                    gene_mask = (np.quantile(X[case_control_mask],
                                             1 - min_nonzero_fraction,
                                             axis=0) > 0) & \
                                (np.quantile(X[~case_control_mask],
                                             1 - min_nonzero_fraction,
                                             axis=0) > 0)
                    X = X[:, gene_mask]
                    var = var.filter(gene_mask)
                    if verbose:
                        print(f'[{cell_type}] {len(var):,} genes remain '
                              f'after filtering to genes with at least one '
                              f'count in {100 * min_nonzero_fraction}% of '
                              f'cases and {100 * min_nonzero_fraction}% of '
                              f'controls.')
                else:
                    if verbose:
                        print(f'[{cell_type}] Filtering to genes with at '
                              f'least one count in '
                              f'{100 * min_nonzero_fraction}% of samples...')
                    gene_mask = np.quantile(X, 1 - min_nonzero_fraction,
                                            axis=0) > 0
                    X = X[:, gene_mask]
                    var = var.filter(gene_mask)
                    if verbose:
                        print(f'[{cell_type}] {len(var):,} genes remain '
                              f'after filtering to genes with at least one '
                              f'count in {100 * min_nonzero_fraction}% of '
                              f'samples.')
            X_qced[cell_type] = X
            obs_qced[cell_type] = obs
            var_qced[cell_type] = var
            if verbose:
                print()
        return Pseudobulk(X=X_qced, obs=obs_qced, var=var_qced)
    
    @staticmethod
    def _calc_norm_factors(X: np.ndarray[2, np.integer | np.floating],
                           *,
                           logratio_trim: int | float | np.integer |
                                          np.floating = 0.3,
                           sum_trim: int | float | np.integer |
                                     np.floating = 0.05,
                           do_weighting: bool = True,
                           A_cutoff: int | float | np.integer |
                                     np.floating = -1e10) -> \
            np.ndarray[2, np.floating]:
        """
        A drop-in replacement for edgeR's calcNormFactors with method='TMM'.
        
        Results were verified to match edgeR to within floating-point error.
        
        Does not support the lib.size and refColumn arguments to
        calcNormFactors; these are both assumed to be NULL (the default) and
        will always be calculated internally.
        
        Args:
            X: a matrix of raw (read) counts
            logratio_trim: the amount of trim to use on log-ratios ("M"
                           values); must be between 0 and 1
            sum_trim: the amount of trim to use on the combined absolute levels
                      ("A" values); must be between 0 and 1
            do_weighting: whether to compute (asymptotic binomial precision)
                          weights
            A_cutoff: the cutoff on "A" values to use before trimming
        
        Returns:
            A 1D NumPy array with the norm factors for each column of X.
        """
        # Check inputs
        check_type(logratio_trim, 'logratio_trim', float,
                   'a floating-point number')
        check_bounds(logratio_trim, 'logratio_trim', 0, 1, left_open=True,
                     right_open=True)
        check_type(sum_trim, 'sum_trim', float, 'a floating-point number')
        check_bounds(sum_trim, 'sum_trim', 0, 1, left_open=True,
                     right_open=True)
        check_type(do_weighting, 'do_weighting', bool, 'Boolean')
        check_type(A_cutoff, 'A_cutoff', float, 'a floating-point number')
        
        # Degenerate cases
        if X.shape[0] == 0 or X.shape[1] == 1:
            return np.ones(X.shape[1])
        
        # Remove all-zero rows
        any_non_zero = (X != 0).any(axis=1)
        if not any_non_zero.all():
            X = X[any_non_zero]
        
        # Calculate library sizes
        lib_size = X.sum(axis=0)
        
        # Determine which column is the reference column
        f75 = np.quantile(X, 0.75, axis=0) / lib_size
        if f75.min() == 0:
            import warnings
            warning_message = 'one or more quantiles are zero'
            warnings.warn(warning_message)
        ref_column = np.argmax(np.sqrt(X).sum(axis=0)) \
            if np.median(f75) < 1e-20 else \
            np.argmin(np.abs(f75 - f75.mean()))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate the log ratio of expression accounting for library size
            normalized_X = X / lib_size
            logR_ = np.log2(normalized_X / normalized_X[:, [ref_column]])
            
            # Calculate absolute expression
            log_normalized_X = np.log2(normalized_X)
            absE_ = 0.5 * (log_normalized_X +
                           log_normalized_X[:, [ref_column]])
            
            # Calculate estimated asymptotic variance
            if do_weighting:
                sum_of_reciprocals = 1 / X + 1 / lib_size
                v_ = sum_of_reciprocals + sum_of_reciprocals[:, [ref_column]]
        
        # Remove infinite values, cutoff based on A
        finite_ = (logR_ != -np.inf) & (absE_ > A_cutoff)
        
        # Calculate the normalization factors
        factors = np.empty(X.shape[1])
        for i in range(X.shape[1]):
            finite = finite_[:, i]
            logR = logR_[finite, i]
            absE = absE_[finite, i]
            if np.abs(logR).max() < 1e-6:
                factors[i] = 1
                continue
            n = len(logR)
            loL = int(n * logratio_trim)
            hiL = n - loL
            loS = int(n * sum_trim)
            hiS = n - loS
            logR_rank = rankdata(logR)
            absE_rank = rankdata(absE)
            keep = (logR_rank >= loL + 1) & (logR_rank <= hiL) & \
                   (absE_rank >= loS + 1) & (absE_rank <= hiS)
            factors[i] = 2 ** np.average(logR[keep],
                                         weights=1 / v_[finite, i][keep]
                                                 if do_weighting else None)
        
        # Results will be missing if the two libraries share no features with
        # positive counts; in this case, set to unity
        np.nan_to_num(factors, copy=False, nan=1)
        
        # Factors should multiply to one
        factors /= np.exp(np.log(factors).mean())
        return factors
    
    def CPM(self) -> Pseudobulk:
        """
        Calculate counts per million for each cell type.

        Returns:
            A new Pseudobulk dataset containing the CPMs.
        """
        CPMs = {}
        for cell_type in self:
            X = self._X[cell_type]
            library_size = X.sum(axis=1) * self._calc_norm_factors(X.T)
            CPMs[cell_type] = X / library_size[:, None] * 1e6
        return Pseudobulk(X=CPMs, obs=self._obs, var=self._var)
    
    def log_CPM(self,
                *,
                prior_count: int | float | np.integer |
                             np.floating = 2) -> Pseudobulk:
        """
        Calculate log counts per million for each cell type.
        
        Do NOT run this before DE(), since DE() already runs it internally.
        
        Based on the R translation of edgeR's C++ cpm() code at
        bioinformatics.stackexchange.com/a/4990.
        
        Results were verified to match edgeR to within floating-point error.
        
        Args:
            prior_count: the pseudocount to add before log-transforming. In the
                         current version of edgeR, prior.count is now 2 instead
                         of the old value of 0.5: code.bioconductor.org/browse/
                         edgeR/blob/RELEASE_3_18/R/cpm.R
        
        Returns:
            A new Pseudobulk dataset containing the log(CPMs).
        """
        check_type(prior_count, 'prior_count', (int, float),
                   'a positive number')
        check_bounds(prior_count, 'prior_count', 0, left_open=True)
        log_CPMs = {}
        for cell_type in self:
            X = self._X[cell_type]
            library_size = X.sum(axis=1) * self._calc_norm_factors(X.T)
            pseudocount = prior_count * library_size / library_size.mean()
            library_size += 2 * pseudocount
            log_CPMs[cell_type] = np.log2(X + pseudocount[:, None]) - \
                np.log2(library_size[:, None]) + np.log2(1e6)
        return Pseudobulk(X=log_CPMs, obs=self._obs, var=self._var)

    def regress_out_obs(self,
                        covariate_columns: Sequence[
                            PseudobulkColumn |
                            Sequence[PseudobulkColumn | None]],
                        *,
                        error_if_int: bool = True) -> Pseudobulk:
        """
        Regress out covariates from obs. Must be run after log_CPM().

        Args:
            covariate_columns: a sequence of columns of obs to regress out.
                               Each element of the sequence can be a column
                               name, a polars expression, a polars Series, a
                               1D NumPy array, or a function that takes in this
                               Pseudobulk dataset and a cell type and returns a
                               polars Series or 1D NumPy array. Each element of
                               the sequence may also itself be a sequence of
                               any combination of these for each cell type, or
                               None to not include that covariate for that cell
                               type.
            error_if_int: if True, raise an error if `X.dtype` is integer
                          (indicating the user may not have run log_CPM() yet)

        Returns:
            A new Pseudobulk dataset with covariates regressed out.
        """
        # Check inputs
        if covariate_columns is None:
            error_message = 'covariate_columns is None'
            raise TypeError(error_message)
        for index, column in enumerate(covariate_columns):
            if column is None:
                error_message = f'covariate_columns[{index}] is None'
                raise TypeError(error_message)
        covariate_columns = [
            self._get_columns('obs', column, f'covariate_columns[{index}]',
                              ('integer', 'floating-point', pl.Categorical,
                               pl.Enum))
            for index, column in enumerate(covariate_columns)]
        check_type(error_if_int, 'error_if_int', bool, 'Boolean')
        # For each cell type...
        residuals = {}
        for cell_type, (X, obs, var) in self.items():
            # If error_if_int=True, raise an error if X has an integer dtype
            if error_if_int and np.issubdtype(X.dtype, np.integer):
                error_message = (
                    f'X[{cell_type!r}].dtype is {X.dtype!r}, an integer data '
                    f'type; did you forget to run log_CPM() before '
                    f'regress_out()?')
                raise ValueError(error_message)
            # Get the covariates for this cell type
            covariates = pl.DataFrame([column[cell_type]
                                       for column in covariate_columns])
            # Convert the covariates to a Numpy array, and add an intercept
            covariates = covariates.to_numpy()
            if not np.issubdtype(covariates.dtype, np.number):
                error_message = (
                    f'obs[{cell_type!r}].select(covariate_columns) must be '
                    f'convertible to a numeric NumPy array, but converted to '
                    f'an array of data type {str(covariates.dtype)!r}')
                raise TypeError(error_message)
            covariates = np.column_stack(
                (np.ones(len(covariates), covariates.dtype), covariates))
            # Regress out the covariates; silence warnings with rcond=None
            beta = np.linalg.lstsq(covariates, X, rcond=None)[0]
            # Calculate the residuals
            residuals[cell_type] = X - covariates @ beta
        # Return a new Pseudobulk datasets with the residuals
        return Pseudobulk(X=residuals, obs=self._obs, var=self._var)
    
    def regress_out_var(self,
                        covariate_columns: Sequence[
                            PseudobulkColumn |
                            Sequence[PseudobulkColumn | None]],
                        *,
                        error_if_int: bool = True) -> Pseudobulk:
        """
        Regress out covariates from var. Must be run after log_CPM().

        Args:
            covariate_columns: a sequence of columns of var to regress out.
                               Each element of the sequence can be a column
                               name, a polars expression, a polars Series, a
                               1D NumPy array, or a function that takes in this
                               Pseudobulk dataset and a cell type and returns a
                               polars Series or 1D NumPy array. Each element of
                               the sequence may also itself be a sequence of
                               any combination of these for each cell type, or
                               None to not include that covariate for that cell
                               type.
            error_if_int: if True, raise an error if `X.dtype` is integer
                          (indicating the user may not have run log_CPM() yet)

        Returns:
            A new Pseudobulk dataset with covariates regressed out.
        """
        # Check inputs
        if covariate_columns is None:
            error_message = 'covariate_columns is None'
            raise TypeError(error_message)
        for index, column in enumerate(covariate_columns):
            if column is None:
                error_message = f'covariate_columns[{index}] is None'
                raise TypeError(error_message)
        covariate_columns = [
            self._get_columns('var', column, f'covariate_columns[{index}]',
                              ('integer', 'floating-point', pl.Categorical,
                               pl.Enum))
            for index, column in enumerate(covariate_columns)]
        check_type(error_if_int, 'error_if_int', bool, 'Boolean')
        # For each cell type...
        residuals = {}
        for cell_type, (X, obs, var) in self.items():
            # If error_if_int=True, raise an error if X has an integer dtype
            if error_if_int and np.issubdtype(X.dtype, np.integer):
                error_message = (
                    f'X[{cell_type!r}].dtype is {X.dtype!r}, an integer data '
                    f'type; did you forget to run log_CPM() before '
                    f'regress_out()?')
                raise ValueError(error_message)
            # Get the covariates for this cell type
            covariates = pl.DataFrame([column[cell_type]
                                       for column in covariate_columns])
            # Convert the covariates to a Numpy array, and add an intercept
            covariates = covariates.to_numpy()
            if not np.issubdtype(covariates.dtype, np.number):
                error_message = (
                    f'var[{cell_type!r}].select(covariate_columns) must be '
                    f'convertible to a numeric NumPy array, but converted to '
                    f'an array of data type {str(covariates.dtype)!r}')
                raise TypeError(error_message)
            covariates = np.column_stack(
                (np.ones(len(covariates), covariates.dtype), covariates))
            # Regress out the covariates; silence warnings with rcond=None
            beta = np.linalg.lstsq(covariates, X.T, rcond=None)[0]
            # Calculate the residuals
            residuals[cell_type] = (X.T - covariates @ beta).T
        # Return a new Pseudobulk datasets with the residuals
        return Pseudobulk(X=residuals, obs=self._obs, var=self._var)
    
    # A slightly reformatted version of the voomByGroup source code from
    # github.com/YOU-k/voomByGroup/blob/main/voomByGroup.R, which is available
    # under the MIT license. Copyright (c) 2023 Yue You.
    _voomByGroup_source_code = r'''
    voomByGroup <- function (counts, group = NULL, design = NULL,
                             lib.size = NULL, dynamic = NULL,
                             normalize.method = "none", span = 0.5,
                             save.plot = FALSE, print = TRUE, plot = c("none",
                             "all", "separate", "combine"),
                             col.lines = NULL, pos.legend = c("inside",
                             "outside", "none"), fix.y.axis = FALSE, ...) {
      out <- list()
      if (is(counts, "DGEList")) {
        out$genes <- counts$genes
        out$targets <- counts$samples
        if(is.null(group))
          group <- counts$samples$group
        if (is.null(lib.size))
          lib.size <- with(counts$samples, lib.size * norm.factors)
        counts <- counts$counts
      }
      else {
        isExpressionSet <-
          suppressPackageStartupMessages(is(counts, "ExpressionSet"))
        if (isExpressionSet) {
          if (length(Biobase::fData(counts)))
            out$genes <- Biobase::fData(counts)
          if (length(Biobase::pData(counts)))
            out$targets <- Biobase::pData(counts)
          counts <- Biobase::exprs(counts)
        }
        else {
          counts <- as.matrix(counts)
        }
      }
      if (nrow(counts) < 2L)
        stop("Need at least two genes to fit a mean-variance trend")
      # Library size
      if(is.null(lib.size))
        lib.size <- colSums(counts)
      # Group
      if(is.null(group))
        group <- rep("Group1", ncol(counts))
      group <- as.factor(group)
      intgroup <- as.integer(group)
      levgroup <- levels(group)
      ngroups <- length(levgroup)
      # Design matrix
      if (is.null(design)) {
        design <- matrix(1L, ncol(counts), 1)
        rownames(design) <- colnames(counts)
        colnames(design) <- "GrandMean"
      }
      # Dynamic
      if (is.null(dynamic)) {
        dynamic <- rep(FALSE, ngroups)
      }
      # voom by group
      if(print)
        cat("Group:\n")
      E <- w <- counts
      xy <- line <- as.list(rep(NA, ngroups))
      names(xy) <- names(line) <- levgroup
      for (lev in 1L:ngroups) {
        if(print)
          cat(lev, levgroup[lev], "\n")
        i <- intgroup == lev
        countsi <- counts[, i]
        libsizei <- lib.size[i]
        designi <- design[i, , drop = FALSE]
        QR <- qr(designi)
        if(QR$rank<ncol(designi))
          designi <- designi[,QR$pivot[1L:QR$rank], drop = FALSE]
        if(ncol(designi)==ncol(countsi))
          designi <- matrix(1L, ncol(countsi), 1)
        voomi <- voom(counts = countsi, design = designi, lib.size = libsizei,
                      normalize.method = normalize.method, span = span,
                      plot = FALSE, save.plot = TRUE, ...)
        E[, i] <- voomi$E
        w[, i] <- voomi$weights
        xy[[lev]] <- voomi$voom.xy
        line[[lev]] <- voomi$voom.line
      }
      #voom overall
      if (TRUE %in% dynamic){
        voom_all <- voom(counts = counts, design = design, lib.size = lib.size,
                         normalize.method = normalize.method, span = span,
                         plot = FALSE, save.plot = TRUE, ...)
        E_all <- voom_all$E
        w_all <- voom_all$weights
        xy_all <- voom_all$voom.xy
        line_all <- voom_all$voom.line
        dge <- DGEList(counts)
        disp <- estimateCommonDisp(dge)
        disp_all <- disp$common
      }
      # Plot, can be "both", "none", "separate", or "combine"
      plot <- plot[1]
      if(plot!="none"){
        disp.group <- c()
        for (lev in levgroup) {
          dge.sub <- DGEList(counts[,group == lev])
          disp <- estimateCommonDisp(dge.sub)
          disp.group[lev] <- disp$common
        }
        if(plot %in% c("all", "separate")){
          if (fix.y.axis == TRUE) {
            yrange <- sapply(levgroup, function(lev){
              c(min(xy[[lev]]$y), max(xy[[lev]]$y))
            }, simplify = TRUE)
            yrange <- c(min(yrange[1,]) - 0.1, max(yrange[2,]) + 0.1)
          }
          for (lev in 1L:ngroups) {
            if (fix.y.axis == TRUE){
              plot(xy[[lev]], xlab = "log2( count size + 0.5 )",
                   ylab = "Sqrt( standard deviation )", pch = 16, cex = 0.25,
                   ylim = yrange)
            } else {
              plot(xy[[lev]], xlab = "log2( count size + 0.5 )",
                   ylab = "Sqrt( standard deviation )", pch = 16, cex = 0.25)
            }
            title(paste("voom: Mean-variance trend,", levgroup[lev]))
            lines(line[[lev]], col = "red")
            legend("topleft", bty="n", paste("BCV:",
              round(sqrt(disp.group[lev]), 3)), text.col="red")
          }
        }
        
        if(plot %in% c("all", "combine")){
          if(is.null(col.lines))
            col.lines <- 1L:ngroups
          if(length(col.lines)<ngroups)
            col.lines <- rep(col.lines, ngroups)
          xrange <- unlist(lapply(line, `[[`, "x"))
          xrange <- c(min(xrange)-0.3, max(xrange)+0.3)
          yrange <- unlist(lapply(line, `[[`, "y"))
          yrange <- c(min(yrange)-0.1, max(yrange)+0.3)
          plot(1L,1L, type="n", ylim=yrange, xlim=xrange,
               xlab = "log2( count size + 0.5 )",
               ylab = "Sqrt( standard deviation )")
          title("voom: Mean-variance trend")
          if (TRUE %in% dynamic){
            for (dy in which(dynamic)){
              line[[dy]] <- line_all
              disp.group[dy] <- disp_all
              levgroup[dy] <- paste0(levgroup[dy]," (all)")
            }
          }
          for (lev in 1L:ngroups)
            lines(line[[lev]], col=col.lines[lev], lwd=2)
          pos.legend <- pos.legend[1]
          disp.order <- order(disp.group, decreasing = TRUE)
          text.legend <-
            paste(levgroup, ", BCV: ", round(sqrt(disp.group), 3), sep="")
          if(pos.legend %in% c("inside", "outside")){
            if(pos.legend=="outside"){
              plot(1,1, type="n", yaxt="n", xaxt="n", ylab="", xlab="",
                   frame.plot=FALSE)
              legend("topleft", text.col=col.lines[disp.order],
                     text.legend[disp.order], bty="n")
            } else {
              legend("topright", text.col=col.lines[disp.order],
                     text.legend[disp.order], bty="n")
            }
          }
        }
      }
      # Output
      if (TRUE %in% dynamic){
        E[,intgroup %in% which(dynamic)] <-
          E_all[,intgroup %in% which(dynamic)]
        w[,intgroup %in% which(dynamic)] <-
          w_all[,intgroup %in% which(dynamic)]
      }
      out$E <- E
      out$weights <- w
      out$design <- design
      if(save.plot){
        out$voom.line <- line
        out$voom.xy <- xy
      }
      new("EList", out)
    }
    '''
    
    def DE(self,
           label_column: PseudobulkColumn | Sequence[PseudobulkColumn],
           covariate_columns: Sequence[PseudobulkColumn | None |
                                       Sequence[PseudobulkColumn | None]] |
                              None,
           *,
           case_control: bool = True,
           cell_types: str | Iterable[str] | None = None,
           excluded_cell_types: str | Iterable[str] | None = None,
           min_samples: int | np.integer = 2,
           include_library_size_as_covariate: bool = True,
           include_num_cells_as_covariate: bool = True,
           return_voom_info: bool = True,
           allow_float: bool = False,
           verbose: bool = True) -> DE:
        """
        Perform differential expression (DE) on a Pseudobulk dataset with
        limma-voom. Uses voomByGroup when case_control=True, which is better
        than regular voom for case-control DE.
        
        Loosely based on the `de_pseudobulk()` function from
        github.com/tluquez/utils/blob/main/utils.R, which is itself based on
        github.com/neurorestore/Libra/blob/main/R/pseudobulk_de.R.

        Args:
            label_column: the column of obs to calculate DE with respect to.
                          Can be a column name, a polars expression, a polars
                          Series, a 1D NumPy array, or a function that takes in
                          this Pseudobulk dataset and a cell type and returns a
                          polars Series or 1D NumPy array. Can also be a
                          sequence of any combination of these for each cell
                          type. If `case_control=True`, must be Boolean,
                          integer, floating-point, or Enum with cases = 1/True
                          and controls = 0/False. If `case_control=False`, must
                          be integer or floating-point.
            covariate_columns: an optional sequence of columns of obs to use
                               as covariates, or None to not include
                               covariates. Each element of the sequence can be
                               a column name, a polars expression, a polars
                               Series, a 1D NumPy array, or a function that
                               takes in this Pseudobulk dataset and a cell type
                               and returns a polars Series or 1D NumPy array.
                               Each element of the sequence may also itself be
                               a sequence of any combination of these for each
                               cell type, or None to not include that covariate
                               for that cell type.
            case_control: whether the analysis is case-control or with respect
                          to a quantitative variable.
                          If True, uses voomByGroup instead of regular voom,
                          and uses `label_column` as the `group` argument to
                          calcNormFactors().
            cell_types: one or more cell types to test for differential
                        expression; if None, test all cell types. Mutually
                        exclusive with `excluded_cell_types`.
            excluded_cell_types: cell types to exclude when testing for
                                 differential expression; mutually exclusive
                                 with `cell_types`
            min_samples: only compute DE for cell types with at least this many
                         cases and this many controls, or with at least this
                         many total samples if `case_control=False`
            include_library_size_as_covariate: whether to include the log2 of
                                               the library size, calculated
                                               according to the method of
                                               edgeR's calcNormFactors(), as an
                                               additional covariate
            include_num_cells_as_covariate: whether to include the log2 of the
                                            `'num_cells'` column of obs, i.e.
                                            the number of cells that went into
                                            each sample's pseudobulk in each
                                            cell type, as an additional
                                            covariate
            return_voom_info: whether to include the voom weights and voom plot
                              data in the returned DE object; set to False for
                              reduced runtime if you do not need to use the
                              voom weights or generate voom plots
            allow_float: if False, raise an error if `X.dtype` is
                         floating-point (suggesting the user may not be using
                         the raw counts, e.g. due to accidentally having run
                         log_CPM() already); if True, disable this sanity check
            verbose: whether to print out details of the DE estimation

        Returns:
            A DE object with a `table` attribute containing a polars DataFrame
            of the DE results. If `return_voom_info=True`, also includes a
            `voom_weights` attribute containing a {cell_type: DataFrame}
            dictionary of voom weights, and a `voom_plot_data` attribute
            containing a {cell_type: DataFrame} dictionary of info necessary to
            construct a voom plot with `DE.plot_voom()`.
        """
        # Import required Python and R packages
        from ryp import r, to_py, to_r
        r('suppressPackageStartupMessages(library(edgeR))')
        # Source voomByGroup code
        if case_control:
            r(self._voomByGroup_source_code)
        # Check inputs
        label_column = self._get_columns(
            'obs', label_column, 'label_column',
            (pl.Boolean, 'integer', 'floating-point', pl.Enum)
            if case_control else ('integer', 'floating-point'),
            allow_None=False)
        if covariate_columns is not None:
            covariate_columns = [
                self._get_columns('obs', column, f'covariate_columns[{index}]',
                                  ('integer', 'floating-point', pl.Categorical,
                                   pl.Enum))
                for index, column in enumerate(covariate_columns)]
        check_type(case_control, 'case_control', bool, 'Boolean')
        check_type(min_samples, 'min_samples', int,
                   'an integer greater than or equal to 2')
        check_bounds(min_samples, 'min_samples', 2)
        if cell_types is not None:
            if excluded_cell_types is not None:
                error_message = (
                    'cell_types and excluded_cell_types cannot both be '
                    'specified')
                raise ValueError(error_message)
            is_string = isinstance(cell_types, str)
            cell_types = to_tuple(cell_types)
            if len(cell_types) == 0:
                error_message = 'cell_types is empty'
                raise ValueError(error_message)
            check_types(cell_types, 'cell_types', str, 'strings')
            for cell_type in cell_types:
                if cell_type not in self._X:
                    if is_string:
                        error_message = (
                            f'cell_types is {cell_type!r}, which is not a '
                            f'cell type in this Pseudobulk dataset')
                        raise ValueError(error_message)
                    else:
                        error_message = (
                            f'one of the elements of cell_types, '
                            f'{cell_type!r}, is not a cell type in this '
                            f'Pseudobulk dataset')
                        raise ValueError(error_message)
        elif excluded_cell_types is not None:
            excluded_cell_types = to_tuple(excluded_cell_types)
            check_types(excluded_cell_types, 'cell_types', str, 'strings')
            for cell_type in excluded_cell_types:
                if cell_type not in self._X:
                    if excluded_cell_types:
                        error_message = (
                            f'excluded_cell_types is {cell_type!r}, which is '
                            f'not a cell type in this Pseudobulk dataset')
                        raise ValueError(error_message)
                    else:
                        error_message = (
                            f'one of the elements of excluded_cell_types, '
                            f'{cell_type!r}, is not a cell type in this '
                            f'Pseudobulk dataset')
                        raise ValueError(error_message)
            cell_types = [cell_type for cell_type in self._X
                          if cell_type not in excluded_cell_types]
            if len(cell_types) == 0:
                error_message = \
                    'all cell types were excluded by excluded_cell_types'
                raise ValueError(error_message)
        else:
            cell_types = self._X
        check_type(include_library_size_as_covariate,
                   'include_library_size_as_covariate', bool, 'Boolean')
        check_type(include_num_cells_as_covariate,
                   'include_num_cells_as_covariate', bool, 'Boolean')
        if include_num_cells_as_covariate:
            for cell_type in self:
                if 'num_cells' not in self._obs[cell_type]:
                    error_message = (
                        f"include_num_cells_as_covariate is True, but "
                        f"'num_cells' is not a column of obs[{cell_type!r}]")
                    raise ValueError(error_message)
        check_type(allow_float, 'allow_float', bool, 'Boolean')
        check_type(verbose, 'verbose', bool, 'Boolean')
        # Compute DE for each cell type
        DE_results = {}
        if return_voom_info:
            voom_weights = {}
            voom_plot_data = {}
        for cell_type in cell_types:
            X = self._X[cell_type]
            obs = self._obs[cell_type]
            var = self._var[cell_type]
            # If `case_control=False`, skip cell types with fewer than
            # `min_samples` total samples
            num_samples = len(obs)
            if not case_control and num_samples < min_samples:
                if verbose:
                    print(f'[{cell_type}] Skipping this cell type because it '
                          f'has only {num_samples:,} '
                          f'{plural("sample", num_samples)}, which is fewer '
                          f'than min_samples ({min_samples:,})')
                continue
            with Timer(f'[{cell_type}] Calculating DE', verbose=verbose):
                # If `allow_float=False`, raise an error if `X` is
                # floating-point
                if not allow_float and np.issubdtype(X.dtype, np.floating):
                    error_message = (
                        f"DE() requires raw counts but X[{cell_type!r}].dtype "
                        f"is {X.dtype!r}, a floating-point data type. If you "
                        f"are sure that all values are integers, i.e. that "
                        f"X[{cell_type!r}].data == X[{cell_type!r}].data"
                        f".astype(int)).all(), then set allow_float=True (or "
                        f"just cast X to an integer data type). "
                        f"Alternatively, did you accidentally run log_CPM() "
                        f"before DE()?")
                    raise TypeError(error_message)
                # Get the DE labels and covariates for this cell type
                DE_labels = label_column[cell_type]
                covariates = pl.DataFrame([column[cell_type]
                                           for column in covariate_columns]) \
                    if covariate_columns is not None else pl.DataFrame()
                if include_num_cells_as_covariate:
                    covariates = \
                        covariates.with_columns(obs['num_cells'].log(2))
                if case_control:
                    # If `case_control=True`, check that the DE labels have
                    # only two unique values
                    if DE_labels.dtype != pl.Boolean:
                        if DE_labels.dtype == pl.Enum:
                            categories = DE_labels.cat.get_categories()
                            if len(categories) != 2:
                                suffix = 'y' if len(categories) == 1 else 'ies'
                                error_message = (
                                    f'[{cell_type}] label_column is an Enum '
                                    f'column with {len(categories):,} '
                                    f'categor{suffix}, but must have 2 (cases '
                                    f'and controls)')
                                raise ValueError(error_message)
                            DE_labels = DE_labels.to_physical()
                        else:
                            unique_labels = DE_labels.unique()
                            num_unique_labels = len(unique_labels)
                            if num_unique_labels != 2:
                                plural_string = \
                                    plural('unique value', num_unique_labels)
                                error_message = (
                                    f'[{cell_type}] label_column is a numeric '
                                    f'column with {num_unique_labels:,} '
                                    f'{plural_string}, but must have 2 '
                                    f'(cases = 1, controls = 0) unless '
                                    f'case_control=False')
                                raise ValueError(error_message)
                            if not unique_labels.sort()\
                                    .equals(pl.Series([0, 1])):
                                error_message = (
                                    f'[{cell_type}] label_column is a numeric '
                                    f'column with 2 unique values, '
                                    f'{unique_labels[0]} and '
                                    f'{unique_labels[1]}, but must have '
                                    f'cases = 1 and controls = 0')
                                raise ValueError(error_message)
                    # If `case_control=True`, skip cell types with fewer than
                    # `min_samples` cases or `min_samples` controls
                    num_cases = DE_labels.sum()
                    if num_cases < min_samples:
                        if verbose:
                            print(f'[{cell_type}] Skipping this cell type '
                                  f'because it has only {num_cases:,} '
                                  f'{plural("case", num_cases)}, which is '
                                  f'fewer than min_samples ({min_samples:,})')
                        continue
                    num_controls = len(DE_labels) - num_cases
                    if num_controls < min_samples:
                        if verbose:
                            print(f'[{cell_type}] Skipping this cell type '
                                  f'because it has only {num_controls:,} '
                                  f'{plural("control", num_controls)}, which '
                                  f'is fewer than min_samples '
                                  f'{min_samples:,})')
                        continue
                # Get the design matrix
                if verbose:
                    print(f'[{cell_type}] Generating design matrix...')
                design_matrix = \
                    obs.select(pl.lit(1).alias('intercept'), DE_labels)
                if covariates.width:
                    design_matrix = pl.concat([
                        design_matrix,
                        covariates.to_dummies(covariates.select(
                            pl.col(pl.Categorical, pl.Enum)).columns,
                            drop_first=True)],
                        how='horizontal')
                # Estimate library sizes
                if verbose:
                    print(f'[{cell_type}] Estimating library sizes...')
                library_size = X.sum(axis=1) * self._calc_norm_factors(X.T)
                if library_size.min() == 0:
                    error_message = f'[{cell_type}] some library sizes are 0'
                    raise ValueError(error_message)
                if include_library_size_as_covariate:
                    design_matrix = design_matrix\
                        .with_columns(library_size=np.log2(library_size))
                # Check that the design matrix has more rows than columns, and
                # is full-rank
                if verbose:
                    print(f'[{cell_type}] Sanity-checking the design matrix')
                if design_matrix.width >= design_matrix.height:
                    error_message = (
                        f'[{cell_type}] the design matrix must have more rows '
                        f'(samples) than columns (one plus the number of '
                        f'covariates), but has {design_matrix.width:,} rows '
                        f'and {design_matrix.height:,} columns; either reduce '
                        f'the number of covariates or exclude this cell type '
                        f'with e.g. excluded_cell_types={cell_type!r}')
                    raise ValueError(error_message)
                if np.linalg.matrix_rank(design_matrix.to_numpy()) < \
                        design_matrix.width:
                    error_message = (
                        f'[{cell_type}] the design matrix is not full-rank; '
                        f'some of your covariates are linear combinations of '
                        f'other covariates')
                    raise ValueError(error_message)
                try:
                    # Convert the expression matrix, design matrix, and library
                    # sizes to R
                    if verbose:
                        print(f'[{cell_type}] Converting the expression '
                              f'matrix, design matrix and library sizes to '
                              f'R...')
                    to_r(X.T, '.Pseudobulk.X.T', rownames=var[:, 0],
                         colnames=obs['ID'])
                    to_r(design_matrix, '.Pseudobulk.design.matrix',
                         rownames=obs['ID'])
                    to_r(library_size, '.Pseudobulk.library.size',
                         rownames=obs['ID'])
                    # Run voom
                    to_r(return_voom_info, 'save.plot')
                    if case_control:
                        if verbose:
                            print(f'[{cell_type}] Running voomByGroup...')
                        to_r(DE_labels, '.Pseudobulk.DE.labels',
                             rownames=obs['ID'])
                        r('.Pseudobulk.voom.result = voomByGroup('
                          '.Pseudobulk.X.T, .Pseudobulk.DE.labels, '
                          '.Pseudobulk.design.matrix, '
                          '.Pseudobulk.library.size, save.plot=save.plot, '
                          'print=FALSE)')
                    else:
                        if verbose:
                            print(f'[{cell_type}] Running voom...')
                        r('.Pseudobulk.voom.result = voom(.Pseudobulk.X.T, '
                          '.Pseudobulk.design.matrix, '
                          '.Pseudobulk.library.size, save.plot=save.plot)')
                    if return_voom_info:
                        # noinspection PyUnboundLocalVariable
                        voom_weights[cell_type] = \
                            to_py('.Pseudobulk.voom.result$weights',
                                  index='gene')
                        # noinspection PyUnboundLocalVariable
                        voom_plot_data[cell_type] = pl.DataFrame({
                            f'{prop}_{dim}_{case}': to_py(
                                f'.Pseudobulk.voom.result$voom.{prop}$'
                                f'`{case_label}`${dim}', format='numpy')
                            for prop in ('xy', 'line') for dim in ('x', 'y')
                            for case, case_label in zip(
                                (False, True), ('FALSE', 'TRUE')
                                if DE_labels.dtype == pl.Boolean else (0, 1))}
                            if case_control else {
                            f'{prop}_{dim}': to_py(
                                f'.Pseudobulk.voom.result$voom.{prop}${dim}',
                                format='numpy')
                            for prop in ('xy', 'line') for dim in ('x', 'y')})
                    # Run limma
                    if verbose:
                        print(f'[{cell_type}] Running lmFit...')
                    r('.Pseudobulk.lmFit.result = lmFit('
                      '.Pseudobulk.voom.result, .Pseudobulk.design.matrix)')
                    if verbose:
                        print(f'[{cell_type}] Running eBayes...')
                    r('.Pseudobulk.eBayes.result = eBayes('
                      '.Pseudobulk.lmFit.result, trend=FALSE, robust=FALSE)')
                    # Get results table
                    if verbose:
                        print(f'[{cell_type}] Running topTable...')
                    to_r(DE_labels.name, '.Pseudobulk.coef')
                    r('.Pseudobulk.topTable.result = topTable('
                      '.Pseudobulk.eBayes.result, coef=.Pseudobulk.coef, '
                      'number=Inf, adjust.method="none", sort.by="P", '
                      'confint=TRUE)')
                    if verbose:
                        print(f'[{cell_type}] Collating results...')
                    DE_results[cell_type] = \
                        to_py('.Pseudobulk.topTable.result', index='gene')\
                        .select('gene',
                                logFC=pl.col.logFC,
                                SE=to_py('.Pseudobulk.eBayes.result$s2.post')
                                   .sqrt() *
                                   to_py('.Pseudobulk.eBayes.result$stdev.'
                                         'unscaled[,1]', index=False),
                                LCI=pl.col('CI.L'),
                                UCI=pl.col('CI.R'),
                                AveExpr=pl.col.AveExpr,
                                P=pl.col('P.Value'),
                                Bonferroni=bonferroni(pl.col('P.Value')),
                                FDR=fdr(pl.col('P.Value')))
                finally:
                    r('rm(list = Filter(exists, c(".Pseudobulk.X.T", '
                      '".Pseudobulk.DE.labels", ".Pseudobulk.design.matrix", '
                      '".Pseudobulk.library.size", ".Pseudobulk.voom.result", '
                      '".Pseudobulk.lmFit.result", '
                      '".Pseudobulk.eBayes.result", ".Pseudobulk.coef", '
                      '".Pseudobulk.topTable.result")))')
        # Concatenate across cell types
        table = pl.concat([
            cell_type_DE_results
            .select(pl.lit(cell_type).alias('cell_type'), pl.all())
            for cell_type, cell_type_DE_results in DE_results.items()])
        if return_voom_info:
            return DE(table, case_control, voom_weights, voom_plot_data)
        else:
            return DE(table, case_control)
    

class DE:
    """
    Differential expression results returned by Pseudobulk.DE().
    """
    
    def __init__(self,
                 table: pl.DataFrame,
                 case_control: bool | None = None,
                 voom_weights: dict[str, pl.DataFrame] | None = None,
                 voom_plot_data: dict[str, pl.DataFrame] | None = None) -> \
            None:
        """
        Initialize the DE object.
        
        Args:
            table: a polars DataFrame containing the DE results, with columns:
                   - cell_type: the cell type in which DE was tested
                   - gene: the gene for which DE was tested
                   - logFC: the log fold change of the gene, i.e. its effect
                            size
                   - SE: the standard error of the effect size
                   - LCI: the lower 95% confidence interval of the effect size
                   - UCI: the upper 95% confidence interval of the effect size
                   - AveExpr: the gene's average expression in this cell type,
                              in log CPM
                   - P: the DE p-value
                   - Bonferroni: the Bonferroni-corrected DE p-value
                   - FDR: the FDR q-value for the DE
                   Or, a directory containing a DE object saved with `save()`.
            case_control: whether the analysis is case-control or with respect
                          to a quantitative variable. Must be specified unless
                          `table` is a directory.
            voom_weights: an optional {cell_type: DataFrame} dictionary of voom
                         weights, where rows are genes and columns are samples.
                         The first column of each cell type's DataFrame,
                         'gene', contains the gene names.
            voom_plot_data: an optional {cell_type: DataFrame} dictionary of
                            info necessary to construct a voom plot with
                            `DE.plot_voom()`
        """
        if isinstance(table, pl.DataFrame):
            check_type(case_control, 'case_control', bool, 'Boolean')
            if voom_weights is not None:
                if voom_plot_data is None:
                    error_message = (
                        'voom_plot_data must be specified when voom_weights '
                        'is specified')
                    raise ValueError(error_message)
                check_type(voom_weights, 'voom_weights', dict, 'a dictionary')
                if voom_weights.keys() != voom_plot_data.keys():
                    error_message = (
                        'voom_weights and voom_plot_data must have matching '
                        'keys (cell types)')
                    raise ValueError(error_message)
                for key in voom_weights:
                    if not isinstance(key, str):
                        error_message = (
                            f'all keys of voom_weights and voom_plot_data '
                            f'must be strings (cell types), but they contain '
                            f'a key of type {type(key).__name__!r}')
                        raise TypeError(error_message)
            if voom_plot_data is not None:
                if voom_weights is None:
                    error_message = (
                        'voom_weights must be specified when voom_plot_data '
                        'is specified')
                    raise ValueError(error_message)
                check_type(voom_plot_data, 'voom_plot_data', dict,
                           'a dictionary')
        elif isinstance(table, (str, Path)):
            table = str(table)
            if not os.path.exists(table):
                error_message = f'DE object directory {table!r} does not exist'
                raise FileNotFoundError(error_message)
            cell_types = [line.rstrip('\n') for line in
                          open(f'{table}/cell_types.txt')]
            voom_weights = {cell_type: pl.read_parquet(
                os.path.join(table, f'{cell_type.replace("/", "-")}.'
                                    f'voom_weights.parquet'))
                for cell_type in cell_types}
            voom_plot_data = {cell_type: pl.read_parquet(
                os.path.join(table, f'{cell_type.replace("/", "-")}.'
                                    f'voom_plot_data.parquet'))
                for cell_type in cell_types}
            # noinspection PyUnresolvedReferences
            case_control = \
                next(iter(voom_plot_data.values())).columns[0].count('_') == 2
            table = pl.read_parquet(os.path.join(table, 'table.parquet'))
        else:
            error_message = (
                f'table must be a polars DataFrame or a directory (string or '
                f'pathlib.Path) containing a saved DE object, but has type '
                f'{type(table).__name__!r}')
            raise TypeError(error_message)
        self.table = table
        self.case_control = case_control
        self.voom_weights = voom_weights
        self.voom_plot_data = voom_plot_data
    
    def __repr__(self) -> str:
        """
        Get a string representation of this DE object.
        
        Returns:
            A string summarizing the object.
        """
        num_cell_types = self.table['cell_type'].n_unique()
        descr = (
            f'DE object with {len(self.table):,} '
            f'{"entries" if len(self.table) != 1 else "entry"} across '
            f'{num_cell_types:,} {plural("cell type", num_cell_types)}')
        return descr
    
    def __eq__(self, other: DE) -> bool:
        """
        Test for equality with another DE object.
        
        Args:
            other: the other DE object to test for equality with

        Returns:
            Whether the two DE objects are identical.
        """
        if not isinstance(other, DE):
            error_message = (
                f'the left-hand operand of `==` is a DE object, but '
                f'the right-hand operand has type {type(other).__name__!r}')
            raise TypeError(error_message)
        return self.table.equals(other.table) and \
            self.case_control == other.case_control and \
            (other.voom_weights is None if self.voom_weights is None else
             self.voom_weights.keys() == other.voom_weights.keys() and
             all(self.voom_weights[cell_type].equals(
                     other.voom_weights[cell_type]) and
                 self.voom_plot_data[cell_type].equals(
                     other.voom_plot_data[cell_type])
                 for cell_type in self.voom_weights))
    
    def save(self, directory: str | Path, overwrite: bool = False) -> None:
        """
        Save a DE object to `directory` (which must not exist unless
        `overwrite=True`, and will be created) with the table at table.parquet.
        
        If the DE object contains voom info (i.e. was created with
        `return_voom_info=True` in `Pseudobulk.DE()`, the default), also saves
        each cell type's voom weights and voom plot data to
        f'{cell_type}_voom_weights.parquet' and
        f'{cell_type}_voom_plot_data.parquet', as well as a text file,
        cell_types.txt, containing the cell types.
        
        Args:
            directory: the directory to save the DE object to
            overwrite: if False, raises an error if the directory exists; if
                       True, overwrites files inside it as necessary
        """
        check_type(directory, 'directory', (str, Path),
                   'a string or pathlib.Path')
        directory = str(directory)
        if not overwrite and os.path.exists(directory):
            error_message = (
                f'directory {directory!r} already exists; set overwrite=True '
                f'to overwrite')
            raise FileExistsError(error_message)
        os.makedirs(directory, exist_ok=overwrite)
        self.table.write_parquet(os.path.join(directory, 'table.parquet'))
        if self.voom_weights is not None:
            with open(os.path.join(directory, 'cell_types.txt'), 'w') as f:
                print('\n'.join(self.voom_weights), file=f)
            for cell_type in self.voom_weights:
                escaped_cell_type = cell_type.replace('/', '-')
                self.voom_weights[cell_type].write_parquet(
                    os.path.join(directory, f'{escaped_cell_type}.'
                                            f'voom_weights.parquet'))
                self.voom_plot_data[cell_type].write_parquet(
                    os.path.join(directory, f'{escaped_cell_type}.'
                                            f'voom_plot_data.parquet'))
    
    def get_hits(self,
                 significance_column: str = 'FDR',
                 threshold: int | float | np.integer | np.floating = 0.05,
                 num_top_hits: int | None = None) -> pl.DataFrame:
        """
        Get all (or the top) differentially expressed genes.
        
        Args:
            significance_column: the name of a Boolean column of self.table to
                                 determine significance from
            threshold: the significance threshold corresponding to
                       significance_column
            num_top_hits: the number of top hits to report for each cell type;
                          if None, report all hits

        Returns:
            The `table` attribute of this DE object, subset to (top) DE hits.
        """
        check_type(significance_column, 'significance_column', str, 'a string')
        if significance_column not in self.table:
            error_message = 'significance_column is not a column of self.table'
            raise ValueError(error_message)
        check_dtype(self.table[significance_column],
                    f'self.table[{significance_column!r}]', 'floating-point')
        check_type(threshold, 'threshold', (int, float),
                   'a number > 0 and ≤ 1')
        check_bounds(threshold, 'threshold', 0, 1, left_open=True)
        if num_top_hits is not None:
            check_type(num_top_hits, 'num_top_hits', int, 'a positive integer')
            check_bounds(num_top_hits, 'num_top_hits', 1)
        return self.table\
            .filter(pl.col(significance_column) < threshold)\
            .pipe(lambda df: df.group_by('cell_type', maintain_order=True)
                  .head(num_top_hits) if num_top_hits is not None else df)
    
    def get_num_hits(self,
                     significance_column: str = 'FDR',
                     threshold: int | float | np.integer |
                                np.floating = 0.05) -> pl.DataFrame:
        """
        Get the number of differentially expressed genes in each cell type.
        
        Args:
            significance_column: the name of a Boolean column of self.table to
                                 determine significance from
            threshold: the significance threshold corresponding to
                       significance_column

        Returns:
            A DataFrame with one row per cell type and two columns:
            'cell_type' and 'num_hits'.
        """
        check_type(significance_column, 'significance_column', str, 'a string')
        if significance_column not in self.table:
            error_message = 'significance_column is not a column of self.table'
            raise ValueError(error_message)
        check_dtype(self.table[significance_column],
                    f'self.table[{significance_column!r}]', 'floating-point')
        check_type(threshold, 'threshold', (int, float),
                   'a number > 0 and ≤ 1')
        check_bounds(threshold, 'threshold', 0, 1, left_open=True)
        return self.table\
            .filter(pl.col(significance_column) < threshold)\
            .group_by('cell_type', maintain_order=True)\
            .agg(num_hits=pl.len())\
            .sort('cell_type')
    
    def plot_voom(self,
                  directory: str | Path,
                  *,
                  point_color: Color = '#666666',
                  case_point_color: Color = '#ff6666',
                  point_size: int | float | np.integer | np.floating = 1,
                  case_point_size: int | float | np.integer | np.floating = 1,
                  line_color: Color = '#000000',
                  case_line_color: Color = '#ff0000',
                  line_width: int | float | np.integer | np.floating = 1.5,
                  case_line_width: int | float | np.integer |
                                   np.floating = 1.5,
                  scatter_kwargs: dict[str, Any] | None = None,
                  case_scatter_kwargs: dict[str, Any] | None = None,
                  plot_kwargs: dict[str, Any] | None = None,
                  case_plot_kwargs: dict[str, Any] | None = None,
                  legend_labels: list[str] |
                                 tuple[str, str] = ('Controls', 'Cases'),
                  legend_kwargs: dict[str, Any] | None = None,
                  xlabel: str = 'Average log2(count + 0.5)',
                  xlabel_kwargs: dict[str, Any] | None = None,
                  ylabel: str = 'sqrt(standard deviation)',
                  ylabel_kwargs: dict[str, Any] | None = None,
                  title: bool | str | dict[str, str] |
                         Callable[[str], str] = False,
                  title_kwargs: dict[str, Any] | None = None,
                  despine: bool = True,
                  overwrite: bool = False,
                  PNG: bool = False,
                  savefig_kwargs: dict[str, Any] | None = None) -> None:
        """
        Generate a voom plot for each cell type that differential expression
        was calculated for.
        
        Voom plots consist of a scatter plot with one point per gene. They
        visualize how the mean expression of each gene across samples (x)
        relates to its variation across samples (y). The plot also includes a
        LOESS (also called LOWESS) fit, a type of non-linear curve fit, of the
        mean-variance (x-y) trend.
        
        Specifically, the x position of a gene's point is the average, across
        samples, of the base-2 logarithm of the gene's count in each sample
        (plus a pseudocount of 0.5): in other words, mean(log2(count + 0.5)).
        The y position is the square root of the standard deviation, across
        samples, of the gene's log counts per million after regressing out,
        across samples, the differential expression design matrix.
        
        For case-control differential expression (`case_control=True` in
        `Pseudobulk.DE()`), voom is run separately for cases and controls
        ("voomByGroup"), and so the voom plots will show a separate LOESS
        trendline for each of the two groups, with the points and trendlines
        for the two groups shown in different colors.
        
        Args:
            directory: the directory to save voom plots to; will be created if
                       it does not exist. Each cell type's voom plot will be
                       saved to f'{cell_type}.pdf' in this directory, or
                       f'{cell_type}.png' if `PNG=True`.
            point_color: the color of the points in the voom plot; if
                         case-control, only points for controls will be plotted
                         in this color
            case_point_color: the color of the points for cases; ignored for
                              non-case-control differential expression
            point_size: the size of the points in the voom plot; if
                        case-control, only the control points will be plotted
                        with this size
            case_point_size: the size of the points for cases; ignored for
                             non-case-control differential expression
            line_color: the color of the LOESS trendline in the voom plot; if
                        case-control, only the control trendline will be
                        plotted in this color
            case_line_color: the color of the LOESS trendline for cases;
                             ignored for non-case-control differential
                             expression
            line_width: the width of the LOESS trendline in the voom plot; if
                        case-control, only the control trendline will be
                        plotted with this width
            case_line_width: the width of the LOESS trendline for cases;
                             ignored for non-case-control differential
                             expression
            scatter_kwargs: a dictionary of keyword arguments to be passed to
                            `ax.scatter()`, such as:
                            - `rasterized`: whether to convert the scatter plot
                              points to a raster (bitmap) image when saving to
                              a vector format like PDF. Defaults to True,
                              instead of the Matplotlib default of False.
                            - `marker`: the shape to use for plotting each cell
                            - `norm`, `vmin`, and `vmax`: control how the
                              numbers in `color_column` are converted to
                              colors, if `color_column` is numeric
                            - `alpha`: the transparency of each point
                            - `linewidths` and `edgecolors`: the width and
                              color of the borders around each marker. These
                              are absent by default (`linewidths=0`), unlike
                              Matplotlib's default. Both arguments can be
                              either single values or sequences.
                            - `zorder`: the order in which the cells are
                              plotted, with higher values appearing on top of
                              lower ones.
                            Specifying `s` or `c`/`color` will raise an error,
                            since these arguments conflict with the
                            `point_size` and `point_color` arguments,
                            respectively.
                            If case-control and `case_scatter_kwargs` is not
                            None, these settings only apply to control points.
            case_scatter_kwargs: a dictionary of keyword arguments to be passed
                                 to `plt.scatter()` for case points. Like for
                                 `scatter_kwargs`, `rasterized=True` is the
                                 default, and specifying `s` or `c`/`color`
                                 will raise an error. If None and
                                 `scatter_kwargs` is not None, the settings in
                                 `scatter_kwargs` apply to all points. Can only
                                 be specified for case-control differential
                                 expression.
            plot_kwargs: a dictionary of keyword arguments to be passed to
                         `plt.plot()` when plotting the trendlines, such as
                         `linestyle` for dashed trendlines. Specifying
                         `color`/`c` or `linewidth` will raise an error, since
                         these arguments conflict with the `line_color` and
                         `line_width` arguments, respectively.
            case_plot_kwargs: a dictionary of keyword arguments to be passed to
                              `plt.plot()` when plotting the case trendlines.
                              Specifying `color`/`c` or `linewidth` will raise
                              an error, like for `plot_kwargs`. If None and
                              `plot_kwargs` is not None, the settings in
                              `plot_kwargs` apply to all points. Can only be
                              specified for case-control differential
                              expression.
            legend_labels: a two-element tuple or list of labels for controls
                           and cases (in that order) in the legend, or None to
                           not include a legend. Ignored for non-case-control
                           differential expression.
            legend_kwargs: a dictionary of keyword arguments to be passed to
                           `plt.legend()` to modify the legend, such as:
                           - `loc`, `bbox_to_anchor`, and `bbox_transform` to
                             set its location.
                           - `prop`, `fontsize`, and `labelcolor` to set its
                             font properties
                           - `facecolor` and `framealpha` to set its background
                             color and transparency
                           - `frameon=True` or `edgecolor` to add or color
                             its border (`frameon` is False by default,
                             unlike Matplotlib's default of True)
                           - `title` to add a legend title
                           Can only be specified for case-control differential
                           expression.
            xlabel: the x-axis label for each voom plot, or None to not include
                    an x-axis label
            xlabel_kwargs: a dictionary of keyword arguments to be passed to
                          `plt.xlabel()` to control the text properties, such
                          as `color` and `size` to modify the text color/size
            ylabel: the y-axis label for each voom plot, or None to not include
                    a y-axis label
            ylabel_kwargs: a dictionary of keyword arguments to be passed to
                          `plt.ylabel()` to control the text properties, such
                          as `color` and `size` to modify the text color/size
            title: what to use as the title. If False, do not include a
                   title. If True, use the cell type as a title. If a string,
                   use the string as the title for every cell type. If a
                   dictionary, use `title[cell_type]` as the title; every cell
                   type must be present in the dictionary. If a function, use
                   `title(cell_type)` as the title.
            title_kwargs: a dictionary of keyword arguments to be passed to
                          `plt.title()` to control the text properties, such
                          as `color` and `size` to modify the text color/size.
                          Cannot be specified when `title=False`.
            despine: whether to remove the top and right spines (borders of the
                     plot area) from the voom plots
            overwrite: if False, raises an error if the directory exists; if
                       True, overwrites files inside it as necessary
            PNG: whether to save the voom plots in PNG instead of PDF format
            savefig_kwargs: a dictionary of keyword arguments to be passed to
                            `plt.savefig()`, such as:
                            - `dpi`: defaults to 300 instead of Matplotlib's
                              default of 150
                            - `bbox_inches`: the bounding box of the portion of
                              the figure to save; defaults to 'tight' (crop out
                              any blank borders) instead of Matplotlib's
                              default of None (save the entire figure)
                            - `pad_inches`: the number of inches of padding to
                              add on each of the four sides of the figure when
                              saving. Defaults to 'layout' (use the padding
                              from the constrained layout engine) instead of
                              Matplotlib's default of 0.1.
                            - `transparent`: whether to save with a transparent
                              background; defaults to True if saving to a PDF
                              (i.e. when `PNG=False`) and False if saving to
                              a PNG, instead of Matplotlib's default of always
                              being False.
        """
        import matplotlib.pyplot as plt
        # Make sure this DE object contains `voom_plot_data`
        if self.voom_plot_data is None:
            error_message = (
                'this DE object does not contain the voom_plot_data '
                'attribute, which is necessary to generate voom plots; re-run '
                'Pseudobulk.DE() with return_voom_info=True to include this '
                'attribute')
            raise AttributeError(error_message)
        # Check that `directory` is a string or pathlib.Path
        check_type(directory, 'voom_plot_directory', (str, Path),
                   'a string or pathlib.Path')
        directory = str(directory)
        # Check that each of the colors are valid Matplotlib colors, and
        # convert them to hex
        for color, color_name in ((point_color, 'point_color'),
                                  (line_color, 'line_color'),
                                  (case_point_color, 'case_point_color'),
                                  (case_line_color, 'case_line_color')):
            if not plt.matplotlib.colors.is_color_like(color):
                error_message = f'{color_name} is not a valid Matplotlib color'
                raise ValueError(error_message)
        point_color = plt.matplotlib.colors.to_hex(point_color)
        line_color = plt.matplotlib.colors.to_hex(line_color)
        case_point_color = plt.matplotlib.colors.to_hex(case_point_color)
        case_line_color = plt.matplotlib.colors.to_hex(case_line_color)
        # Check that point sizes are positive numbers
        check_type(point_size, 'point_size', (int, float), 'a positive number')
        check_bounds(point_size, 'point_size', 0, left_open=True)
        check_type(case_point_size, 'case_point_size', (int, float),
                   'a positive number')
        check_bounds(case_point_size, 'case_point_size', 0, left_open=True)
        # For each of the kwargs arguments, if the argument is not None, check
        # that it is a dictionary and that all its keys are strings.
        for kwargs, kwargs_name in (
                (scatter_kwargs, 'scatter_kwargs'),
                (case_scatter_kwargs, 'case_scatter_kwargs'),
                (plot_kwargs, 'plot_kwargs'),
                (case_plot_kwargs, 'case_plot_kwargs'),
                (legend_kwargs, 'legend_kwargs'),
                (xlabel_kwargs, 'xlabel_kwargs'),
                (ylabel_kwargs, 'ylabel_kwargs'),
                (title_kwargs, 'title_kwargs')):
            if kwargs is not None:
                check_type(kwargs, kwargs_name, dict, 'a dictionary')
                for key in kwargs:
                    if not isinstance(key, str):
                        error_message = (
                            f'all keys of {kwargs_name} must be strings, but '
                            f'it contains a key of type '
                            f'{type(key).__name__!r}')
                        raise TypeError(error_message)
        # Check that `case_scatter_kwargs` and `case_plot_kwargs` are None for
        # non-case-control differential expression. If None, use the settings
        # from `scatter_kwargs` and `plot_kwargs`, respectively. Also set
        # `plot_kwargs` to {} if it is None.
        if case_scatter_kwargs is None:
            case_scatter_kwargs = scatter_kwargs
        elif not self.case_control:
            error_message = (
                'case_scatter_kwargs can only be specified for case-control '
                'differential expression')
            raise ValueError(error_message)
        if plot_kwargs is None:
            plot_kwargs = {}
        if case_plot_kwargs is None:
            case_plot_kwargs = plot_kwargs
        elif not self.case_control:
            error_message = (
                'case_plot_kwargs can only be specified for case-control '
                'differential expression')
            raise ValueError(error_message)
        # Override the defaults for certain keys of `scatter_kwargs` and
        # `case_scatter_kwargs`
        default_scatter_kwargs = dict(rasterized=True, linewidths=0)
        scatter_kwargs = default_scatter_kwargs | scatter_kwargs \
            if scatter_kwargs is not None else default_scatter_kwargs
        case_scatter_kwargs = default_scatter_kwargs | case_scatter_kwargs \
            if case_scatter_kwargs is not None else default_scatter_kwargs
        # Check that `scatter_kwargs` and `case_scatter_kwargs` do not contain
        # the `s` or `c`/`color` keys, and that `plot_kwargs` and
        # `case_plot_kwargs` do not contain the `c`/`color` or `linewidth` keys
        for plot, kwargs_set in enumerate(
                (((scatter_kwargs, 'scatter_kwargs'),
                  (case_scatter_kwargs, 'case_scatter_kwargs')),
                 ((plot_kwargs, 'plot_kwargs'),
                  (case_plot_kwargs, 'case_plot_kwargs')))):
            for kwargs, kwargs_name in kwargs_set:
                if kwargs is None:
                    continue
                for bad_key, alternate_argument in (
                        ('linewidth', 'line_width') if plot else
                        ('s', 'point_size'),
                        ('c', 'point_color' if plot else 'line_color'),
                        ('color', 'point_color' if plot else 'line_color')):
                    if bad_key in kwargs:
                        error_message = (
                            f'{bad_key!r} cannot be specified as a key in '
                            f'{kwargs_name}; specify the {alternate_argument} '
                            f'argument instead')
                        raise ValueError(error_message)
        # Check that `legend_labels` is a two-element tuple or list of strings
        check_type(legend_labels, 'legend_labels', (tuple, list),
                   'a length-2 tuple or list of strings')
        if len(legend_labels) != 2:
            error_message = (
                f'legend_labels must have a length of 2, but has a length of '
                f'{len(legend_labels):,}')
            raise ValueError(error_message)
        check_type(legend_labels[0], 'legend_labels[0]', str, 'a string')
        check_type(legend_labels[1], 'legend_labels[1]', str, 'a string')
        # Override the defaults for certain values of `legend_kwargs`; check
        # that it is None for non-case-control differential expression
        default_legend_kwargs = dict(frameon=False)
        if legend_kwargs is not None:
            if not self.case_control:
                error_message = (
                    'legend_kwargs can only be specified for case-control '
                    'differential expression')
                raise ValueError(error_message)
            legend_kwargs = default_legend_kwargs | legend_kwargs
        else:
            legend_kwargs = default_legend_kwargs
        # Check that `xlabel` and `ylabel` are strings, or None
        if xlabel is not None:
            check_type(xlabel, 'xlabel', str, 'a string')
        if ylabel is not None:
            check_type(ylabel, 'ylabel', str, 'a string')
        # Check that `title` is Boolean, a string, a dictionary where all keys
        # are cell types and every cell type is present, or a function
        check_type(title, 'title', (bool, str, dict, Callable),
                   'Boolean, a string, a dictionary, or a function')
        if isinstance(title, dict):
            if len(title) != len(self.voom_plot_data) or \
                    set(title) != set(self.voom_plot_data):
                error_message = (
                    'when title is a dictionary, all its keys must be cell '
                    'types, and every cell type must be present')
                raise ValueError(error_message)
        # Check that `title_kwargs` is None when `title=False`
        if title is False and title_kwargs is not None:
            error_message = 'title_kwargs cannot be specified when title=False'
            raise ValueError(error_message)
        # Check that `overwrite` and `PNG` are Boolean
        check_type(overwrite, 'overwrite', bool, 'Boolean')
        check_type(PNG, 'PNG', bool, 'Boolean')
        # Override the defaults for certain values of `savefig_kwargs`
        default_savefig_kwargs = \
            dict(dpi=300, bbox_inches='tight', pad_inches='layout',
                 transparent=not PNG)
        savefig_kwargs = default_savefig_kwargs | savefig_kwargs \
            if savefig_kwargs is not None else default_savefig_kwargs
        # Create plot directory
        if not overwrite and os.path.exists(directory):
            error_message = (
                f'directory {directory!r} already exists; set overwrite=True '
                f'to overwrite')
            raise FileExistsError(error_message)
        os.makedirs(directory, exist_ok=overwrite)
        # Save each cell type's voom plot in this directory
        add_legend = self.case_control and legend_labels is not None
        for cell_type, voom_plot_data in self.voom_plot_data.items():
            voom_plot_file = os.path.join(
                directory,
                f'{cell_type.replace("/", "-")}.{"png" if PNG else "pdf"}')
            if self.case_control:
                if add_legend:
                    legend_patches = []
                for case in False, True:
                    plt.scatter(voom_plot_data[f'xy_x_{case}'],
                                voom_plot_data[f'xy_y_{case}'],
                                s=case_point_size if case else point_size,
                                c=case_point_color if case else point_color,
                                **(case_scatter_kwargs if case else
                                   scatter_kwargs))
                    plt.plot(voom_plot_data[f'line_x_{case}'],
                             voom_plot_data[f'line_y_{case}'],
                             c=case_line_color if case else line_color,
                             linewidth=case_line_width if case else line_width,
                             **(case_plot_kwargs if case else plot_kwargs))
                    if add_legend:
                        # noinspection PyUnboundLocalVariable
                        # noinspection PyUnresolvedReferences
                        legend_patches.append(
                            plt.matplotlib.patches.Patch(
                                facecolor=case_point_color if case else
                                point_color,
                                edgecolor=case_line_color if case else
                                line_color,
                                linewidth=case_line_width if case else
                                line_width,
                                label=legend_labels[case]))
                if add_legend:
                    plt.legend(handles=legend_patches, **legend_kwargs)
            else:
                plt.scatter(voom_plot_data['xy_x'], voom_plot_data['xy_y'],
                            s=point_size, c=point_color, **scatter_kwargs)
                plt.plot(voom_plot_data['line_x'], voom_plot_data['line_y'],
                         c=line_color, linewidth=line_width, **plot_kwargs)
            if xlabel_kwargs is None:
                xlabel_kwargs = {}
            if ylabel_kwargs is None:
                ylabel_kwargs = {}
            plt.xlabel(xlabel, **xlabel_kwargs)
            plt.ylabel(ylabel, **ylabel_kwargs)
            if title is not False:
                if title_kwargs is None:
                    title_kwargs = {}
                # noinspection PyCallingNonCallable
                plt.title(title[cell_type] if isinstance(title, dict)
                          else title if isinstance(title, str) else
                          title(cell_type) if isinstance(title, Callable) else
                          cell_type, **title_kwargs)
            if despine:
                spines = plt.gca().spines
                spines['top'].set_visible(False)
                spines['right'].set_visible(False)
            plt.savefig(voom_plot_file, **savefig_kwargs)
            plt.close()
