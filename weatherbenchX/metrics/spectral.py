# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Spectral metrics for WeatherBench-X."""

import os
import subprocess
import tempfile
from typing import Hashable, Mapping, Optional

from absl import logging
import numpy as np
import xarray as xr
from weatherbenchX.metrics import base

try:
  from cdo import Cdo
except ImportError:
  Cdo = None


def find_cdo_binary(cdo_path: Optional[str] = None) -> str:
  """Finds the CDO binary in the environment."""
  if cdo_path and os.path.exists(cdo_path):
    return cdo_path

  # Check standard locations or environment-specific ones
  # We look for a .conda_env in the current or parent directory
  cwd = os.getcwd()
  possible_paths = [
      os.path.join(cwd, 'experiments', '.conda_env', 'bin', 'cdo'),
      os.path.join(cwd, '.conda_env', 'bin', 'cdo'),
      os.path.join(os.path.dirname(cwd), 'experiments', '.conda_env', 'bin', 'cdo'),
  ]

  for path in possible_paths:
    if os.path.exists(path):
      return path

  return 'cdo'


class KESpectrum(base.Statistic):
  """Kinetic Energy Spectrum statistic.

  Computes the Kinetic Energy (KE) spectrum as a function of spherical harmonic
  wavenumber l.
  """

  def __init__(
      self,
      u_name: str = 'u_component_of_wind',
      v_name: str = 'v_component_of_wind',
      cdo_path: Optional[str] = None,
      earth_radius: float = 6371000.0,
      truncation: Optional[int] = None,
      grid: str = 'n80',
  ):
    """Init.

    Args:
      u_name: Name of the u wind component.
      v_name: Name of the v wind component.
      cdo_path: Path to the CDO binary. If None, it will try to find it in PATH.
      earth_radius: Radius of the Earth in meters.
      truncation: Triangular truncation T. If None, it will be determined from
        the output of CDO.
      grid: Gaussian grid to remap to before computing uv2dv (e.g., 'n80').
    """
    self._u_name = u_name
    self._v_name = v_name
    self._earth_radius = earth_radius
    self._truncation = truncation
    self._grid = grid
    self._cdo_path_arg = cdo_path
    self._cdo_path = None

  @property
  def cdo_path(self) -> str:
    """Discovers and returns the CDO binary path."""
    if self._cdo_path is None:
      self._cdo_path = find_cdo_binary(self._cdo_path_arg)
    return self._cdo_path

  @property
  def unique_name(self) -> str:
    return f'KESpectrum_{self._u_name}_{self._v_name}_{self._grid}'

  def compute(
      self,
      predictions: Mapping[Hashable, xr.DataArray],
      targets: Mapping[Hashable, xr.DataArray],
  ) -> Mapping[Hashable, xr.DataArray]:
    """Computes KE spectrum for predictions and targets."""
    out = {}
    pred_ke = self._compute_ke_spectrum(predictions)
    if pred_ke is not None:
      out['pred_ke_spectrum'] = pred_ke

    target_ke = self._compute_ke_spectrum(targets)
    if target_ke is not None:
      out['target_ke_spectrum'] = target_ke

    return out

  def _prepare_dataset_for_cdo(
      self, u: xr.DataArray, v: xr.DataArray
  ) -> tuple[xr.Dataset, list[str]]:
    """Prepares u and v DataArrays for CDO processing."""
    ds = xr.Dataset({'u': u, 'v': v})

    # Standardize spatial dimensions
    spatial_dims = ['latitude', 'longitude', 'lat', 'lon']
    rename_dict = {}
    if 'latitude' in ds.coords:
      rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords:
      rename_dict['longitude'] = 'lon'
    if rename_dict:
      ds = ds.rename(rename_dict)

    # Ensure decreasing latitude (required by some CDO operators)
    if ds.lat[0] < ds.lat[-1]:
      ds = ds.reindex(lat=ds.lat[::-1])

    # Add required CF-compliant attributes
    ds.lat.attrs = {'units': 'degrees_north', 'standard_name': 'latitude'}
    ds.lon.attrs = {'units': 'degrees_east', 'standard_name': 'longitude'}
    ds.u.attrs = {'units': 'm s-1', 'standard_name': 'eastward_wind'}
    ds.v.attrs = {'units': 'm s-1', 'standard_name': 'northward_wind'}

    # Stack non-spatial dimensions into a single 'time' dimension for CDO
    non_spatial_dims = [d for d in ds.dims if d not in ['lat', 'lon']]

    if not non_spatial_dims:
      ds_for_cdo = ds.expand_dims('time').assign_coords(time=[0])
    elif len(non_spatial_dims) == 1:
      dim = non_spatial_dims[0]
      if dim != 'time':
        ds_for_cdo = ds.rename({dim: 'time'})
      else:
        ds_for_cdo = ds
    else:
      # Multiple non-spatial dims. To avoid 'time' conflict during stacking:
      temp_ds = ds
      stack_dims = non_spatial_dims
      if 'time' in ds.dims:
        temp_ds = ds.rename({'time': '_original_time'})
        stack_dims = [d if d != 'time' else '_original_time' for d in non_spatial_dims]
      
      ds_for_cdo = temp_ds.stack(time=stack_dims)
      # Drop internal coordinates to avoid NetCDF writing issues with MultiIndex
      ds_for_cdo = ds_for_cdo.drop_vars(stack_dims)

    # Standardize 'time' to simple integers for CDO
    ds_for_cdo = ds_for_cdo.assign_coords(time=np.arange(len(ds_for_cdo.time)))

    return ds_for_cdo.transpose('time', 'lat', 'lon'), non_spatial_dims

  def _run_cdo_uv2dv(self, infile: str, outfile: str):
    """Executes CDO uv2dv operator as a subprocess."""
    cdo_bin = self.cdo_path
    cmd = [
        cdo_bin,
        '-s',  # silent
        '-f', 'nc4',
        '-uv2dv',
        f'-remapbil,{self._grid}',
        infile,
        outfile
    ]

    env = os.environ.copy()
    cdo_dir = os.path.dirname(cdo_bin)
    if cdo_dir:
      env['PATH'] = cdo_dir + os.pathsep + env.get('PATH', '')
      # On Linux, ensure libraries are found if CDO is in a custom environment
      env['LD_LIBRARY_PATH'] = (
          os.path.join(cdo_dir, '..', 'lib')
          + os.pathsep
          + env.get('LD_LIBRARY_PATH', '')
      )

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
      logging.error('CDO Error (return code %d):\n%s',
                    result.returncode, result.stderr)
      raise RuntimeError(f'CDO execution failed: {result.stderr}')

  def _calculate_ke_from_spectral(self, ds_sp: xr.Dataset) -> xr.DataArray:
    """Calculates KE spectrum from spectral coefficients."""
    # sd: divergence, svo: vorticity
    sd = ds_sp.sd
    svo = ds_sp.svo
    nsp = ds_sp.sizes['nsp']

    # Determine truncation T from number of spectral coefficients
    # nsp = (T+1)*(T+2) for complex coeffs in CDO format
    T = int((-3 + np.sqrt(1 + 8 * nsp)) / 2)
    if self._truncation is not None:
      T = self._truncation

    # Magnitude squared: |coeff|^2 = real^2 + imag^2
    # CDO stores real/imag in 'nc2' dimension
    sd_mag_sq = (sd.sel(nc2=0)**2 + sd.sel(nc2=1)**2)
    svo_mag_sq = (svo.sel(nc2=0)**2 + svo.sel(nc2=1)**2)
    combined_mag_sq = sd_mag_sq + svo_mag_sq

    # Map each spectral index to its total wavenumber (l) and compute summation factors
    ls = np.zeros(nsp, dtype=int)
    factors = np.zeros(nsp, dtype=float)
    for m in range(T + 1):
      for l in range(m, T + 1):
        idx = int(m * (T + 1) - m * (m - 1) / 2 + (l - m))
        if idx < nsp:
          ls[idx] = l
          # CDO's uv2dv output uses a specific normalization
          # We need a factor of 2 for m > 0 because of symmetry
          factors[idx] = 1.0 if m == 0 else 2.0

    # Weighted sum over m for each l
    combined_weighted = combined_mag_sq * xr.DataArray(
        factors, dims=['nsp'], coords={'nsp': np.arange(nsp)})
    combined_weighted = combined_weighted.assign_coords(wavenumber=('nsp', ls))

    # Sum across all indices for each total wavenumber l
    l_sums = combined_weighted.groupby('wavenumber').sum('nsp')

    # Apply Kinetic Energy formula: KE(l) = [r^2 / (2 * l * (l+1))] * Sum_m (|sd|^2 + |svo|^2)
    wavenumbers = l_sums.wavenumber.values
    coeffs = np.zeros_like(wavenumbers, dtype=float)
    valid_l = wavenumbers > 0
    coeffs[valid_l] = (self._earth_radius**2) / (
        2 * wavenumbers[valid_l] * (wavenumbers[valid_l] + 1))

    ke_spectrum = l_sums * xr.DataArray(
        coeffs, dims=['wavenumber'], coords={'wavenumber': wavenumbers})
    ke_spectrum.name = 'ke_spectrum'
    return ke_spectrum

  def _compute_ke_spectrum(
      self, data: Mapping[Hashable, xr.DataArray]
  ) -> Optional[xr.DataArray]:
    """Internal method to compute KE spectrum from a mapping of variables."""
    if self._u_name not in data or self._v_name not in data:
      return None

    u = data[self._u_name]
    v = data[self._v_name]

    try:
      # 1. Prepare data
      ds_for_cdo, non_spatial_dims = self._prepare_dataset_for_cdo(u, v)

      with tempfile.TemporaryDirectory() as tmpdir:
        infile = os.path.join(tmpdir, 'input.nc')
        outfile = os.path.join(tmpdir, 'spectral.nc')

        # 2. Run CDO
        # Use NETCDF3_64BIT for input for maximum compatibility with CDO versions
        ds_for_cdo.fillna(0.0).to_netcdf(infile, format='NETCDF3_64BIT')
        self._run_cdo_uv2dv(infile, outfile)

        # 3. Post-process
        with xr.open_dataset(outfile) as ds_sp:
          # Force load as we are in a temporary directory
          ds_sp.load()
          ke_spectrum_stacked = self._calculate_ke_from_spectral(ds_sp)

      # 4. Restore original dimensions
      if not non_spatial_dims:
        return ke_spectrum_stacked.isel(time=0, drop=True)

      # Reconstruct the original dimensions from the 'time' dimension
      # We need the original coords for non-spatial dims
      original_coords = {d: u.coords[d] for d in non_spatial_dims if d in u.coords}

      if len(non_spatial_dims) == 1:
        dim = non_spatial_dims[0]
        return ke_spectrum_stacked.rename({'time': dim}).assign_coords(
            {dim: u.coords[dim] if dim in u.coords else np.arange(u.sizes[dim])}
        )

      # For multiple non-spatial dims, we need to unstack
      # Use a unique name for the temporary stacked dimension to avoid conflicts with coord names
      temp_stack_dim = '_stacked_dims'
      multi_index = xr.Dataset(coords=original_coords).stack(
          **{temp_stack_dim: non_spatial_dims}
      )[temp_stack_dim]
      return ke_spectrum_stacked.rename({'time': temp_stack_dim}).assign_coords(
          {temp_stack_dim: multi_index}
      ).unstack(temp_stack_dim)

    except Exception as e:  # pylint: disable=broad-except
      logging.error('Error computing KESpectrum: %s', e)
      return None
