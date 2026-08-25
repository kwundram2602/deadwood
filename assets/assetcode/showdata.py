import numpy as np
import rioxarray
import matplotlib.pyplot as plt
from matplotlib import colormaps


# MS = "/home/kjell/projects/py_projects/InnoLabDL/deadwood/datafiles/process_out/images/20260313_Airport_Main_MAVICM3MFIXEDM3M_OM_MS_merged_stack.tif"
MS = "/home/kjell/projects/py_projects/InnoLabDL/deadwood/datafiles/OM_domAligned/20260313_Airport_Main_MAVICM3MFIXEDM3M_OM_coreg.tif"

# Keep this low enough for 3D plotting. 3D surfaces scale poorly with resolution.
MAX_PLOT_SIDE = 1600
MAX_PLOT_PIXELS = 1_800_000
USE_DASK = True
CHUNK_SIZE = 1024


def _compute_stride(height: int, width: int, max_side: int, max_pixels: int) -> int:
	if max_side <= 0:
		raise ValueError("max_side must be > 0")
	if max_pixels <= 0:
		raise ValueError("max_pixels must be > 0")

	stride_side = int(np.ceil(max(height, width) / max_side))
	stride_pixels = int(np.ceil(np.sqrt((height * width) / max_pixels)))
	return max(1, stride_side, stride_pixels)


open_kwargs = {"masked": False}
if USE_DASK:
	open_kwargs["chunks"] = {"band": 1, "y": CHUNK_SIZE, "x": CHUNK_SIZE}

try:
	da = rioxarray.open_rasterio(MS, **open_kwargs)
except Exception as exc:
	if USE_DASK and "chunks" in open_kwargs:
		print(f"Chunked open failed ({exc}); retry without chunks.")
		open_kwargs.pop("chunks", None)
		da = rioxarray.open_rasterio(MS, **open_kwargs)
	else:
		raise

band_names = ["red", "green", "blue", "green_ms", "red_ms", "rededge", "nir"]
if da.sizes["band"] != len(band_names):
	raise ValueError(f"Expected {len(band_names)} bands, got {da.sizes['band']}")

height, width = da.sizes["y"], da.sizes["x"]
stride = _compute_stride(height, width, MAX_PLOT_SIDE, MAX_PLOT_PIXELS)

if da.rio.nodata is not None and np.isfinite(da.rio.nodata):
	da_for_plot = da.where(da != da.rio.nodata)
else:
	da_for_plot = da.where(da != 0)

if stride > 1:
	# Coarsened averaging looks smoother than nearest-neighbor skipping.
	plot_da = da_for_plot.coarsen(y=stride, x=stride, boundary="trim").mean(skipna=True)
else:
	plot_da = da_for_plot

plot_height, plot_width = plot_da.sizes["y"], plot_da.sizes["x"]
sample_y = np.linspace(0, 1, plot_height, dtype=np.float32)
sample_x = np.linspace(0, 1, plot_width, dtype=np.float32)
x_grid, y_grid = np.meshgrid(sample_x, sample_y)

stack_shift = 0.14
stack_depth = 0.08
stack_tilt = 0.22

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")
band_colormaps = {
	"red": "Reds",
	"green": "Greens",
	"blue": "Blues",
	"green_ms": "YlGn",
	"red_ms": "YlOrRd",
	"rededge": "Oranges",
	"nir": "magma",
}

for band_index, band_name in enumerate(band_names):
	values = plot_da.isel(band=band_index).values.astype(np.float32)
	valid = np.isfinite(values)

	finite_values = values[valid]
	if finite_values.size == 0:
		continue

	lower, upper = np.percentile(finite_values, [2, 98])
	normalized = np.clip((values - lower) / (upper - lower + 1e-12), 0, 1)
	normalized[~valid] = 0

	colors = colormaps[band_colormaps[band_name]](normalized).astype(np.float32)
	colors[..., 3] = valid

	shifted_y = y_grid + band_index * stack_shift
	tilted_z = band_index * stack_depth + shifted_y * stack_tilt
	tilted_z = np.ma.masked_where(~valid, tilted_z)

	ax.plot_surface(
		x_grid,
		shifted_y,
		tilted_z,
		facecolors=colors,
		linewidth=0,
		antialiased=True,
		shade=False,
	)

ax.view_init(elev=24, azim=-62)
ax.set_box_aspect((1, 1 + (len(band_names) - 1) * stack_shift, 1.8))
ax.set_axis_off()
ax.set_facecolor((0, 0, 0, 0))
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
	axis.pane.fill = False
	axis.pane.set_edgecolor((0, 0, 0, 0))

fig.patch.set_alpha(0)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
fig.savefig(
	"assets/showdata_3d.png",
	dpi=500,
	bbox_inches="tight",
	pad_inches=0,
	transparent=True,
)

print(
	f"Saved assets/showdata_3d.png using stride={stride} "
	f"(source={height}x{width}, plotted={plot_height}x{plot_width})"
)