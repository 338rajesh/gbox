""" Generating raw data using UnitCellGenerator.jl """

using UnitCellGenerator
using NPZ
using HDF5

# ===============================================
#               Input data
# ===============================================
const rve_size = 25.0
const SSDratio = 0.15
const r_mean = 5.0
const r_std = 0.0
const VERBOSE = 0
const working_dir = @__DIR__
const num_samples = 20
const fvf_c = 0.5 * ones(Float64, num_samples)  # fibre volume fractions of circles
const fvf_e = 0.0 * ones(Float64, num_samples)  # fibre volume fractions of ellipse


TARGET_RAW_DATA_FILE = joinpath(working_dir, "rve_raw_data_circular_non_periodic.h5")

# -------------
bbox_ruc = ((-0.5, -0.5, 0.5, 0.5) .* (rve_size * r_mean))
#
rve_info = RUC_data(
	bbox=BBox2D(bbox_ruc...,),
	ssd_ratio=SSDratio,
	inclusion_distr=RANDOM,
	periodicity=false,
)
for (i, a_fvf) in enumerate(fvf_c)
	fvf_total = fvf_c[i] + fvf_e[i]
	print("Generating $i/$(num_samples) RVE:")
	ith_vf_tag = join([i=='.' ? "p" : i for i in string(fvf_total)])
	inclusion_information = (
		Inclusion_data(
		volume_fraction=fvf_c[i],
		shape=Circle,
		size_params=Dict(:RADIUS => Normal(r_mean, r_std),),
		),
		Inclusion_data(
		volume_fraction=fvf_e[i],
		shape=Ellipse,
		size_params=Dict(:EQRAD => Normal(r_mean, r_std), :ASPRA => Normal(2.0, 0.0)),
		),
	)
	# ===============================================
	#               GENERATION
	# ===============================================
	ruc = generate_unit_cell(
		rve_info,
		inclusion_information,
		max_num_iterations=2000,
		max_num_fg_evaluations=5000,
		non_monotone_memory=50,
		verbose=VERBOSE,
		adjust_ruc_bbox=true,
	)
	# ===============================================
	#               SAVING TO .h5
	# ===============================================
	arves_path = string(/, "N$(i)-VF$(ith_vf_tag)")
	h5open(TARGET_RAW_DATA_FILE, isfile(TARGET_RAW_DATA_FILE) ? "r+" : "w") do h5file_id
		for (k, v) in ruc
			if k != "bbox"
				write(h5file_id, "$arves_path/$k", v)
			end
		end
		#
		rvesg = h5file_id[arves_path]
		attributes(rvesg)["eqrad"] = r_mean
		attributes(rvesg)["ivf"] = fvf_total
		attributes(rvesg)["periodic"] = rve_info.periodicity
		attributes(rvesg)["xlb"] = ruc["bbox"][1]
		attributes(rvesg)["ylb"] = ruc["bbox"][2]
		attributes(rvesg)["xub"] = ruc["bbox"][3]
		attributes(rvesg)["yub"] = ruc["bbox"][4]
	end
	println("\tDone!")
end
