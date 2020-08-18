## Documentation for two pupil input-output datasets with pupil-coding SLM system ##

The two datasets are:
1. Y:\Regina\slm_calibration_2020\pupil_pair_dataset_uncorrectedForObjectiveAber.mat
2. Y:\Regina\slm_calibration_2020\pupil_pair_dataset_correctedForObjectiveAber.mat

Both datasets contain input-output pupil pairs, where the input pupil is the pupil 
phase displayed on the SLM and the output pupil is reconstructed with 250 iterations
of Fourier ptychography optimization.

The two datasets are the same except that in dataset #2 ('correctedForObjectiveAber') 
all output pupils have had the objective's pupil phase subtracted from the reconstructed
SLM pupil phase. Accordingly, the 'noCoding' pupils (pupil pairs where the SLM displayed 
zero phase, which reconstructs the objective lens' pupil function) have been removed from
dataset #2.

Dataset #1 has n_img = 30, since it includes the 'noCoding' pupils (where the SLM displayed
phase = 0 radians). Dataset #2 has n_img = 25, since it excludes the 'noCoding' pupils.

Contained in dataset #2 only is:
- output_pupil_objCorrected_data: output pupil amplitude and phase, with objective lens
					 pupil phase aberrations subtracted          [500,500,n_img] (complex)

The data contained in both datsets are:
- input_pupil_data:  input pupil amplitude and phase,            [500,500,n_img] (complex)
- input_phase_data:  input pupil phase, wrapped 0 to 2pi,        [500,500,n_img] (double)
- input_slm_data:    input pupil phase in slm values (0 to 255), [500,500,n_img] (double)
- output_pupil_data: output pupil amplitude and phase, with no subtraction of objective 
					 lens pupil aberrations                      [500,500,n_img] (complex)

- base_pupil_support: logical pupil support mask for NA 0.8      [500,500] (logical)
- base_filename_list: cell array of filenames, containing identifying information
- base_dir_list:      cell array of base directory names, helpful for sorting data
- pupil_sigma_data:   matrix of [sigma_y; sigma_x] Gaussian blur std dev in y and x for 
					  each pupil 								 [2,n_img] (double)
- pupil_type_id:      logical matrix of identifying information, showing whether each pupil
					  is 'noCoding' (row 1), 'slmCalibration' (row 2), 'slmDefocus' (row 3),
					  'slmRandom' (row 4), 'slmZernike' (row 5)  [5,n_img] (double)

- metadata: Struct containing following values:
-- pupil_type_label: cell array of labels for pupil_type_id matrix
-- shift_amt:        pixel shift applied to the input data from each base directory to align 
					 to outputs
-- NA:               system NA
-- mag:              system magnification
-- lambda:           wavelength (um)
-- dpix_c:           camera pixel size (um)
-- sz_recon:         size of reconstructed images
-- sz_slm:           native SLM size
-- du_cam:           frequency space pixel extent of reconstruction (1/um)
-- du_slm:           frequency space pixel extent of native SLM (1/um)


Understanding pupil types:
- 'noCoding': SLM displays phase 0 radians; reconstructed pupil should therefore contain the 
			  objective lens' pupil aberrations only
- 'slmCalibration': A special calibration pattern used for determining SLM shift. Contains 
			  piecewise constant phase values.
- 'slmDefocus': Fresnel defocus kernels for randomly selected distances
- 'slmPixel':   Uniform random noise that has been smoothed
- 'slmZernike': Randomly weighted Zernike polynomials 
