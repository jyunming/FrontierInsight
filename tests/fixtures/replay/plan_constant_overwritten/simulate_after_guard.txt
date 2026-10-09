        absorbed=rng.binomial(incident,absorption)
        transmitted=incident-absorbed
        partition_error=np.abs(incident-absorbed-transmitted)
        mean_absorbed=float(np.mean(absorbed))
        var_absorbed=float(np.var(absorbed,ddof=1)) if n_trials > 1 else 0.0
        mean_incident=float(np.mean(incident))
        return {"mean_absorbed_photons":mean_absorbed,"var_absorbed_photons":var_absorbed,"mean_incident_photons":mean_incident,"absorbed_count":float(absorbed.sum()),"incident_count":float(incident.sum()),"transmitted_count":float(transmitted.sum()),"photon_partition_error":float(np.max(partition_error)),"fraction_zero_absorption":float(np.mean(absorbed==0)),"max_abs_incident_minus_absorbed_minus_transmitted":float(np.max(partition_error)),"expected_absorbed":expected}
    grid_nm=.5;shape=(2000,96) if pattern=="lines" else (128,128);aerial=_aerial(pattern,shape,grid_nm,sensitivity)
    # E1
    photon_mean=model.photon_density(dose)*aerial*grid_nm*grid_nm;incident=rng.poisson(photon_mean);absorbed=rng.binomial(incident,absorption)
    reference_dose=float(cell["exposure_dose"])
    reference_photon_mean=model.photon_density(reference_dose)*aerial*grid_nm*grid_nm
    if chemistry=="off":proxy=absorbed.astype(float);threshold=float(np.quantile(reference_photon_mean*absorption,.62))
    elif resist=="CAR":
        acid=rng.poisson(absorbed*yield_mean).astype(float);proxy=ndimage.gaussian_filter(acid,electron_blur/grid_nm,mode="nearest");proxy=ndimage.gaussian_filter(proxy,diffusion_blur/grid_nm,mode="nearest")
        qrng=np.random.default_rng(np.random.SeedSequence([int(cell.get("quencher_field_seed",0)),trial_id]));q=qrng.poisson(quencher*reference_photon_mean*absorption*yield_mean).astype(float);proxy-=q;threshold=.55*reference_dose*absorption*yield_mean*grid_nm*grid_nm
    else:proxy=ndimage.gaussian_filter(absorbed.astype(float),electron_blur/grid_nm,mode="nearest");threshold=.48*dose*absorption*grid_nm*grid_nm
