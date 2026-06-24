from skopt.space import Real, Integer, Categorical

space = [
    Real(1e-5, 5e-3, prior="log-uniform", name="learningRate"),
    Real(0.0, 0.4, name="dropout_rate"),
    Real(0.0, 0.6, name="dropout_rate_dense"),
    Real(0.0, 0.8, name="dropout_rate_psf"),
    Integer(512, 4000, name="n_units_dense"),
     #Categorical([3, 5, 7], name="ksz_enc"),
    #Categorical([3, 5], name="ksz_psf"),
    #Categorical([3, 5], name="ksz_wf"),
     #Categorical([64, 96, 128], name="nfilts_enc"),
    #Categorical([32, 64, 96], name="nfilts_psf"),
    #Categorical([32, 64, 96], name="nfilts_wf"),
     #Categorical(["relu", "elu", "gelu"], name="actFunc"),
]
