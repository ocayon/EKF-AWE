#%% Dictionary of existing kite models
# Configuration dictionary for kite models
kite_models = {
    "v3": {
        "KCU": True,
        "mass": 15,
        "area": 19.75,
        "distance_kcu_kite": 11.5,
        "total_length_bridle_lines": 96,
        "diameter_bridle_lines": 2.5e-3,
    },
    "v9": {
        "KCU": True,
        "mass": 62,
        "area": 46.854,
        "distance_kcu_kite": 15.45,
        "total_length_bridle_lines": 300,
        "diameter_bridle_lines": 4e-3,
    },
    "custom": {
        "KCU": True,
        "mass": 25,
        "area": 30,
        "distance_kcu_kite": 10,
        "total_length_bridle_lines": 120,
        "diameter_bridle_lines": 3e-3,
    },
}
tether_materials = {
    "Dyneema-SK78": {
        "density": 970,
        "cd": 1.1,
        "Youngs_modulus": 132e9,
    },
    "Dyneema-SK75": {
        "density": 970,
        "cd": 1.1,
        "Youngs_modulus": 109e9,
        }
}
kcu_cylinders = {
    "KP1": {
        "length": 1,
        "diameter": 0.48,
        "mass": 18 + 1.6 + 8,
    },
    "KP2": {
        "length": 1.2,
        "diameter": 0.62,
        "mass": 18 + 1.6 + 12,
    },
    "custom": {
        "length": 1.2,
        "diameter": 0.62,
        "mass": 18 + 1.6 + 12,
    },
}