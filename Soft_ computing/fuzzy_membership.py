import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def create_fuzzy_variables():
    # Inputs
    breathlessness = ctrl.Antecedent(np.arange(0, 11, 1), 'breathlessness')
    wheezing = ctrl.Antecedent(np.arange(0, 11, 1), 'wheezing')
    cough = ctrl.Antecedent(np.arange(0, 11, 1), 'cough')

    # Output
    asthma_severity = ctrl.Consequent(np.arange(0, 11, 1), 'asthma_severity')

    # Membership functions
    for var in [breathlessness, wheezing, cough]:
        var['low'] = fuzz.trimf(var.universe, [0, 0, 4])
        var['medium'] = fuzz.trimf(var.universe, [3, 5, 7])
        var['high'] = fuzz.trimf(var.universe, [6, 10, 10])

    asthma_severity['mild'] = fuzz.trimf(asthma_severity.universe, [0, 0, 4])
    asthma_severity['moderate'] = fuzz.trimf(asthma_severity.universe, [3, 5, 7])
    asthma_severity['severe'] = fuzz.trimf(asthma_severity.universe, [6, 10, 10])

    return breathlessness, wheezing, cough, asthma_severity
