from skfuzzy import control as ctrl
from fuzzy_membership import create_fuzzy_variables

def create_asthma_rules():
    breathlessness, wheezing, cough, asthma_severity = create_fuzzy_variables()

    rules = [
        ctrl.Rule(breathlessness['high'] & wheezing['high'],
                  asthma_severity['severe']),

        ctrl.Rule(cough['medium'] & wheezing['low'],
                  asthma_severity['mild']),

        ctrl.Rule(breathlessness['medium'] & wheezing['medium'],
                  asthma_severity['moderate']),

        ctrl.Rule(breathlessness['low'] & wheezing['low'] & cough['low'],
                  asthma_severity['mild'])
    ]

    return rules
