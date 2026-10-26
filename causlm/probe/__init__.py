"""CauProbe-Edge: decision-relevant acquisition of causal knowledge on a gateway.

The gateway chooses, at every step, between acting on what it believes, asking an
on-device language model to read the commissioning text, running a short safe
experiment on the plant, or falling back to a verified controller. All four are
priced in the same reward units and ranked by the value of the information they
return for the decisions that remain.
"""
