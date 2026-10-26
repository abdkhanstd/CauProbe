"""What the trained expert reads from a site description of bestest_air.

The text paraphrases the BOPTEST documentation of the test case (it is not
generated from the notes vocabulary used to train or test the expert), so this
is one genuinely out-of-distribution reading.
"""
import json

from cp_common import BASE_05, CP_RESULTS, LORA_DIR
from causlm.probe.slm_expert import load_model, score_texts

TEXT = ("Single-zone test office based on the BESTEST Case 900 definition, 6 m by "
        "8 m with a 2.7 m ceiling. Exterior walls are concrete block with "
        "insulation and the floor is a concrete slab; the roof is a wood frame "
        "with insulation. Two people work there from 8am to 6pm. A four-pipe fan "
        "coil unit provides heating and cooling.")

tok, model = load_model(BASE_05, str(LORA_DIR))
z = score_texts(tok, model, [TEXT])[0]
(CP_RESULTS / "expert_outputs_boptest.json").write_text(
    json.dumps({"text": TEXT, "z": z}, indent=1))
print(z)
