# Explicit implementation provenance

Run-local model.py, token_alignment.py, methods.py, runtime.py candidate.py, framework_base.py, generator_base.py, base_model.py, rf_native.py, input_base.py, view_base.py, candidate_base.py and encoder_base.py copied explicitly from candidate_accept_v3_51e6680d; architecture and source/neighbor helpers unchanged. No sys.path, implicit import or runtime dependency on /home/rbf/TA-WF. runtime only changes STATUS prefix. New routing.py/worker.py/frontier.py/audit.py/controller.py implement this registered experiment.

Source checkpoint: pure_source_backbone_c1d249f6, 18,553 pure rows, three seeds. Immutable diagnostic caches: multifactor_diagnostic_v1_4fc8f13d. Gaussian source bank: gmm_routing_lora_v2_28dc7c4b. Source anchors/calibration and controls: candidate_accept_v3_51e6680d. Reused controls are historical outputs, not independent repetitions. Raw file SHA and old delivery audit are frozen before scoring.
