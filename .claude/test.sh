HF_HUB_DISABLE_XET=1 with-proxy pytest -v -s tests/v1/e2e/test_correctness_sliding_window.py
HF_HUB_DISABLE_XET=1 with-proxy pytest -v -s tests/v1/engine/test_engine_args.py::test_defaults_with_usage_contex
HIP_VISIBLE_DEVICES=2 HF_HUB_DISABLE_XET=1 VLLM_ROCM_USE_AITER=1 with-proxy pytest -v -s tests/v1/sample/test_sampling_params_e2e.py::test_bad_words
HF_HUB_DISABLE_XET=1 with-proxy pytest -v -s tests/samplers/test_ranks.py
HF_HUB_DISABLE_XET=1 wp pytest -s tests/entrypoints/openai/correctness/test_transcription_api_correctness.py
    - pytest -v -s compile/test_noop_elimination.py
    - pytest -v -s compile/test_aot_compile.py
    compile/test_basic_correctness.py
    tests/entrypoints/openai/correctness/test_transcription_api_correctness.py::test_wer_correctness


 HF_HUB_DISABLE_XET=1 with-proxy pytest -v -s tests/models/multimodal/generation/test_common.py::test_single_image_models
 HF_HUB_DISABLE_XET=1 wp pytest -v -s  'tests/entrypoints/openai/test_transcription_validation.py::test_basic_audio
 HF_HUB_DISABLE_XET=1 wp pytest -v -s tests/entrypoints/openai/test_optional_middleware.py
 HF_HUB_DISABLE_XET=1 wp pytest -v -s 'tests/tool_use/test_tool_calls.py::test_tool_call_and_choice[granite-3.0-8b]'


HF_HUB_DISABLE_XET=1 wp pytest -s -v tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py::test_streaming_tool_call

HF_HUB_DISABLE_XET=1 wp pytest -s -v 'tests/tool_use/test_tool_calls.py::test_tool_call_and_choice[granite-3.0-8b]'

HF_HUB_DISABLE_XET=1 wp pytest -s -v test_lm_eval_correctness.py --config-list-file=configs/models-large-rocm.txt --tp-size=8
