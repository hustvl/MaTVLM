## Evaluating base models

Please check [llm_eval](https://github.com/jxiw/MambaInLlama/tree/main/benchmark/llm_eval) to evaluate base models.

## Evaluating chat models

Please check [alpaca_eval](https://github.com/jxiw/MambaInLlama/tree/main/benchmark/alpaca_eval) and [mt_bench](https://github.com/jxiw/MambaInLlama/tree/main/benchmark/mt_bench) to evaluate chat models.

Please check [zero_eval](https://github.com/jxiw/ZeroEval) to evaluate chat models in zero shot.

## Speed

PYTHONPATH=. TRITON_CACHE_DIR=/data/yingyueli/.triton/autotune bash benchmark/tinyllava_speed.sh 2>&1 | tee -a output.txt