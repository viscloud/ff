
## Performance experiments
1. Build the FilterForward code following the instructions [here](https://github.com/viscloud/filterforward/tree/master/cpp#building).
2. Navigate to the `build` directory where the compiled code is located.
3. Edit [line 40](https://github.com/viscloud/filterforward/blob/master/scripts/e2e/configs/saf_config/cameras.toml#L40) of `e2e/configs/saf_config/cameras.toml` to point to any long H.264-encoded 1080p video file,
4. Edit [line 18](https://github.com/viscloud/filterforward/blob/master/scripts/e2e/run_all.sh#L18) of `e2e/run_all.sh` so that the `FF_SCRIPTS_PATH` variable points to the `e2e` directory.
5. Uncomment lines in `e2e/run_all.sh` to specify which experiments to run.
6. Execute `e2e/run_all.sh`. ***Note:*** The directory `~/out` will be created and output files will be stored there.
7. Examine the contents of `~/out/ff` and/or `~/out/nnbench`. In each results file, each line corresponds to metrics recorded when a frame emerged from the end of the pipeline. To prevent log overflow, the FilterForward experiments do not log all frames. The first entry of each line is the average frame rate up to that point in the experiment (the number of frames that have emerged from the pipeline divided by the elasped time of the experiment).

## Accuracy, bandwidth, and compute cost (multiply-adds) experiments

[Contact the authors](https://github.com/viscloud/filterforward#contact) for instructions on how to run all other experiments.
